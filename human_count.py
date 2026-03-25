import os
import logging
import requests
import cv2
import numpy as np

from concurrent.futures import Future
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form
from queue import Queue, Full
from threading import Thread
from ultralytics import YOLO

load_dotenv()

# ================= CONFIG =================
WEBHOOK_URL        = os.getenv("WEBHOOK_URL", "https://sumsel.smart-gateway.net/api/webhook/detection")
HC_SAVE_LIMIT      = int(os.getenv("SAVE_LIMIT", 300))   # 0 = unlimited
HC_WORKER_COUNT    = int(os.getenv("WORKER", 2))
HC_YOLO_MODEL      = os.getenv("HC_YOLO_MODEL", "yolov8n.pt")
HC_CONFIDENCE      = int(os.getenv("HC_CONFIDENCE", 50))     # 1-100
HC_MODE_CCTV       = os.getenv("HC_MODE_CCTV", "true").lower() == "true"

# ================= DIRS =================
HC_DIR = "human_count"
os.makedirs(HC_DIR, exist_ok=True)

# ================= LOGGER =================
logger = logging.getLogger("human_count")

# ================= YOLO MODEL =================
yolo_model = YOLO(HC_YOLO_MODEL)

# ================= ROUTER =================
router = APIRouter()

# ================= QUEUES =================
hc_task_queue:    Queue = Queue(maxsize=500)
hc_webhook_queue: Queue = Queue(maxsize=500)


# ================= FILTERING PARAMS =================
def _get_filter_params(mode_cctv: bool) -> dict:
    if mode_cctv:
        return dict(MIN_W=15, MIN_H=30, MIN_AREA=3000, MIN_RATIO=0.18, MAX_RATIO=1.2)
    return dict(MIN_W=25, MIN_H=50, MIN_AREA=6000, MIN_RATIO=0.25, MAX_RATIO=0.85)


# ================= WEBHOOK WORKER =================
def hc_webhook_worker():
    logger.info("HC WEBHOOK WORKER STARTED")
    while True:
        item = hc_webhook_queue.get()
        if item is None:
            hc_webhook_queue.task_done()
            break

        try:
            frame_bytes, frame_name, human_count, channel_id, client_id, cctv_name, timestamp = item

            payload = {
                "status":     "success",
                "type":       "human_count",
                "data" : {
                    "person": human_count,
                    "channel_id": channel_id,
                    "client_id":  client_id,
                    "cctv_name":  cctv_name,
                },
                "timestamp" : timestamp,
            }

            resp = requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=payload,
                timeout=10,
            )
            logger.info(
                f"[HC WEBHOOK OK] {frame_name} | cctv={cctv_name} | "
                f"count={human_count} | status={resp.status_code}"
            )

        except Exception as e:
            logger.error(f"[HC WEBHOOK ERROR] {e}")
        finally:
            hc_webhook_queue.task_done()


# ================= FACE WORKER =================
def hc_face_worker(worker_id: int = 1):
    logger.info(f"HC WORKER #{worker_id} STARTED")
    yolo_conf = HC_CONFIDENCE / 100
    fp        = _get_filter_params(HC_MODE_CCTV)

    while True:
        task = hc_task_queue.get()
        if task is None:
            hc_task_queue.task_done()
            break

        img, channel_id, client_id, timestamp, cctv_name, future = task
        try:
            results = yolo_model.predict(img, conf=yolo_conf)

            human_count = 0
            bg_img = img.copy()

            for r in results:
                for box in r.boxes:
                    cls      = int(box.cls[0])
                    conf_det = float(box.conf[0])

                    if cls != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w     = x2 - x1
                    h     = y2 - y1
                    area  = w * h
                    ratio = w / h if h > 0 else 1

                    if w < fp["MIN_W"] or h < fp["MIN_H"]:
                        continue
                    if area < fp["MIN_AREA"]:
                        continue
                    if not (fp["MIN_RATIO"] < ratio < fp["MAX_RATIO"]):
                        continue
                    if conf_det < yolo_conf:
                        continue

                    human_count += 1
                    cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            ts         = timestamp.replace(":", "").replace("-", "").replace("T", "_")
            frame_name = f"{channel_id}_{ts}_bg.jpg"
            frame_path = os.path.join(HC_DIR, frame_name)

            # Simpan dengan limit per channel_id
            if HC_SAVE_LIMIT > 0:
                existing = sorted([
                    f for f in os.listdir(HC_DIR)
                    if f.startswith(channel_id + "_")
                ])
                if len(existing) >= HC_SAVE_LIMIT:
                    oldest = existing[0]
                    try:
                        os.remove(os.path.join(HC_DIR, oldest))
                        logger.info(f"[HC] Save limit reached, removed: {oldest}")
                    except FileNotFoundError:
                        pass

            cv2.imwrite(frame_path, bg_img)

            _, frame_buf = cv2.imencode(".jpg", bg_img)
            frame_bytes  = frame_buf.tobytes()

            logger.info(
                f"[HC] cctv={cctv_name} | channel={channel_id} | "
                f"count={human_count} | file={frame_name}"
            )

            if WEBHOOK_URL:
                try:
                    hc_webhook_queue.put_nowait((
                        frame_bytes, frame_name,
                        human_count,
                        channel_id, client_id,
                        cctv_name, timestamp,
                    ))
                except Full:
                    logger.warning(f"[HC] Webhook queue full, skipped: {frame_name}")

            # Kirim hasil ke endpoint yang menunggu
            future.set_result(human_count)

        except Exception as e:
            logger.error(f"HC WORKER #{worker_id} ERROR: {e}")
            future.set_exception(e)
        finally:
            hc_task_queue.task_done()


# ================= START WORKERS =================
for _i in range(HC_WORKER_COUNT):
    Thread(target=hc_face_worker, args=(_i + 1,), daemon=True).start()

if WEBHOOK_URL:
    Thread(target=hc_webhook_worker, daemon=True).start()


# ================= ENDPOINT =================
@router.post("/human-count")
async def human_count_detect(
    image_bg:   UploadFile = File(...),
    channel_id: str        = Form(...),
    client_id:  str        = Form(...),
    timestamp:  str        = Form(...),
    cctv_name:  str        = Form(...),
):
    image_bytes = await image_bg.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "decode_failed"}

    future: Future = Future()

    try:
        hc_task_queue.put_nowait((img, channel_id, client_id, timestamp, cctv_name, future))
    except Full:
        return {"status": "queue_full"}

    try:
        # Block (non-CPU) sampai worker selesai, timeout 30 detik
        import asyncio
        loop = asyncio.get_event_loop()
        human_count = await loop.run_in_executor(None, lambda: future.result(timeout=30))
    except Exception as e:
        logger.error(f"[HC] Detection error: {e}")
        return {"status": "error", "message": str(e)}

    return {
        "status":     "success",
        "type":       "human_count",
        "data" : {
            "person": human_count,
            "channel_id": channel_id,
            "client_id":  client_id,
            "cctv_name":  cctv_name,
        },
        "timestamp" : timestamp,
    }
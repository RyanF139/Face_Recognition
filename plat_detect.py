import os
import logging
import requests
import cv2
import numpy as np

from concurrent.futures import Future
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form
from queue import Queue, Full
from threading import Thread, Lock
from ultralytics import YOLO

load_dotenv()

# ================= CONFIG =================
WEBHOOK_URL        = os.getenv("WEBHOOK_URL", "")
PLATE_SAVE_LIMIT   = int(os.getenv("SAVE_LIMIT", 10))
PLATE_WORKER_COUNT = int(os.getenv("WORKER", 2))
PLATE_MODEL_PATH   = os.getenv("PLATE_MODEL", "best.pt")
PLATE_CONFIDENCE   = int(os.getenv("PLATE_CONFIDENCE", 25))

# ================= DIR =================
PLATE_DIR = "plate_image"
FRAME_DIR = os.path.join(PLATE_DIR, "frame")
CROP_DIR  = os.path.join(PLATE_DIR, "plate")

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# ================= LOGGER =================
logger = logging.getLogger("plate_crop")

# ================= MODEL =================
plate_model = YOLO(PLATE_MODEL_PATH)

# ================= ROUTER =================
router = APIRouter()

# ================= QUEUE =================
plate_task_queue: Queue = Queue(maxsize=500)
plate_webhook_queue: Queue = Queue(maxsize=500)

# ================= LOCK =================
_plate_channel_locks: dict[str, Lock] = {}
_plate_channel_meta = Lock()

def _get_channel_lock(channel_id: str) -> Lock:
    with _plate_channel_meta:
        if channel_id not in _plate_channel_locks:
            _plate_channel_locks[channel_id] = Lock()
        return _plate_channel_locks[channel_id]


# ================= CLEANUP =================
def cleanup_old_files(channel_id: str):
    """Pastikan hanya simpan N file terbaru"""
    existing = [
        f for f in os.listdir(FRAME_DIR)
        if f.startswith(channel_id + "_")
    ]

    if len(existing) <= PLATE_SAVE_LIMIT:
        return

    existing.sort(key=lambda f: os.path.getmtime(os.path.join(FRAME_DIR, f)))

    while len(existing) > PLATE_SAVE_LIMIT:
        oldest = existing.pop(0)
        prefix = oldest.replace(".jpg", "")

        try:
            os.remove(os.path.join(FRAME_DIR, oldest))
        except FileNotFoundError:
            pass

        for f in os.listdir(CROP_DIR):
            if f.startswith(prefix + "_"):
                try:
                    os.remove(os.path.join(CROP_DIR, f))
                except FileNotFoundError:
                    pass

        logger.info(f"[PLATE] Cleanup removed: {prefix}")


# ================= WEBHOOK WORKER =================
def plate_webhook_worker():
    logger.info("PLATE WEBHOOK WORKER STARTED")
    while True:
        item = plate_webhook_queue.get()
        if item is None:
            plate_webhook_queue.task_done()
            break

        try:
            event_id, frame_bytes, frame_name, total, channel_id, client_id, cctv_name, timestamp = item

            payload = {
                "status": "success",
                "type": "plate_crop",
                "event_id": event_id,
                "total_plate": total,
                "channel_id": channel_id,
                "client_id": client_id,
                "cctv_name": cctv_name,
                "timestamp": timestamp,
            }

            requests.post(
                WEBHOOK_URL,
                files=[("files", (frame_name, frame_bytes, "image/jpeg"))],
                data=payload,
                timeout=10
            )

            logger.info(f"[PLATE WEBHOOK] {frame_name} | total={total}")

        except Exception as e:
            logger.error(f"[PLATE WEBHOOK ERROR] {e}")
        finally:
            plate_webhook_queue.task_done()


# ================= WORKER =================
def plate_worker(worker_id: int = 1):
    logger.info(f"PLATE WORKER #{worker_id} STARTED")

    conf_thres = PLATE_CONFIDENCE / 100

    while True:
        task = plate_task_queue.get()
        if task is None:
            plate_task_queue.task_done()
            break

        event_id, img, channel_id, client_id, timestamp, cctv_name, future = task

        try:
            # Sama seperti code basic — tanpa conf di predict()
            results = plate_model.predict(img)

            draw_img = img.copy()
            crop_paths = []

            ts = str(timestamp)
            plate_index = 0

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    logger.info(f"[PLATE] Detected conf={conf:.2f} | threshold={conf_thres:.2f}")

                    if conf < conf_thres:
                        logger.warning(f"[PLATE] SKIPPED — conf={conf:.2f} < threshold={conf_thres:.2f}")
                        continue

                    logger.info(f"[PLATE] ACCEPTED — conf={conf:.2f}")

                    x1, y1, x2, y2 = map(int, box)

                    margin = 5
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)

                    crop = img[y1:y2, x1:x2]

                    crop_name = f"{channel_id}_{ts}_{plate_index}.jpg"
                    crop_path = os.path.join(CROP_DIR, crop_name)
                    cv2.imwrite(crop_path, crop)

                    crop_paths.append(crop_path)
                    plate_index += 1

                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ===== SAVE FRAME =====
            frame_name = f"{channel_id}_{ts}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_name)
            cv2.imwrite(frame_path, draw_img)

            # ===== CLEANUP (SETELAH SAVE) =====
            if PLATE_SAVE_LIMIT > 0:
                ch_lock = _get_channel_lock(channel_id)
                with ch_lock:
                    cleanup_old_files(channel_id)

            # encode
            _, buf = cv2.imencode(".jpg", draw_img)
            frame_bytes = buf.tobytes()

            logger.info(
                f"[PLATE] cctv={cctv_name} | channel={channel_id} | total={len(crop_paths)}"
            )

            # ===== WEBHOOK =====
            if WEBHOOK_URL:
                try:
                    plate_webhook_queue.put_nowait((
                        event_id,
                        frame_bytes,
                        frame_name,
                        len(crop_paths),
                        channel_id,
                        client_id,
                        cctv_name,
                        timestamp
                    ))
                except Full:
                    logger.warning("[PLATE] Webhook queue full")

            future.set_result({
                "total_plate": len(crop_paths),
                "frame": frame_path,
                "crops": crop_paths
            })

        except Exception as e:
            logger.error(f"PLATE WORKER #{worker_id} ERROR: {e}")
            future.set_exception(e)
        finally:
            plate_task_queue.task_done()


# ================= START WORKERS =================
for i in range(PLATE_WORKER_COUNT):
    Thread(target=plate_worker, args=(i+1,), daemon=True).start()

if WEBHOOK_URL:
    Thread(target=plate_webhook_worker, daemon=True).start()


# ================= ENDPOINT =================
@router.post("/plate-crop")
async def plate_crop_detect(
    image_bg: UploadFile = File(...),
    event_id: str = Form(...),
    channel_id: str = Form(...),
    client_id: str = Form(...),
    timestamp: str = Form(...),
    cctv_name: str = Form(...),
):
    image_bytes = await image_bg.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "decode_failed"}

    future: Future = Future()

    try:
        plate_task_queue.put_nowait((
            event_id, img, channel_id, client_id, timestamp, cctv_name, future
        ))
    except Full:
        return {"status": "queue_full"}

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: future.result(timeout=30))
    except Exception as e:
        logger.error(f"[PLATE] error: {e}")
        return {"status": "error", "message": str(e)}

    return {
        "status": "success",
        "type": "plate_crop",
        "event_id": event_id,
        "data": {
            "total_plate": result["total_plate"],
            "frame_path": result["frame"],
            "crops": result["crops"],
            "channel_id": channel_id,
            "client_id": client_id,
            "cctv_name": cctv_name,
        },
        "timestamp": timestamp,
    }
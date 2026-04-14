import os
import uuid
import re
import json
import math
import shutil
import cv2
import numpy as np
import insightface
import logging
import requests
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from queue import Queue, Full
from threading import Thread, Lock
from concurrent.futures import Future

from human_count import router as hc_router
from plat_detect import router as plate_router

load_dotenv()

FACE_LIB_PATH = os.getenv("FACE_LIB_PATH", "face_library")
DB_FILE        = os.getenv("DB_FILE", "db.json")
THRESHOLD      = float(os.getenv("THRESHOLD", 0.35))
MODEL_NAME     = os.getenv("MODEL_NAME", "buffalo_l")
MODEL_CTX      = int(os.getenv("MODEL_CTX", -1))
MODEL_DET_SIZE = int(os.getenv("MODEL_DET_SIZE", 320))  # [OPT] turunkan dari 640 default

FACE_CROP_MARGIN    = float(os.getenv("FACE_CROP_MARGIN", 0.3))
FACE_MIN_SIZE       = int(os.getenv("FACE_MIN_SIZE", 80))
FACE_DET_SCORE      = float(os.getenv("FACE_DET_SCORE", 0.6))
FACE_MAX_ANGLE      = float(os.getenv("FACE_MAX_ANGLE", 35))
FACE_BLUR_THRESHOLD = float(os.getenv("FACE_BLUR_THRESHOLD", 20))

WEBHOOK_URL     = os.getenv("WEBHOOK_URL", "https://sumsel.smart-gateway.net/api/webhook/detection")
FACE_SAVE_LIMIT = int(os.getenv("SAVE_LIMIT", 300))
WORKER_COUNT    = int(os.getenv("WORKER", 3))

os.makedirs(FACE_LIB_PATH, exist_ok=True)

BASE_DIR = "face_capture"
FACE_DIR = os.path.join(BASE_DIR, "face")
BG_DIR   = os.path.join(BASE_DIR, "background")
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(BG_DIR, exist_ok=True)

app = FastAPI()
app.mount("/face_library", StaticFiles(directory="face_library"), name="face_library")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("face_api")


# ================= PYDANTIC MODELS =================
class UpdateFolderRequest(BaseModel):
    name: str
    fdid: str

class CreateFolderRequest(BaseModel):
    name: str
    fdid: str


# ================= LOAD MODEL =================
model = insightface.app.FaceAnalysis(name=MODEL_NAME)
# [OPT] det_size lebih kecil = inferensi lebih cepat di CPU
model.prepare(ctx_id=MODEL_CTX, det_size=(MODEL_DET_SIZE, MODEL_DET_SIZE))


# ================= DATABASE (dengan in-memory cache + lock) =================
_db_cache: dict | None = None
_db_lock = Lock()

def load_db() -> dict:
    global _db_cache
    if _db_cache is not None:
        return _db_cache
    if not os.path.exists(DB_FILE):
        _db_cache = {}
        return _db_cache
    try:
        with open(DB_FILE, "r") as f:
            content = f.read().strip()
            _db_cache = json.loads(content) if content else {}
    except json.JSONDecodeError:
        _db_cache = {}
    return _db_cache

def save_db(db: dict) -> None:
    global _db_cache
    with open(DB_FILE, "w") as f:
        json.dump(db, f)
    _db_cache = db
    # [OPT] tandai embedding matrix perlu di-rebuild
    _mark_emb_dirty()


# ================= [OPT] EMBEDDING MATRIX CACHE =================
# Alih-alih loop cosine per entri, kita pre-stack semua embedding ke satu
# matrix numpy dan pakai satu operasi matmul. Jauh lebih cepat untuk DB besar.

_emb_matrix: np.ndarray | None = None   # shape (N, emb_dim)
_emb_names:  list[str]  = []
_emb_fpids:  list[str]  = []
_emb_fdids:  list[str]  = []
_emb_dirty:  bool       = True
_emb_lock   = Lock()


def _mark_emb_dirty():
    global _emb_dirty
    _emb_dirty = True


def _rebuild_emb_matrix():
    """Rebuild embedding matrix dari DB. Dipanggil saat _emb_dirty=True."""
    global _emb_matrix, _emb_names, _emb_fpids, _emb_fdids, _emb_dirty
    db = load_db()
    names, fpids, fdids, vecs = [], [], [], []
    for name, data in db.items():
        for emb in data.get("embeddings", []):
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            vecs.append(arr)
            names.append(name)
            fpids.append(data.get("fpid", ""))
            fdids.append(data.get("fdid", ""))
    if vecs:
        _emb_matrix = np.stack(vecs, axis=0)   # (N, dim)
    else:
        _emb_matrix = np.empty((0, 512), dtype=np.float32)
    _emb_names = names
    _emb_fpids = fpids
    _emb_fdids = fdids
    _emb_dirty = False
    logger.info(f"[EMB CACHE] Rebuilt: {len(vecs)} embeddings")


def fast_recognize(emb: np.ndarray) -> tuple[str | None, float, str | None, str | None]:
    """
    [OPT] Cosine similarity via matrix multiply — O(N) satu operasi,
    bukan loop Python per entri.
    Return: (name, score, fpid, fdid)
    """
    with _emb_lock:
        if _emb_dirty or _emb_matrix is None:
            _rebuild_emb_matrix()
        if _emb_matrix.shape[0] == 0:
            return None, -1.0, None, None
        scores = _emb_matrix @ emb          # (N,) — satu matmul
        idx    = int(np.argmax(scores))
        return _emb_names[idx], float(scores[idx]), _emb_fpids[idx], _emb_fdids[idx]


# ================= FACE VALIDATION HELPERS =================
def _check_blur(img: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_real_face(img: np.ndarray):
    faces = model.get(img)
    if not faces:
        return False, None

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        w, h = x2 - x1, y2 - y1
        ratio = w / h if h != 0 else 0

        logger.debug(f"score: {face.det_score:.3f}  size: {w}x{h}  ratio: {ratio:.2f}")

        if face.det_score < FACE_DET_SCORE:
            continue
        if w < FACE_MIN_SIZE or h < FACE_MIN_SIZE:
            continue
        if not (0.4 <= ratio <= 1.6):
            continue

        blur = _check_blur(img, face.bbox)
        logger.debug(f"blur: {blur:.2f}")
        if blur < FACE_BLUR_THRESHOLD:
            continue

        try:
            left_eye, right_eye = face.kps[0], face.kps[1]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = abs(math.degrees(math.atan2(dy, dx)))
            logger.debug(f"angle: {angle:.2f}")
        except Exception:
            pass

        return True, face

    return False, None


def validate_face(img: np.ndarray, face) -> bool:
    if getattr(face, "det_score", 1) < FACE_DET_SCORE:
        return False

    x1, y1, x2, y2 = map(int, face.bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw < FACE_MIN_SIZE or bh < FACE_MIN_SIZE:
        return False

    if hasattr(face, "pose") and face.pose is not None:
        yaw, pitch, roll = face.pose
        if max(abs(yaw), abs(pitch), abs(roll)) > FACE_MAX_ANGLE:
            return False

    if _check_blur(img, face.bbox) < FACE_BLUR_THRESHOLD:
        return False

    return True


# ================= EMBEDDING =================
def get_embedding(face) -> np.ndarray:
    if face is None:
        raise ValueError("Face object is None")
    emb = face.embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise ValueError("Invalid embedding")
    return emb / norm


# ================= COSINE SIMILARITY (legacy, masih dipakai di register/edit) =================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ================= REGISTER =================
@app.post("/register")
async def register_person(
    name: str = Form(...),
    fdid: str = Form(...),
    fpid: str = Form(None),
    file: UploadFile = File(...)
):
    name = name.strip() if name else None
    fdid = fdid.strip() if fdid else None
    fpid = fpid.strip() if fpid else None

    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not fdid:
        raise HTTPException(status_code=400, detail="FDID is required")

    with _db_lock:
        db = load_db()

        face_folder = next(
            (f for f in os.listdir("face_library") if f.startswith(fdid)), None
        )
        if not face_folder:
            return {"status": "error", "message": "FDID folder not found", "fdid": fdid}

        folder_path = os.path.join("face_library", face_folder)

        if fpid in (None, "string", ""):
            fpid = str(uuid.uuid4())

        for person in db.values():
            if person.get("fpid") == fpid:
                return {"status": "error", "message": "FPID already exists", "fpid": fpid}

        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Image decode failed")

        is_face, face = is_real_face(img)
        if not is_face:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Image is not a valid human face"}
            )

        file_path = os.path.join(folder_path, f"{fpid}.jpg")
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        emb      = get_embedding(face)
        mean_emb = emb / np.linalg.norm(emb)

        db[name] = {
            "fpid":       fpid,
            "fdid":       fdid,
            "embeddings": [mean_emb.tolist()]
        }
        save_db(db)  # otomatis mark dirty

    logger.info(f"Registered: {name} | FPID: {fpid} | FDID: {fdid}")
    return {"status": "registered", "name": name, "fpid": fpid, "fdid": fdid}


# ================= RECOGNIZE =================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")

    is_face, face = is_real_face(img)
    if not is_face:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Image is not a valid human face"}
        )

    db = load_db()
    if not db:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Database empty"})

    try:
        emb = get_embedding(face)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})

    # [OPT] Pakai fast_recognize (matmul) alih-alih loop manual
    best_match, best_score, best_fpid, best_fdid = fast_recognize(emb)

    if best_match is None:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Database empty"}
        )

    percentage        = (best_score + 1) / 2 * 100
    threshold_percent = (THRESHOLD + 1) / 2 * 100

    if best_score > THRESHOLD:
        logger.info(
            f"Recognition success: {best_match} | FPID: {best_fpid} | FDID: {best_fdid} "
            f"| Cosine: {best_score:.4f} | Similarity: {percentage:.2f}%"
        )
        return {
            "status":            "success",
            "match":             best_match,
            "fpid":              best_fpid,
            "fdid":              best_fdid,
            "cosine_score":      round(best_score, 4),
            "similarity_percent": round(percentage, 2),
            "threshold_cosine":  THRESHOLD,
            "threshold_percent": round(threshold_percent, 2)
        }

    raise HTTPException(
        status_code=400,
        detail={
            "status":             "no_match",
            "message":            "No matching face found in the database",
            "cosine_score":       round(best_score, 4),
            "similarity_percent": round(percentage, 2),
            "threshold_cosine":   THRESHOLD,
            "threshold_percent":  round(threshold_percent, 2)
        }
    )


# ================= RECOGNIZE TOP-K =================
@app.post("/recognize/top-k")
async def recognize_top_k(
    file: UploadFile = File(...),
    top_k: int = 5,
    min_similarity: float = 0.0
):
    if not (0.0 <= min_similarity <= 100.0):
        raise HTTPException(status_code=422, detail={"status": "error", "message": "min_similarity harus antara 0.0 dan 100.0"})
    if top_k < 1:
        raise HTTPException(status_code=422, detail={"status": "error", "message": "top_k harus minimal 1"})

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")

    is_face, face = is_real_face(img)
    if not is_face:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Image is not a valid human face"}
        )

    db = load_db()
    if not db:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Database empty"})

    try:
        emb = get_embedding(face)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})

    min_cosine = (min_similarity / 100 * 2) - 1

    # [OPT] Hitung semua skor sekaligus dengan matmul
    with _emb_lock:
        if _emb_dirty or _emb_matrix is None:
            _rebuild_emb_matrix()
        if _emb_matrix.shape[0] == 0:
            return {"status": "success", "candidates": [], "total": 0, "top_k": top_k, "min_similarity": min_similarity}
        scores = _emb_matrix @ emb  # (N,)

    # Aggregasi skor terbaik per orang (ada kemungkinan 1 orang punya >1 embedding)
    best_per_person: dict[str, dict] = {}
    for i, (name, score) in enumerate(zip(_emb_names, scores.tolist())):
        if score < min_cosine:
            continue
        if name not in best_per_person or score > best_per_person[name]["cosine_score"]:
            best_per_person[name] = {
                "name":        name,
                "fpid":        _emb_fpids[i],
                "fdid":        _emb_fdids[i],
                "cosine_score": score,
            }

    if not best_per_person:
        return {"status": "success", "candidates": [], "total": 0, "top_k": top_k, "min_similarity": min_similarity}

    sorted_candidates = sorted(best_per_person.values(), key=lambda x: x["cosine_score"], reverse=True)
    top_candidates    = sorted_candidates[:top_k]

    candidates = []
    for c in top_candidates:
        score      = c["cosine_score"]
        percentage = (score + 1) / 2 * 100
        candidates.append({
            "name":              c["name"],
            "fpid":              c["fpid"],
            "fdid":              c["fdid"],
            "cosine_score":      round(score, 4),
            "similarity_percent": round(percentage, 2),
        })

    logger.info(f"Recognition top-k: found {len(candidates)} candidates | top_k={top_k} | min_similarity={min_similarity}%")
    return {"status": "success", "candidates": candidates, "total": len(candidates), "top_k": top_k, "min_similarity": min_similarity}


# ================= GET ALL PERSONS =================
@app.get("/persons")
async def get_persons_all(request: Request):
    db = load_db()
    if not db:
        return {"total": 0, "persons": []}

    scheme   = request.url.scheme
    host     = request.headers.get("host")
    base_url = f"{scheme}://{host}"

    folder_map: dict[str, str] = {}
    for folder in os.listdir(FACE_LIB_PATH):
        if "_" in folder:
            fdid_part = folder.split("_", 1)[0]
            folder_map[fdid_part] = folder

    persons = []
    for name, data in db.items():
        fpid      = data.get("fpid")
        fdid      = data.get("fdid")
        image_url = None

        folder = folder_map.get(fdid)
        if folder:
            folder_path = os.path.join(FACE_LIB_PATH, folder)
            for filename in os.listdir(folder_path):
                if filename.startswith(fpid):
                    image_url = f"{base_url}/face_library/{folder}/{filename}"
                    break

        persons.append({"name": name, "fpid": fpid, "fdid": fdid, "image_url": image_url})

    return {"total": len(persons), "persons": persons}


# ================= GET PERSONS BY FDID =================
@app.get("/persons/by-fdid/{fdid}")
def get_list_persons_by_fdid(fdid: str):
    db = load_db()
    persons = [
        {
            "name":            name,
            "fpid":            data.get("fpid"),
            "fdid":            data.get("fdid"),
            "embedding_count": len(data.get("embeddings", []))
        }
        for name, data in db.items()
        if data.get("fdid") == fdid
    ]
    if not persons:
        raise HTTPException(status_code=404, detail="No persons found for this FDID")
    return {"fdid": fdid, "total": len(persons), "persons": persons}


# ================= GET PERSON BY FPID =================
@app.get("/persons/by-fpid/{fpid}")
async def get_person_by_fpid(fpid: str, request: Request):
    db = load_db()

    scheme   = request.url.scheme
    host     = request.headers.get("host")
    base_url = f"{scheme}://{host}"

    for name, data in db.items():
        if data.get("fpid") != fpid:
            continue

        fdid      = data.get("fdid")
        image_url = None

        for folder in os.listdir(FACE_LIB_PATH):
            folder_path = os.path.join(FACE_LIB_PATH, folder)
            if os.path.isdir(folder_path) and folder.startswith(fdid + "_"):
                img_file = os.path.join(folder_path, f"{fpid}.jpg")
                if os.path.isfile(img_file):
                    image_url = f"{base_url}/face_library/{folder}/{fpid}.jpg"
                break

        return {
            "name":            name,
            "fpid":            fpid,
            "fdid":            fdid,
            "embedding_count": len(data.get("embeddings", [])),
            "image_url":       image_url
        }

    raise HTTPException(status_code=404, detail="Person not found")


# ================= EDIT PERSON =================
@app.put("/persons/{fpid}")
async def edit_person(
    fpid: str,
    new_name: str = Form(None),
    file: UploadFile = File(None)
):
    with _db_lock:
        db = load_db()

        target_name = next(
            (name for name, data in db.items() if data.get("fpid") == fpid), None
        )
        if target_name is None:
            raise HTTPException(status_code=404, detail="Person not found")

        if new_name in (None, "string", ""):
            return {"status": "error", "message": "New name is required", "fpid": fpid}

        if new_name != target_name and new_name in db:
            raise HTTPException(status_code=400, detail="New name already exists")

        db[new_name] = db.pop(target_name)
        target_name  = new_name

        if file:
            image_bytes = await file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Image decode failed")

            faces = model.get(img)
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected")

            embeddings = [faces[0].embedding]
            for alpha, beta in [(1.0, 0), (1.1, 10), (1.2, 0)]:
                aug   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                aug_f = model.get(aug)
                if aug_f:
                    embeddings.append(aug_f[0].embedding)

            img_flip   = cv2.flip(img, 1)
            faces_flip = model.get(img_flip)
            if faces_flip:
                embeddings.append(faces_flip[0].embedding)

            emb_arr  = np.array(embeddings, dtype=np.float32)
            mean_emb = np.mean(emb_arr, axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            db[target_name]["embeddings"] = [mean_emb.tolist()]

            fdid = db[target_name]["fdid"]
            for folder in os.listdir(FACE_LIB_PATH):
                if folder.startswith(fdid + "_"):
                    folder_path = os.path.join(FACE_LIB_PATH, folder)
                    for filename in os.listdir(folder_path):
                        if filename.startswith(fpid):
                            os.remove(os.path.join(folder_path, filename))
                    with open(os.path.join(folder_path, f"{fpid}.jpg"), "wb") as f:
                        f.write(image_bytes)
                    break

        save_db(db)  # otomatis mark dirty

    logger.info(f"Updated FPID: {fpid} | New Name: {target_name} | Image: {'Yes' if file else 'No'}")
    return {"status": "success", "message": "person updated", "name": target_name, "fpid": fpid}


# ================= DELETE PERSON =================
@app.delete("/persons/{fpid}")
async def delete_person_by_fpid(fpid: str):
    with _db_lock:
        db = load_db()

        target_name = next(
            (name for name, data in db.items() if data.get("fpid") == fpid), None
        )
        if target_name is None:
            raise HTTPException(status_code=404, detail="Person not found")

        fdid = db[target_name].get("fdid")

        for folder in os.listdir(FACE_LIB_PATH):
            if folder.startswith(fdid + "_"):
                folder_path = os.path.join(FACE_LIB_PATH, folder)
                for filename in os.listdir(folder_path):
                    if filename.startswith(fpid):
                        os.remove(os.path.join(folder_path, filename))
                break

        del db[target_name]
        save_db(db)  # otomatis mark dirty

    logger.info(f"Deleted FPID: {fpid} | Name: {target_name} | FDID: {fdid}")
    return {"status": "success", "message": "person deleted", "name": target_name, "fpid": fpid}


# ================= UTILITY =================
@app.get("/percent-to-cosine")
async def conversion_similarity(percent: float = Query(..., ge=0, le=100)):
    cosine = (percent / 100) * 2 - 1
    return {"input_percent": percent, "cosine_value": round(cosine, 6)}


# ================= FACELIB CRUD =================
@app.post("/create-facelib")
def create_facelib(request: CreateFolderRequest):
    os.makedirs(FACE_LIB_PATH, exist_ok=True)

    if request.name in ("", "string", None):
        raise HTTPException(status_code=400, detail="Nama folder tidak boleh kosong")

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

    for folder in os.listdir(FACE_LIB_PATH):
        if "_" in folder:
            existing_name = folder.split("_", 1)[1]
            if existing_name.lower() == safe_name.lower():
                raise HTTPException(status_code=400, detail=f"Folder dengan nama '{request.name}' sudah ada")

    fdid = str(uuid.uuid4()) if request.fdid in ("", "string", None) else request.fdid.strip()

    for folder in os.listdir(FACE_LIB_PATH):
        if folder.startswith(fdid + "_"):
            raise HTTPException(status_code=400, detail=f"Folder dengan FDID '{fdid}' sudah ada")

    folder_name = f"{fdid}_{safe_name}"
    folder_path = os.path.join(FACE_LIB_PATH, folder_name)
    os.makedirs(folder_path)

    logger.info(f"Created facelib: {folder_name}")
    return {
        "status":      "success",
        "fdid":        fdid,
        "name":        request.name,
        "folder_name": folder_name,
        "folder_path": folder_path
    }


@app.get("/list-facelib")
def get_list_facelib():
    if not os.path.exists(FACE_LIB_PATH):
        return {"total": 0, "folders": []}

    folders_data = []
    for folder in os.listdir(FACE_LIB_PATH):
        folder_path = os.path.join(FACE_LIB_PATH, folder)
        if os.path.isdir(folder_path) and "_" in folder:
            fdid, name = folder.split("_", 1)
            file_count = sum(
                1 for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )
            folders_data.append({
                "fdid":        fdid,
                "name":        name,
                "folder_name": folder,
                "file_count":  file_count,
                "folder_path": folder_path
            })

    return {"total": len(folders_data), "folders": folders_data}


@app.get("/facelib/{fdid}")
def get_facelib_by_fdid(fdid: str):
    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):
        if folder.startswith(fdid + "_"):
            folder_path = os.path.join(FACE_LIB_PATH, folder)
            if not os.path.isdir(folder_path):
                continue
            _, name = folder.split("_", 1)
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            return {
                "fdid":        fdid,
                "name":        name,
                "folder_name": folder,
                "file_count":  len(files),
                "files":       files,
                "folder_path": folder_path
            }

    raise HTTPException(status_code=404, detail="FDID not found")


@app.put("/facelib/{fdid}")
def update_facelib(fdid: str, request: UpdateFolderRequest):
    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    if request.name in ("", "string", None):
        raise HTTPException(status_code=400, detail="Nama folder tidak boleh kosong")

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

    old_folder_name = next(
        (f for f in os.listdir(FACE_LIB_PATH) if f.startswith(fdid + "_")), None
    )
    if not old_folder_name:
        raise HTTPException(status_code=404, detail="FDID not found")

    for folder in os.listdir(FACE_LIB_PATH):
        if "_" in folder:
            _, existing_name = folder.split("_", 1)
            if existing_name.lower() == safe_name.lower():
                raise HTTPException(status_code=400, detail="Nama folder sudah digunakan")

    new_folder_name = f"{fdid}_{safe_name}"
    os.rename(
        os.path.join(FACE_LIB_PATH, old_folder_name),
        os.path.join(FACE_LIB_PATH, new_folder_name)
    )

    logger.info(f"Renamed facelib: {old_folder_name} → {new_folder_name}")
    return {
        "status":      "success",
        "fdid":        fdid,
        "old_name":    old_folder_name.split("_", 1)[1],
        "new_name":    request.name,
        "folder_name": new_folder_name
    }


@app.delete("/facelib/{fdid}")
def delete_facelib(fdid: str):
    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):
        if folder.startswith(fdid + "_"):
            folder_path = os.path.join(FACE_LIB_PATH, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                with _db_lock:
                    db = load_db()
                    to_delete = [n for n, d in db.items() if d.get("fdid") == fdid]
                    for name in to_delete:
                        del db[name]
                    save_db(db)  # otomatis mark dirty
                logger.info(f"Deleted facelib FDID: {fdid} | Persons removed: {len(to_delete)}")
                return {"status": "success", "fdid": fdid, "deleted_folder": folder}

    raise HTTPException(status_code=404, detail="FDID not found")


# ================= [OPT] FACE FILE COUNTER (hindari listdir tiap frame) =================
_face_file_count = 0
_face_count_lock = Lock()
_cleanup_lock    = Lock()


def _init_face_count():
    """Hitung file yang sudah ada saat startup, agar counter akurat dari awal."""
    global _face_file_count
    try:
        _face_file_count = len([
            f for f in os.listdir(FACE_DIR) if f.endswith("_face.jpg")
        ])
        logger.info(f"[CLEANUP] Initial face count: {_face_file_count}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Could not init face count: {e}")

_init_face_count()


def cleanup_old_face_files():
    """
    [OPT] Hapus file terlama hanya ketika counter mencapai limit.
    listdir dipanggil sekali di dalam lock, bukan setiap frame.
    """
    global _face_file_count
    with _cleanup_lock:
        if _face_file_count < FACE_SAVE_LIMIT:
            return
        # Baru lakukan listdir di sini, hanya saat memang perlu cleanup
        existing = sorted([
            f for f in os.listdir(FACE_DIR) if f.endswith("_face.jpg")
        ])
        while len(existing) >= FACE_SAVE_LIMIT:
            oldest = existing.pop(0)
            prefix = oldest.replace("_face.jpg", "")
            for path in [
                os.path.join(FACE_DIR, oldest),
                os.path.join(BG_DIR, f"{prefix}_frame.jpg"),
            ]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            _face_file_count = max(0, _face_file_count - 1)


# ================= WEBHOOK WORKER =================
task_queue:    Queue = Queue(maxsize=500)
webhook_queue: Queue = Queue(maxsize=500)


def webhook_worker():
    logger.info("WEBHOOK WORKER STARTED")
    while True:
        item = webhook_queue.get()
        if item is None:
            webhook_queue.task_done()
            break

        try:
            (event_id, face_bytes, frame_bytes, face_name, frame_name,
             bbox, score, channel_id, client_id, cctv_name, timestamp) = item

            payload = {
                "status":     "success",
                "type":       "face_detect",
                "event_id":   event_id,
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(score, 4),
                "channel_id": channel_id,
                "client_id":  client_id,
                "cctv_name":  cctv_name,
                "timestamp":  timestamp,
            }

            resp = requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (face_name,  face_bytes,  "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=payload,
                timeout=10,
            )
            logger.info(
                f"[WEBHOOK OK] {face_name} | cctv={cctv_name} | "
                f"conf={score:.3f} | status={resp.status_code}"
            )

        except Exception as e:
            logger.error(f"[WEBHOOK ERROR] {e}")
        finally:
            webhook_queue.task_done()


# ================= FACE WORKER =================
def face_worker(worker_id: int = 1):
    logger.info(f"FACE WORKER #{worker_id} STARTED")

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        event_id, img, channel_id, client_id, timestamp, cctv_name, future = task

        try:
            faces = model.get(img)

            if not faces:
                future.set_result(None)
                continue

            h, w, _ = img.shape
            ts = timestamp.replace(":", "").replace("-", "").replace("T", "_")

            face_index   = 0
            first_result = None

            faces = faces[:5]

            for face in faces:
                if not validate_face(img, face):
                    continue

                x1, y1, x2, y2 = map(int, face.bbox)
                bw, bh = x2 - x1, y2 - y1

                margin_x = int(bw * FACE_CROP_MARGIN)
                margin_y = int(bh * FACE_CROP_MARGIN)
                nx1 = max(0, x1 - margin_x)
                ny1 = max(0, y1 - margin_y)
                nx2 = min(w, x2 + margin_x)
                ny2 = min(h, y2 + margin_y)

                face_crop = img[ny1:ny2, nx1:nx2]
                bg_img    = img.copy()
                cv2.rectangle(bg_img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

                face_name  = f"{channel_id}_{ts}_{face_index}_face.jpg"
                frame_name = f"{channel_id}_{ts}_{face_index}_frame.jpg"
                face_path  = os.path.join(FACE_DIR, face_name)
                frame_path = os.path.join(BG_DIR, frame_name)

                # [OPT] Encode sekali, pakai untuk simpan file DAN webhook
                # (tidak perlu imwrite + imencode terpisah)
                _, face_buf  = cv2.imencode(".jpg", face_crop)
                _, frame_buf = cv2.imencode(".jpg", bg_img)
                face_bytes   = face_buf.tobytes()
                frame_bytes  = frame_buf.tobytes()

                # Simpan langsung dari bytes
                with open(face_path, "wb") as fp:
                    fp.write(face_bytes)
                with open(frame_path, "wb") as fp:
                    fp.write(frame_bytes)

                # [OPT] Update counter dan cleanup jika perlu
                if FACE_SAVE_LIMIT > 0:
                    with _face_count_lock:
                        global _face_file_count
                        _face_file_count += 1
                    cleanup_old_face_files()

                # [OPT] Pakai fast_recognize (matmul) — tidak perlu snapshot DB manual
                try:
                    emb = get_embedding(face)
                    best_match, best_score, _, _ = fast_recognize(emb)
                    matched_name = best_match if (best_match and best_score > THRESHOLD) else "unknown"
                    confidence   = best_score if best_match else 0.0
                except Exception as e:
                    logger.warning(f"[FACE] Recognition skipped: {e}")
                    matched_name = "unknown"
                    confidence   = 0.0

                logger.info(
                    f"[FACE] cctv={cctv_name} | channel={channel_id} | "
                    f"match={matched_name} | conf={confidence:.4f}"
                )

                if WEBHOOK_URL:
                    try:
                        webhook_queue.put_nowait((
                            event_id,
                            face_bytes, frame_bytes,
                            face_name, frame_name,
                            list(map(int, face.bbox)),
                            confidence,
                            channel_id, client_id, cctv_name,
                            timestamp
                        ))
                    except Full:
                        logger.warning("[FACE] Webhook queue full")

                if first_result is None:
                    first_result = {
                        "bbox":       f"{x1},{y1},{x2},{y2}",
                        "confidence": round(confidence, 4),
                        "channel_id": channel_id,
                        "client_id":  client_id,
                        "cctv_name":  cctv_name,
                        "face_name":  face_name,
                        "timestamp":  timestamp,
                    }

                face_index += 1

            future.set_result(first_result)

        except Exception as e:
            logger.error(f"FACE WORKER #{worker_id} ERROR: {e}")
            future.set_exception(e)

        finally:
            task_queue.task_done()


# ================= START WORKERS =================
for i in range(WORKER_COUNT):
    Thread(target=face_worker, args=(i + 1,), daemon=True).start()

if WEBHOOK_URL:
    Thread(target=webhook_worker, daemon=True).start()


# ================= ENDPOINT /face-detect =================
@app.post("/face-detect")
async def detect_face(
    image_bg:   UploadFile = File(...),
    event_id:   str        = Form(...),
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
        task_queue.put_nowait((event_id, img, channel_id, client_id, timestamp, cctv_name, future))
    except Full:
        return {"status": "queue_full"}

    try:
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: future.result(timeout=30))
    except Exception as e:
        logger.error(f"Face detect error: {e}")
        return {"status": "error", "message": str(e)}

    if result is None:
        return {
            "status":    "success",
            "event_id":  event_id,
            "type":      "face_detect",
            "data":      None,
            "timestamp": timestamp,
        }

    return {
        "status":   "success",
        "type":     "face_detect",
        "event_id": event_id,
        "data": {
            "bbox":       result["bbox"],
            "confidence": result["confidence"],
            "channel_id": result["channel_id"],
            "client_id":  result["client_id"],
            "cctv_name":  result["cctv_name"],
        },
        "timestamp": timestamp,
    }


app.include_router(hc_router)
app.include_router(plate_router)
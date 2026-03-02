import os
import uuid
import re
import json
import shutil
import cv2
import numpy as np
import insightface
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

FACE_LIB_PATH = os.getenv("FACE_LIB_PATH", "face_library")
DB_FILE = os.getenv("DB_FILE", "db.json")
THRESHOLD = float(os.getenv("THRESHOLD", 0.4))
MODEL_NAME = os.getenv("MODEL_NAME", "buffalo_l")
MODEL_CTX = int(os.getenv("MODEL_CTX", -1))

FACE_LIB_PATH = "face_library"
os.makedirs(FACE_LIB_PATH, exist_ok=True)

app = FastAPI()
app.mount("/face_library", StaticFiles(directory="face_library"), name="face_library")

logging.basicConfig(
    level=logging.INFO,  # Bisa ganti ke DEBUG kalau mau detail
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("face_api")

class UpdateFolderRequest(BaseModel):
    name: str
    fdid: str

class CreateFolderRequest(BaseModel):
    name: str
    fdid: str 


# ================= LOAD MODEL =================
model = insightface.app.FaceAnalysis(name=MODEL_NAME)
model.prepare(ctx_id=MODEL_CTX)
# ================= DATABASE =================
def load_db():
    if not os.path.exists(DB_FILE):
        return {}

    try:
        with open(DB_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

# # ================= GENERATE FPID =================
# def generate_fpid(db):
#     if len(db) == 0:
#         return "FP0001"

    # numbers = []
    # for person in db.values():
    #     if isinstance(person, dict) and "fpid" in person:
    #         numbers.append(int(person["fpid"].replace("FP", "")))

    # if not numbers:
    #     return "FP0001"

    # new_id = max(numbers) + 1
    # return f"FP{str(new_id).zfill(4)}"

# ================= EMBEDDING =================
def get_embedding(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")
    faces = model.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected")

    emb = faces[0].embedding.astype(np.float32)

    # WAJIB normalize untuk cosine
    emb = emb / np.linalg.norm(emb)

    return emb

# ================= COSINE SIMILARITY =================
def cosine_similarity(a, b):
    return float(np.dot(a, b))


# ================= REGISTER =================
@app.post("/register")
async def register_person(
    name: str = Form(...),
    fdid: str = Form(...),
    fpid: str = Form(None),
    file: UploadFile = File(...)
):

    # ================= NORMALISASI INPUT =================
    name = name.strip() if name else None
    fdid = fdid.strip() if fdid else None
    fpid = fpid.strip() if fpid else None

    if not name:
        logger.warning("Attempted registration with Name is required.")
        return {"status": "error", "message": "Name is required"}

    if not fdid:
        logger.warning("Attempted registration with FDID is required.")
        return {"status": "error", "message": "FDID is required"}

    db = load_db()

    # ================= VALIDASI NAMA =================
    if name in db:
        logger.warning(f"Attempted registration with duplicate name: {name}")
        return {
            "status": "error",
            "message": "person already registered",
            "name": name,
            "fpid": db[name]["fpid"]
        }

    # ================= CEK FOLDER FDID =================
    face_folder = None

    for folder in os.listdir("face_library"):
        if folder.startswith(fdid):
            face_folder = folder
            break

    if not face_folder:
        logger.warning(f"Attempted registration with non-existent FDID: {fdid}")
        return {
            "status": "error",
            "message": "FDID folder not found",
            "fdid": fdid
        }

    folder_path = os.path.join("face_library", face_folder)

    # ================= GENERATE FPID JIKA KOSONG =================
    if fpid == "string" or not fpid:
        fpid = str(uuid.uuid4())
    else:
        fpid = fpid.strip()

    # ================= VALIDASI FPID DUPLICATE =================
    for person in db.values():
        if person.get("fpid") == fpid:
            logger.warning(f"Attempted registration with duplicate FPID: {fpid}")
            return {
                "status": "error",
                "message": "FPID already exists",
                "fpid": fpid
            }

    # ================= SIMPAN FOTO =================
    image_bytes = await file.read()
    file_path = os.path.join(folder_path, f"{fpid}.jpg")

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # ================= EMBEDDING PROCESS =================
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("Image decode failed during registration")
        return {"error": "Image decode failed"}

    faces = model.get(img)
    if len(faces) == 0:
        logger.error("No face detected during registration")
        return {"error": "No face detected"}

    embeddings = [faces[0].embedding]

    embeddings = np.array(embeddings, dtype=np.float32)
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    # ================= SAVE KE DB =================
    db[name] = {
        "fpid": fpid,
        "fdid": fdid,
        "embeddings": [mean_embedding.tolist()]
    }

    save_db(db)

    logger.info(f"Registered new person: {name} with FPID: {fpid} in FDID: {fdid}")

    return {
        "status": "registered",
        "name": name,
        "fpid": fpid,
        "fdid": fdid
    }

# ================= RECOGNIZE =================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    db = load_db()
    if len(db) == 0:
        logger.warning("Recognition attempt with empty database")
        return {"status": "error", "message": "Database empty"}

    image_bytes = await file.read()

    try:
        emb = get_embedding(image_bytes)
    except ValueError as e:
        # jika gambar tidak ada wajah
        logger.error(f"Recognition error: {str(e)}")
        return {"status": "error", "message": str(e)}

    best_match = None
    best_score = -1
    best_fpid = None
    best_fdid = None

    for name, data in db.items():
        embeddings = data.get("embeddings")
        fpid = data.get("fpid")
        fdid = data.get("fdid")

        if not embeddings:
            continue

        for stored_embedding in embeddings:
            stored = np.array(stored_embedding, dtype=np.float32)
            stored = stored / np.linalg.norm(stored)
            score = float(np.dot(emb, stored))

            if score > best_score:
                best_score = score
                best_match = name
                best_fpid = fpid
                best_fdid = fdid

    percentage = (best_score + 1) / 2 * 100
    threshold_percent = (THRESHOLD + 1) / 2 * 100

    if best_score > THRESHOLD:
        # match ditemukan
        logger.info(f"Recognition success: {best_match} with FPID: {best_fpid} in FDID: {best_fdid} | Cosine Score: {best_score:.4f} | Similarity: {percentage:.2f}%")
        return {
            "status": "success",
            "match": best_match,
            "fpid": best_fpid,
            "fdid": best_fdid,
            "cosine_score": round(best_score, 4),
            "similarity_percent": round(percentage, 2),
            "threshold_cosine": THRESHOLD,
            "threshold_percent": round(threshold_percent, 2)
        }

    # tidak ada match yang memenuhi threshold
    logger.info(f"Recognition no match | Best Score: {best_score:.4f} | Similarity: {percentage:.2f}% | Threshold: {THRESHOLD:.4f}")
    raise HTTPException(
        status_code=400,
        detail={
            "status": "no_match",
            "message": "No matching face found in the database",
            "cosine_score": round(best_score, 4),
            "similarity_percent": round(percentage, 2),
            "threshold_cosine": THRESHOLD,
            "threshold_percent": round(threshold_percent, 2)
        }
    )

@app.get("/persons")
async def get_persons_all(request: Request):

    db = load_db()

    if len(db) == 0:
        logger.warning("Attempted to retrieve persons list but database is empty")
        return {
            "total": 0,
            "persons": []
        }

    scheme = request.url.scheme
    host = request.headers.get("host")
    base_url = f"{scheme}://{host}"

    persons = []

    for name, data in db.items():

        fpid = data.get("fpid")
        fdid = data.get("fdid")

        image_url = None

        # Cari folder berdasarkan FDID
        for folder in os.listdir(FACE_LIB_PATH):
            if folder.startswith(fdid ):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}"):
                        image_url = f"{base_url}/face_library/{folder}/{filename}"
                        break
        logger.info(f"Retrieved person: {name} with FPID: {fpid} in FDID: {fdid} | Image URL: {image_url}")
        persons.append({
            "name": name,
            "fpid": fpid,
            "fdid": fdid,
            "image_url": image_url
        })
    return {
        "total": len(persons),
        "persons": persons
    }


@app.get("/persons/by-fdid/{fdid}")
def get_list_persons_by_fdid(fdid: str):

    db = load_db()

    persons = []

    for name, data in db.items():

        if data.get("fdid") == fdid:
            logger.info(f"Found person: {name} with FPID: {data.get('fpid')} in FDID: {fdid}")
            persons.append({
                "name": name,
                "fpid": data.get("fpid"),
                "fdid": data.get("fdid"),
                "embedding_count": len(data.get("embeddings", []))
            })

    if not persons:
        logger.warning(f"No persons found for FDID: {fdid}")
        raise HTTPException(
            status_code=404,
            detail="No persons found for this FDID"
        )
    
    logger.info(f"Total persons found for FDID {fdid}: {len(persons)}")
    return {
        "fdid": fdid,
        "total": len(persons),
        "persons": persons
    }

@app.get("/persons/by-fpid/{fpid}")
async def get_person_by_fpid(fpid: str, request: Request):

    db = load_db()
    
    scheme = request.url.scheme
    host = request.headers.get("host")
    base_url = f"{scheme}://{host}"

    for name, data in db.items():
        if data.get("fpid") == fpid:
            # for filename in os.listdir(folder_path):
            #         if filename.startswith(f"{fpid}"):
            #             image_url = f"{base_url}/face_library/{folder}/{filename}"
            #             break    
            for folder in os.listdir(FACE_LIB_PATH):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                if not os.path.isdir(folder_path):
                    continue

                _, fdid_name = folder.split("_", 1)

                if os.path.isdir(folder_path) and "_" in folder:
                    logger.info(f"Found person: {name} with FPID: {fpid} in FDID: {data.get('fdid')}")
                    return {
                        "name": name,
                        "fpid": fpid,
                        "embedding_count": len(data.get("embeddings", [])),
                        "image_url": f"{base_url}/face_library/{data.get('fdid')}_{fdid_name}/{fpid}.jpg"
                    }
                else:
                    logger.info(f"Found person: {name} with FPID: {fpid} in FDID: {data.get('fdid')} but image file not found")
                    return {
                        "name": name,
                        "fpid": fpid,
                        "embedding_count": len(data.get("embeddings", [])),
                        "image_url": None
                    }
    # for folder in os.listdir(FACE_LIB_PATH):
    #         if folder.startswith(fpid):

    #             folder_path = os.path.join(FACE_LIB_PATH, folder)

    #             for filename in os.listdir(folder_path):
    #                 if filename.startswith(f"{fpid}"):
    #                     image_url = f"{base_url}/face_library/{folder}/{filename}"
    #                     break    
               

#================== EDIT PERSON =================
@app.put("/persons/{fpid}")
async def edit_person(
    fpid: str,
    new_name: str = Form(None),
    file: UploadFile = File(None)
):

    db = load_db()

    target_name = None

    # ================= CARI PERSON =================
    for name, data in db.items():
        if data.get("fpid") == fpid:
            target_name = name
           # target_data = data
            break

    if target_name is None:
        logger.warning(f"Attempted to edit person with non-existent FPID: {fpid}")
        raise HTTPException(status_code=404, detail="Person not found")

    # ================= UPDATE NAMA =================
    if new_name == "string" or new_name is None: 
        logger.warning(f"Attempted to update person with FPID: {fpid} but new name is invalid")
        return {
            "status": "error",
            "message": "New name is required",
            "fpid": fpid   
        }
    else:
        if new_name != target_name and new_name in db:
            logger.warning(f"Attempted to update person with FPID: {fpid} to duplicate name: {new_name}")
            raise HTTPException(status_code=400, detail="New name already exists")

        db[new_name] = db.pop(target_name)
        target_name = new_name

    # ================= UPDATE IMAGE =================
    if file:
        image_bytes = await file.read()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error(f"Image decode failed during update for FPID: {fpid}")
            raise HTTPException(status_code=400, detail="Image decode failed")

        faces = model.get(img)
        if len(faces) == 0:
            logger.error(f"No face detected during update for FPID: {fpid}")
            raise HTTPException(status_code=400, detail="No face detected")

        embeddings = []
        embeddings.append(faces[0].embedding)

        img_flip = cv2.flip(img, 1)
        faces_flip = model.get(img_flip)
        if len(faces_flip) > 0:
            embeddings.append(faces_flip[0].embedding)

        img_bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        faces_bright = model.get(img_bright)
        if len(faces_bright) > 0:
            embeddings.append(faces_bright[0].embedding)

        img_contrast = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
        faces_contrast = model.get(img_contrast)
        if len(faces_contrast) > 0:
            embeddings.append(faces_contrast[0].embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        db[target_name]["embeddings"] = [mean_embedding.tolist()]

        # ================= HAPUS FILE LAMA BERDASARKAN FPID =================
        for folder in os.listdir(FACE_LIB_PATH):

            if folder.startswith(db[target_name]["fdid"]):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}_"):
                        os.remove(os.path.join(folder_path))

                # Simpan file baru
                new_path = os.path.join(folder_path, f"{fpid}.jpg")

                with open(new_path, "wb") as f:
                    f.write(image_bytes)

                break
        
    # ================= RENAME FILE JIKA HANYA GANTI NAMA =================
    if not file:
        for folder in os.listdir(FACE_LIB_PATH):

            if folder.startswith(db[target_name]["fdid"] + "_"):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}"):

                        old_path = os.path.join(folder_path)
                        new_filename = f"{fpid}.jpg"
                        new_path = os.path.join(folder_path)

                        # Rename tanpa hapus file
                        os.rename(old_path, new_path)

                break
    save_db(db)
    logger.info(f"Updated person with FPID: {fpid} | New Name: {target_name} | Image Updated: {'Yes' if file else 'No'}")
    return {
        "status": "success",
        "message": "person updated",
        "name": target_name,
        "fpid": fpid
    }

@app.delete("/persons/{fpid}")
async def delete_person_by_fpid(fpid: str):

    db = load_db()

    target_name = None

    # ================= CARI PERSON =================
    for name, data in db.items():
        if data.get("fpid") == fpid:
            target_name = name
            break

    if target_name is None:
        logger.warning(f"Attempted to delete person with non-existent FPID: {fpid}")   
        raise HTTPException(status_code=404, detail="Person not found")

    # ================= HAPUS FILE IMAGE =================
    fdid = db[target_name].get("fdid")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            for filename in os.listdir(folder_path):
                if filename.startswith(f"{fpid}_"):
                    os.remove(os.path.join(folder_path, filename))

            break

    # ================= HAPUS DARI DB =================
    del db[target_name]
    save_db(db)
    logger.info(f"Deleted person with FPID: {fpid} | Name: {target_name} | FDID: {fdid}")
    return {
        "status": "success",
        "message": "person deleted",
        "name": target_name,
        "fpid": fpid
    }


@app.get("/percent-to-cosine")
async def conversion_similarity(percent: float = Query(..., ge=0, le=100)):

    cosine = (percent / 100) * 2 - 1
    logger.info(f"Converted percent: {percent}% to cosine similarity: {cosine:.6f}")
    return {
        "input_percent": percent,
        "cosine_value": round(cosine, 6)
    }

@app.post("/create-facelib")
def create_facelib(request: CreateFolderRequest):

    os.makedirs(FACE_LIB_PATH, exist_ok=True)

    if request.name == "" or request.name == "string" or request.name is None:
        logger.warning("Attempted to create face library folder with invalid name")
        raise HTTPException(
            status_code=400,
            detail="Nama folder tidak boleh kosong"
        )
    else:
        # Sanitasi nama
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

        # 🔎 Cek apakah nama sudah ada
        for folder in os.listdir(FACE_LIB_PATH):
            if "_" in folder:
                existing_name = folder.split("_", 1)[1]
                if existing_name.lower() == safe_name.lower():
                    logger.warning(f"Attempted to create face library folder with duplicate name: {request.name}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Folder dengan nama '{request.name}' sudah ada"
                    )

        if request.fdid == "" or request.fdid == "string" or request.fdid is None:
            # Generate UUID
            fdid = str(uuid.uuid4())
        else:
            fdid = str(request.fdid.strip())

            # Cek apakah FDID sudah ada
            for folder in os.listdir(FACE_LIB_PATH):
                if folder.startswith(fdid + "_"):
                    logger.warning(f"Attempted to create face library folder with duplicate FDID: {fdid}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Folder dengan FDID '{fdid}' sudah ada"
                    )

        folder_name = f"{fdid}_{safe_name}"
        folder_path = os.path.join(FACE_LIB_PATH, folder_name)

        os.makedirs(folder_path)

        logger.info(f"Created new face library folder: {folder_name} with FDID: {fdid} and Name: {request.name} folder_name: {folder_name} folder_path: {folder_path}   ") 
        return {
            "status": "success",
            "fdid": fdid,
            "name": request.name,
            "folder_name": folder_name,
            "folder_path": folder_path
        }

@app.get("/list-facelib")
def get_list_facelib():

    if not os.path.exists(FACE_LIB_PATH):
        logger.warning("Attempted to list face library folders but face library path does not exist")
        return {
            "total": 0,
            "folders": []
        }

    folders_data = []

    for folder in os.listdir(FACE_LIB_PATH):

        folder_path = os.path.join(FACE_LIB_PATH, folder)

        if os.path.isdir(folder_path) and "_" in folder:

            fdid, name = folder.split("_", 1)

            # Hitung jumlah file (bukan folder)
            file_count = sum(
                1 for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )

            logger.info(f"Found face library folder: {folder} with FDID: {fdid}, Name: {name}, File Count: {file_count}, Path: {folder_path}")
            folders_data.append({
                "fdid": fdid,
                "name": name,
                "folder_name": folder,
                "file_count": file_count,
                "folder_path": folder_path
            })
    logger.info(f"Total face library folders found: {len(folders_data)}")
    return {
        "total": len(folders_data),
        "folders": folders_data
    }


@app.get("/facelib/{fdid}")
def get_facelib_by_fdid(fdid: str):

    if not os.path.exists(FACE_LIB_PATH):
        logger.warning(f"Attempted to retrieve face library folder with FDID: {fdid} but face library path does not exist")
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            if not os.path.isdir(folder_path):
                continue

            _, name = folder.split("_", 1)

            files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]
            logger.info(f"Retrieved face library folder with FDID: {fdid} | Name: {name} | File Count: {len(files)} | Path: {folder_path}")
            return {
                "fdid": fdid,
                "name": name,
                "folder_name": folder,
                "file_count": len(files),
                "files": files,
                "folder_path": folder_path
            }
    logger.warning(f"Face library folder not found with FDID: {fdid}")
    raise HTTPException(status_code=404, detail="FDID not found")


@app.put("/facelib/{fdid}")
def update_facelib(fdid: str, request: UpdateFolderRequest):

    if not os.path.exists(FACE_LIB_PATH):
        logger.warning(f"Attempted to update face library folder with FDID: {fdid} but face library path does not exist")
        raise HTTPException(status_code=404, detail="Face library not found")

    if request.name == "" or request.name == "string" or  request.name is None:
        logger.warning(f"Attempted to update face library folder with FDID: {fdid} but new name is invalid")
        raise HTTPException(
            status_code=400,
            detail="Nama folder tidak boleh kosong"
        )
    else:
        # Sanitasi nama baru
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

        old_folder_name = None

        # Cari folder berdasarkan FDID
        for folder in os.listdir(FACE_LIB_PATH):
            if folder.startswith(fdid + "_"):
                old_folder_name = folder
                break

        if not old_folder_name:
            logger.warning(f"Attempted to update face library folder with FDID: {fdid} but folder not found")
            raise HTTPException(status_code=404, detail="FDID not found")

        # Cek duplicate name
        for folder in os.listdir(FACE_LIB_PATH):
            if "_" in folder:
                _, existing_name = folder.split("_", 1)
                if existing_name.lower() == safe_name.lower():
                    logger.warning(f"Attempted to update face library folder with FDID: {fdid} to duplicate name: {request.name}")
                    raise HTTPException(
                        status_code=400,
                        detail="Nama folder sudah digunakan"
                    )

        new_folder_name = f"{fdid}_{safe_name}"

        old_path = os.path.join(FACE_LIB_PATH, old_folder_name)
        new_path = os.path.join(FACE_LIB_PATH, new_folder_name)

        os.rename(old_path, new_path)

        logger.info(f"Updated face library folder with FDID: {fdid} | Old Name: {old_folder_name.split('_', 1)[1]} | New Name: {request.name} | New Folder Name: {new_folder_name}")
        return {
            "status": "success",
            "fdid": fdid,
            "old_name": old_folder_name.split("_", 1)[1],
            "new_name": request.name,
            "folder_name": new_folder_name
        }

@app.delete("/facelib/{fdid}")
def delete_facelib(fdid: str):

    if not os.path.exists(FACE_LIB_PATH):
        logger.warning(f"Attempted to delete face library folder with FDID: {fdid} but face library path does not exist")
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

                # ================= HAPUS PERSON DARI DB =================
                db = load_db()

                to_delete = []

                for name, data in db.items():
                    if data.get("fdid") == fdid:
                        to_delete.append(name)

                for name in to_delete:
                    del db[name]

                save_db(db)
                logger.info(f"Deleted face library folder with FDID: {fdid} | Folder Name: {folder} | Deleted Persons Count: {len(to_delete)}")
                return {
                    "status": "success",
                    "fdid": fdid,
                    "deleted_folder": folder
                }
    logger.warning(f"Attempted to delete face library folder with FDID: {fdid} but folder not found")
    raise HTTPException(status_code=404, detail="FDID not found")
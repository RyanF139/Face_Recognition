import os
import requests

# ================= CONFIG =================
API_URL = "http://localhost:8000/plate-crop"
IMAGE_FOLDER = "sample_plat/sample1"  # folder isi banyak foto

CHANNEL_ID = "1"
CLIENT_ID  = "1"
CCTV_NAME  = "Plate"

# ================= RUN =================
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

event_id = 1

for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    with open(img_path, "rb") as f:
        files = {
            "image_bg": (img_name, f, "image/jpeg")
        }

        data = {
            "event_id": str(event_id),
            "channel_id": CHANNEL_ID,
            "client_id": CLIENT_ID,
            "timestamp": str(event_id),  # 👈 di sini
            "cctv_name": CCTV_NAME,
        }

        res = requests.post(API_URL, files=files, data=data)
        print(f"[{event_id}] {res.status_code}")

    event_id += 1
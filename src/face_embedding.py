import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== 設定 =====
CROPS_DIR = "outputs/crops"
MODEL_PATH = "models/face_landmarker.task"

# ===== MediaPipe Landmarker =====
def load_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return vision.FaceLandmarker.create_from_options(options)

# ===== 支援中文路徑的 imread =====
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# ===== 單張臉 → embedding =====
def get_embedding(face_img_path, landmarker):
    img = imread_unicode(face_img_path)
    if img is None:
        print(f" 讀不到圖片：{face_img_path}")
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        print(f" 無法取得 landmarks：{face_img_path}")
        return None

    landmarks = result.face_landmarks[0]
    emb = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).flatten()

    # 正規化（重要）
    emb = emb / np.linalg.norm(emb)
    return emb

# ===== 主流程 =====
def build_database():
    landmarker = load_landmarker()

    for person_name in os.listdir(CROPS_DIR):
        person_dir = os.path.join(CROPS_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n 處理人物：{person_name}")

        embeddings = []

        for file in os.listdir(person_dir):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(person_dir, file)
            emb = get_embedding(img_path, landmarker)
            if emb is not None:
                embeddings.append(emb)
                print(f"  ✔ {file} → embedding")

        if len(embeddings) == 0:
            print(f" {person_name} 沒有有效人臉，略過")
            continue

        embeddings = np.vstack(embeddings)  # shape: (N, 1404)
        mean_embedding = np.mean(embeddings, axis=0)

        # 存檔（存回同一個人名資料夾）
        np.save(os.path.join(person_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(person_dir, "mean_embedding.npy"), mean_embedding)

        print(f" {person_name} 完成：{embeddings.shape[0]} 張臉")

if __name__ == "__main__":
    build_database()

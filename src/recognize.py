import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================
# 設定區
# =====================
MODEL_DETECTOR = "models/haarcascade_frontalface_default.xml"
MODEL_LANDMARKER = "models/face_landmarker.task"

DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試3.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

THRESHOLD = 0.005   # L1 mean distance 門檻
# =====================


# =====================
# 工具函式
# =====================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_face_detector():
    detector = cv2.CascadeClassifier(MODEL_DETECTOR)
    if detector.empty():
        raise IOError("無法載入 Haar Cascade 模型")
    return detector


def load_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_LANDMARKER)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_embedding_from_image(img, landmarker):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    emb = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).flatten()

    # 正規化（距離法必須）
    emb = emb / np.linalg.norm(emb)
    return emb


# =====================
# 載入資料庫
# =====================
def load_database():
    database = {}

    for person_name in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        emb_path = os.path.join(person_dir, "embeddings.npy")

        if not os.path.isfile(emb_path):
            continue

        database[person_name] = np.load(emb_path)

    print(f"📂 載入人物數量：{len(database)}")
    return database


# =====================
# 主流程（完全無 break）
# =====================
def recognize():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    detector = load_face_detector()
    landmarker = load_landmarker()
    database = load_database()

    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        raise ValueError("讀不到輸入圖片")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("未偵測到人臉")
        return

    # ★ 最終你要的 list
    matched_list = []

    # === 走訪所有 faces ===
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            None,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        face_emb = get_embedding_from_image(face, landmarker)
        if face_emb is None:
            continue

        # === 走訪 database 中所有 embeddings ===
        for person_name, embeddings in database.items():
            for db_emb in embeddings:
                dist = np.mean(np.abs(face_emb - db_emb))

                if dist < THRESHOLD:
                    matched_list.append(person_name)
                    print(f"[MATCH] face → {person_name} | dist={dist:.4f}")

    # === 輸出 ===
    cv2.imwrite(OUTPUT_IMAGE, img)

    print("\n📌 所有命中結果（list，可能重複）：")
    print(matched_list)

    print("\n📌 圖片中出現的資料庫人物（去重）：")
    print(list(set(matched_list)))

    print(f"\n📸 輸出圖片：{OUTPUT_IMAGE}")


# =====================
if __name__ == "__main__":
    recognize()

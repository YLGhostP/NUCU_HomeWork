import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# 設定區
# =====================
MODEL_DETECTOR =  "models/haarcascade_frontalface_default.xml"
MODEL_LANDMARKER = "models/face_landmarker.task"

DATABASE_DIR = "outputs/crops"      # 每個人物資料夾內有 embeddings.npy
INPUT_IMAGE ="test_images/測試1.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

SIM_THRESHOLD = 0.5 # 相似度門檻，可之後調整
# =====================
# 工具函式
# =====================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

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
    emb = emb / np.linalg.norm(emb)
    return emb

# =====================
# 載入人物資料庫
# =====================
def load_database():
    database = {}

    for person_name in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        emb_path = os.path.join(person_dir, "embeddings.npy")

        if not os.path.isfile(emb_path):
            continue

        embeddings = np.load(emb_path)
        database[person_name] = embeddings

    print(f" 載入人物數量：{len(database)}")
    return database

# =====================
# 主流程
# =====================
def recognize():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # ① OpenCV Haar detector
    detector = load_face_detector()   # CascadeClassifier
    landmarker = load_landmarker()
    database = load_database()

    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        raise ValueError("讀不到輸入圖片")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ② OpenCV 偵測人臉（取代 MediaPipe detector）
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("未偵測到人臉")
        return

    found_people = set()

    # ③ 逐張臉 → embedding → 比對
    for (x, y, w, h) in faces:
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]

        emb_query = get_embedding_from_image(face, landmarker)
        if emb_query is None:
            continue

        best_name = "Unknown"
        best_score = 0.0

        for person_name, embeddings in database.items():
            scores = cosine_similarity([emb_query], embeddings)[0]
            score = np.max(scores)

            if score > best_score:
                best_score = score
                best_name = person_name

        if best_score < SIM_THRESHOLD:
            best_name = "Unknown"
        else:
            found_people.add(best_name)

        #  畫框＋名字
        label = f"{best_name} ({best_score:.2f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(OUTPUT_IMAGE, img)

    print("辨識結果：")
    for p in found_people:
        print(f" - {p}")
    print(f"輸出圖片：{OUTPUT_IMAGE}")


# =====================
if __name__ == "__main__":
    recognize()

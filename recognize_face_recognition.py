import os
import numpy as np
import cv2
import face_recognition

# =====================
# 設定區
# =====================
DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試2.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

# 距離門檻（L2 distance：越小越像）
THRESHOLD = 0.55
# =====================


# =====================
# 工具函式
# =====================
def imread_unicode(path):
    """支援中文路徑讀圖"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """支援中文路徑寫圖"""
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        return False
    encoded_img.tofile(path)
    return True


def load_database():
    """
    載入 embeddings.npy
    database[person] = (N, 128)
    """
    database = {}

    for person in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        emb_path = os.path.join(person_dir, "embeddings.npy")
        if not os.path.isfile(emb_path):
            continue

        embs = np.load(emb_path)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        database[person] = embs

    print(f"📂 已載入 {len(database)} 位人物資料")
    return database


def l2_distance(a, b):
    return np.linalg.norm(a - b)


# =====================
# 主流程
# =====================
def recognize():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # 1️⃣ 載入資料庫
    database = load_database()
    if len(database) == 0:
        print("❌ 資料庫是空的")
        return

    # 2️⃣ 讀取圖片
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("❌ 讀不到輸入圖片")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3️⃣ 偵測多張人臉
    face_locations = face_recognition.face_locations(rgb, model="hog")

    if len(face_locations) == 0:
        print("⚠️ 未偵測到人臉")
        return

    # 4️⃣ 計算每張臉的 embedding
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    print(f"🔍 偵測到 {len(face_encodings)} 張人臉，開始辨識...")

    recognized_set = set()
    stranger_count = 0

    # 5️⃣ 每一張臉獨立辨識
    for i, (face_emb, (top, right, bottom, left)) in enumerate(
        zip(face_encodings, face_locations)
    ):
        best_distance = float("inf")
        best_person = None

        # 與整個資料庫比對
        for person_name, db_embeddings in database.items():
            for db_emb in db_embeddings:
                dist = l2_distance(face_emb, db_emb)
                if dist < best_distance:
                    best_distance = dist
                    best_person = person_name

        # === 視覺化判斷 ===
        if best_distance < THRESHOLD:
            recognized_set.add(best_person)

            color = (0, 255, 0)        # 綠色
            thickness = 3              # 框線加粗
            label = "check!"

            print(f"✅ [MATCH] 臉部{i} -> check! (best={best_person}, dist={best_distance:.4f})")
        else:
            stranger_count += 1

            color = (0, 0, 255)        # 紅色
            thickness = 2
            label = "Stranger"

            print(f"🚫 [UNKNOWN] 臉部{i} (best dist={best_distance:.4f})")

        # 畫框
        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)

        # 文字背景（讓字更清楚）
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            img,
            (left, top - th - 12),
            (left + tw + 6, top),
            color,
            -1
        )

        # 文字
        cv2.putText(
            img,
            label,
            (left + 3, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # 6️⃣ 輸出結果圖片
    ok = imwrite_unicode(OUTPUT_IMAGE, img)
    if not ok:
        print(f"❌ 寫入結果圖片失敗：{OUTPUT_IMAGE}")
        return
    
    absent_set = set()
    absent_set = set(database.keys()) - recognized_set

    # 7️⃣ 總結
    print("\n📊 辨識總結:")
    print(f" - 出席名單 (去重): {sorted(recognized_set)}")
    print(f" - 應到:{len(database)} 實到{len(recognized_set)}")
    print(f"缺課人數:{len(absent_set)},{absent_set}")
    print(f" - 陌生人數量: {stranger_count}")
    print(f" - 結果圖片: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    recognize()

import os
import cv2
import numpy as np
import face_recognition

# =====================
# 設定區
# =====================
DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試3.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

# 距離門檻（L2 distance：越小越像）
# 常見建議：0.45 ~ 0.6
THRESHOLD = 0.55
# =====================


# =====================
# 工具函式
# =====================
def imread_unicode(path):
    """支援中文路徑讀圖"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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

    # 1. 載入資料庫
    database = load_database()

    # 2. 讀取圖片
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("❌ 讀不到輸入圖片")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. 偵測多張人臉（大合照）
    face_locations = face_recognition.face_locations(
        rgb, model="hog"  # 可改 cnn（慢但準）
    )

    if len(face_locations) == 0:
        print("⚠️ 未偵測到人臉")
        return

    # 4. 對每一張臉算 embedding
    face_encodings = face_recognition.face_encodings(
        rgb, face_locations
    )

    print(f"🔍 偵測到 {len(face_encodings)} 張人臉，開始辨識...")

    # ✅ 用 set 收集「整張圖有哪些資料庫人物出現過」（避免重複、避免污染）
    recognized_set = set()
    stranger_count = 0

    # 5. 逐臉 membership 判斷
    for i, (face_emb, (top, right, bottom, left)) in enumerate(
        zip(face_encodings, face_locations)
    ):
        # ✅ 這張臉命中的人（可能多個，符合你的 membership 定義）
        face_hits = set()

        # ✅ 顯示用：記錄「最小距離」以及對應的人（不是分類用，只是展示）
        best_distance = float("inf")
        best_person = None

        # === 全資料庫走訪（不 break）===
        for person_name, db_embeddings in database.items():
            for db_emb in db_embeddings:
                dist = l2_distance(face_emb, db_emb)

                # 顯示用：最小距離
                if dist < best_distance:
                    best_distance = dist
                    best_person = person_name

                # membership：只要距離小於門檻，就算命中（不 break）
                if dist < THRESHOLD:
                    face_hits.add(person_name)

        # === 視覺化 + 更新整體名單（重要：在「臉級別」結束後才更新）===
        if len(face_hits) > 0:
            # 這張臉是熟人（至少命中一個）
            recognized_set.update(face_hits)

            color = (0, 255, 0)
            label = f"check! ({best_distance:.2f})"

            # ✅ 印出顯示用的 best_person（最小距離那個人）
            # 同時也把 face_hits 印出來方便你 debug（可留可刪）
            print(f"✅ [MATCH] 臉部{i} best={best_person} (dist={best_distance:.4f}) hits={sorted(face_hits)}")
        else:
            # 這張臉是陌生人
            stranger_count += 1

            color = (0, 0, 255)
            label = f"Stranger ({best_distance:.2f})"
            print(f"🚫 [UNKNOWN] 臉部{i} 是陌生人 (best dist={best_distance:.4f})")

        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(
            img, label, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # 6. 輸出結果
    cv2.imwrite(OUTPUT_IMAGE, img)
    print("\n📊 辨識總結:")
    print(f" - 出席名單 (去重): {sorted(recognized_set)}")
    print(f" - 陌生人數量: {stranger_count}")
    print(f" - 結果圖片: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    recognize()

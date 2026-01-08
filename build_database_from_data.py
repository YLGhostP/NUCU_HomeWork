import os
import re
import numpy as np
import cv2
import face_recognition
# main 

# =====================
# 設定區
# =====================
INPUT_DIR = "data"
OUTPUT_ROOT = "outputs/crops"
EMBEDDING_FILENAME = "embeddings.npy"

# face_recognition 偵測模型
# "hog"：快、CPU 即可
# "cnn"：慢、較準（需 GPU）
DETECT_MODEL = "hog"
# =====================

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        return False
    encoded_img.tofile(path)
    return True


def parse_person_name(filename):
    """
    從檔名解析人物名稱
    例：小名1.jpg → 小名
    """
    name = os.path.splitext(filename)[0]
    return re.sub(r"\d+$", "", name)


def imread_unicode(path):
    """支援中文路徑讀圖"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ 找不到資料夾：{INPUT_DIR}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 暫存：person_name -> embeddings list
    all_embeddings = {}

    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_DIR, file)
        img = imread_unicode(img_path)
        if img is None:
            print(f"❌ 讀不到圖片：{file}")
            continue

        person_name = parse_person_name(file)
        person_dir = os.path.join(OUTPUT_ROOT, person_name)
        os.makedirs(person_dir, exist_ok=True)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1️⃣ 偵測人臉
        face_locations = face_recognition.face_locations(
            rgb, model=DETECT_MODEL
        )

        if len(face_locations) == 0:
            print(f"⚠️ 未偵測到人臉：{file}")
            continue

        # 資料庫前提：每張圖只取第一張臉
        top, right, bottom, left = face_locations[0]
        face_crop = img[top:bottom, left:right]

        # 2️⃣ 存裁切後的人臉圖
        crop_name = f"{os.path.splitext(file)[0]}_face0.jpg"
        crop_path = os.path.join(person_dir, crop_name)
        imwrite_unicode(crop_path, face_crop)

        # 3️⃣ 計算 embedding
        encodings = face_recognition.face_encodings(
            rgb, [face_locations[0]]
        )

        if len(encodings) == 0:
            print(f"⚠️ 無法產生 embedding：{file}")
            continue

        emb = encodings[0]

        all_embeddings.setdefault(person_name, []).append(emb)

        print(f"✅ {file} → {crop_path}")

    # 4️⃣ 將每個人的 embeddings 存成 embeddings.npy
    for person_name, emb_list in all_embeddings.items():
        person_dir = os.path.join(OUTPUT_ROOT, person_name)
        embeddings = np.vstack(emb_list)

        out_path = os.path.join(person_dir, EMBEDDING_FILENAME)
        np.save(out_path, embeddings)

        print(f"\n📦 已建立 {person_name} 的 embeddings")
        print(f"   路徑：{out_path}")
        print(f"   shape：{embeddings.shape}")

    print("\n🎉 資料庫建立完成")


if __name__ == "__main__":
    main()

import os
import numpy as np
import face_recognition

# ========= 設定 =========
CROPS_DIR = "outputs/crops"
EMBEDDING_FILENAME = "embeddings.npy"

# ========================

def build_embeddings_for_person(person_dir):
    """
    讀取某一個人的所有 cropped face images
    回傳該人物的 embeddings ndarray, shape = (N, 128)
    """
    embeddings = []

    for file in os.listdir(person_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_dir, file)

        # 讀取人臉圖片
        image = face_recognition.load_image_file(img_path)

        # 對「已裁切的人臉圖」直接算 embedding
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"  ⚠ 無法產生 embedding：{img_path}")
            continue

        # 理論上只會有一張臉
        embeddings.append(encodings[0])

    if len(embeddings) == 0:
        return None

    return np.vstack(embeddings)  # (N, 128)


def main():
    if not os.path.isdir(CROPS_DIR):
        print(f"找不到資料夾：{CROPS_DIR}")
        return

    for person in os.listdir(CROPS_DIR):
        person_dir = os.path.join(CROPS_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n處理人物：{person}")

        embeddings = build_embeddings_for_person(person_dir)

        if embeddings is None:
            print(f"  ❌ 沒有有效 embeddings，略過 {person}")
            continue

        out_path = os.path.join(person_dir, EMBEDDING_FILENAME)
        np.save(out_path, embeddings)

        print(f"  ✅ 儲存 embeddings：{out_path}")
        print(f"     shape = {embeddings.shape}")

if __name__ == "__main__":
    main()

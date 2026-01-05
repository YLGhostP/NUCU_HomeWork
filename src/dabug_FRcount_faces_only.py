import cv2
import numpy as np
import face_recognition
import os

# =====================
# 設定區
# =====================
INPUT_IMAGE = "test_images/測試8.jpg"
OUTPUT_IMAGE = "outputs/visualized/count_faces.jpg"

# face_recognition 偵測模型
# "hog"：快、CPU 即可
# "cnn"：慢、較準（需 GPU）
DETECT_MODEL = "hog"
# =====================


def imread_unicode(path):
    """支援中文路徑讀圖"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def count_faces():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # 1. 讀取圖片
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("❌ 讀不到圖片")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 偵測人臉
    face_locations = face_recognition.face_locations(
        rgb,
        model=DETECT_MODEL
    )

    num_faces = len(face_locations)

    print(f"👥 偵測到的人臉數量：{num_faces}")

    # 3. 視覺化（可選）
    for i, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            img,
            f"Face {i}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # 4. 存圖
    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"🖼 結果圖片已存：{OUTPUT_IMAGE}")


if __name__ == "__main__":
    count_faces()

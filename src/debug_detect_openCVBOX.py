import cv2
import numpy as np
import os

CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
INPUT_IMAGE = "test_images/測試5.jpg"
OUTPUT_IMAGE = "outputs/visualized/debug_boxes.jpg"

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def load_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)

def main():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("圖片讀取失敗")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = load_face_detector()
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    print(faces)
    if len(faces) == 0:
        print("未偵測到人臉")
        return

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 20)
        cv2.putText(
            img,
            f"Face {i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"偵測到 {len(faces)} 張臉")
    print(f"輸出結果：{OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()

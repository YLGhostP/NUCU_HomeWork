import cv2
import os
import re
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

MODEL_PATH = "models/blaze_face_short_range.tflite"

INPUT_DIR = "data"
OUTPUT_DIR = "outputs/crops"

def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.6
    )
    return vision.FaceDetector.create_from_options(options)

def parse_name(filename):
    
    name = os.path.splitext(filename)[0]
    return re.sub(r"\d+$", "", name)

def imread_unicode(path):
    """支援中文路徑的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def process_all_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = load_detector()

    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_DIR, file)
        img = imread_unicode(img_path)
        if img is None:
            print(f" 讀不到圖片：{file}")
            continue

        person_name = parse_name(file)
        person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)
        if not result.detections:
            print(f" 未偵測到臉：{file}")
            continue

        # 只取第一張臉（模板階段）
        det = result.detections[0]
        box = det.bounding_box
        x, y = box.origin_x, box.origin_y
        bw, bh = box.width, box.height

        x, y = max(0, x), max(0, y)
        face = img[y:y+bh, x:x+bw]

        out_name = f"{os.path.splitext(file)[0]}_face0.jpg"
        out_path = os.path.join(person_dir, out_name)

        # 存檔也用 unicode-safe 方法
        _, buf = cv2.imencode(".jpg", face)
        buf.tofile(out_path)

        print(f" {file} → {out_path}")

if __name__ == "__main__":
    process_all_images()

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

MODEL_DETECTOR = "models/blaze_face_short_range.tflite"
INPUT_IMAGE = "test_images/測試5.jpg"
OUTPUT_IMAGE = "outputs/visualized/debug_boxes.jpg"

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def load_face_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_DETECTOR)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.4
    )
    return vision.FaceDetector.create_from_options(options)

def main():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    img = imread_unicode(INPUT_IMAGE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detector = load_face_detector()
    result = detector.detect(mp_image)

    if not result.detections:
        print("未偵測到人臉")
        return

    for i, det in enumerate(result.detections):
        box = det.bounding_box
        x, y = box.origin_x, box.origin_y
        bw, bh = box.width, box.height

        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 20)
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
    print(f"偵測到 {len(result.detections)} 張臉")
    print(f"輸出結果：{OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()

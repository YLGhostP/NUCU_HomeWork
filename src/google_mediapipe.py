import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity  # 記得要 import 這個

# =====================
# 設定區
# =====================
MODEL_DETECTOR = "models/blaze_face_short_range.tflite" 
MODEL_LANDMARKER = "models/face_landmarker.task"

DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試8.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

# 門檻值 (Cosine Similarity: 越大越像)
# 建議 0.6 ~ 0.7 之間
THRESHOLD = 0.6 
# =====================

# =====================
# 工具函式
# =====================
def imread_unicode(path):
    """支援中文路徑讀圖"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None

def load_face_detector():
    """載入 MediaPipe 偵測器"""
    base_options = python.BaseOptions(model_asset_path=MODEL_DETECTOR)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.6
    )
    return vision.FaceDetector.create_from_options(options)

def load_landmarker():
    """載入特徵提取器"""
    base_options = python.BaseOptions(model_asset_path=MODEL_LANDMARKER)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_embedding_from_image(crop_img, landmarker):
    """從裁切後的人臉圖片提取特徵"""
    if crop_img is None or crop_img.size == 0: return None
    
    # 轉為 RGB
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    # 提取特徵並攤平
    landmarks = result.face_landmarks[0]
    emb = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).flatten()

    # L2 正規化
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

# =====================
# 載入資料庫
# =====================
def load_database():
    database = {}
    if not os.path.exists(DATABASE_DIR):
        print("❌ 資料庫目錄不存在")
        return database

    for person_name in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        emb_path = os.path.join(person_dir, "embeddings.npy")
        if not os.path.isfile(emb_path): continue

        # 支援載入單一向量或多個向量
        embs = np.load(emb_path)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
            
        database[person_name] = embs
    print(f"📂 已載入 {len(database)} 位人物資料")
    return database

# =====================
# 主流程
# =====================
def recognize():
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # 1. 載入模型與資料庫
    detector = load_face_detector()
    landmarker = load_landmarker()
    database = load_database()

    # 2. 讀取圖片
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("❌ 讀不到輸入圖片")
        return

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 3. 偵測人臉
    detection_result = detector.detect(mp_image)
    
    if not detection_result.detections:
        print("⚠️ 未偵測到人臉")
        return

    print(f"🔍 偵測到 {len(detection_result.detections)} 張人臉，開始辨識...")
    
    # 建立兩個名單：出席者 與 陌生人計數
    recognized_names = []
    stranger_count = 0

    # 4. 針對每一張臉進行處理
    for i, detection in enumerate(detection_result.detections):
        bbox = detection.bounding_box
        
        # ★ 修正點：強制轉為 int，避免 OpenCV 報錯
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        bw, bh = int(bbox.width), int(bbox.height)

        # 邊界保護 & 裁切
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        
        face_crop = img[y1:y2, x1:x2]
        
        # 提取當前人臉特徵
        face_emb = get_embedding_from_image(face_crop, landmarker)

        final_name = "Unknown"
        final_score = 0.0
        is_known = False  # 標記是否為熟人

        if face_emb is not None:
            # 為了 sklearn 計算，轉為二維
            face_emb_reshaped = face_emb.reshape(1, -1)
            
            candidate_scores = [] # 記分板

            # === 第一步：與資料庫所有人比對 ===
            for person_name, db_embeddings in database.items():
                if db_embeddings.ndim == 1:
                    db_embeddings = db_embeddings.reshape(1, -1)
                
                # 計算相似度 (越大越好)
                similarities = (face_emb_reshaped, db_embeddings)
                avg_score = np.mean(similarities)
                
                candidate_scores.append((person_name, avg_score))

            # === 第二步：找出最高分 ===
            if len(candidate_scores) > 0:
                best_person, best_score = max(candidate_scores, key=lambda item: item[1])
                final_score = best_score # 記錄下來為了顯示

                # === 第三步：陌生人判斷 (關鍵閾值檢查) ===
                if best_score > THRESHOLD:
                    # 分數夠高 -> 認定是這個人
                    final_name = best_person
                    recognized_names.append(final_name)
                    is_known = True
                    print(f"✅ [MATCH] 臉部{i} 是 {final_name} (相似度: {best_score:.4f})")
                else:
                    # 分數不夠高 -> 認定是陌生人
                    final_name = "Stranger"
                    stranger_count += 1
                    print(f"🚫 [UNKNOWN] 臉部{i} 是陌生人 (雖然最像 {best_person}, 但分數僅 {best_score:.4f})")
            else:
                print(f"⚠️ 資料庫為空，無法比對")

        # === 第四步：視覺化 (熟人綠框，陌生人紅框) ===
        if is_known:
            color = (0, 255, 0) # 綠色
            label = f"{final_name} ({final_score:.2f})"
        else:
            color = (0, 0, 255) # 紅色
            label = f"Stranger ({final_score:.2f})" 

        # 畫框與文字
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. 存檔與總結
    cv2.imwrite(OUTPUT_IMAGE, img)
    print("\n📊 點名總結:")
    print(f" - 出席名單 (去重): {list(set(recognized_names))}")
    print(f" - 偵測到陌生人數: {stranger_count}")
    print(f" - 結果圖檔: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    recognize()
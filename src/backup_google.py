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
# 改用 MediaPipe 的偵測模型 (請確保您有下載此檔案)
MODEL_DETECTOR = "models/blaze_face_short_range.tflite" 
MODEL_LANDMARKER = "models/face_landmarker.task"

DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試7.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

# 門檻值建議 (視您的 embedding 計算方式而定，若覺得太嚴格可調大至 0.5~0.7)
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
    """載入 MediaPipe 偵測器 (比 Haar Cascade 強大)"""
    base_options = python.BaseOptions(model_asset_path=MODEL_DETECTOR)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
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
    if crop_img.size == 0: return None
    
    # 轉為 RGB
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    # 提取特徵並攤平
    landmarks = result.face_landmarks[0]
    emb = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).flatten()

    # L2 正規化 (對於比較距離至關重要)
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

    # 3. 偵測人臉 (改用 MediaPipe)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.detections:
        print("⚠️ 未偵測到人臉")
        return

    print(f"🔍 偵測到 {len(detection_result.detections)} 張人臉，開始辨識...")
    
    recognized_names = []

    # 4. 針對每一張臉進行處理
    for i, detection in enumerate(detection_result.detections):
        bbox = detection.bounding_box
        x = bbox.origin_x
        y = bbox.origin_y
        bw = bbox.width
        bh = bbox.height
        print("測試圖片幾張臉----------------------")
        # 邊界保護 (防止裁切超出圖片)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        # 裁切人臉
        face_crop = img[y1:y2, x1:x2]
        
        # 提取當前人臉特徵
 # 提取當前人臉特徵
        face_emb = get_embedding_from_image(face_crop, landmarker)

        # 暫存這位「鏡頭臉」與資料庫所有人的比對結果
        # 格式範例: [('王小明', 0.85), ('李大華', 0.12), ...]
        candidate_scores = [] 

        if face_emb is not None:
            # 為了給 sklearn 計算，需要將 (128,) 轉為 (1, 128) 二維陣列
            face_emb_reshaped = face_emb.reshape(1, -1)

            # === 第一步：走訪資料庫，算出每個人與這張臉的相似度 ===
            for person_name, db_embeddings in database.items():
                
                # 確保 db_embeddings 是 2D (N, 128)
                if db_embeddings.ndim == 1:
                    db_embeddings = db_embeddings.reshape(1, -1)
                
                # ★ 核心計算：一次算出 face_emb 跟該人物所有照片的相似度
                # result 是一個矩陣，例如 [[0.8, 0.82, 0.79]]
                similarities = cosine_similarity(face_emb_reshaped, db_embeddings)
                
                # 取「平均相似度」作為這個人的最終得分
                # (也可以改用 np.max(similarities) 取最高分，看您偏好)
                avg_score = np.mean(similarities)
                
                # 把這個人的名字跟分數存起來 (這就是您要的變數)
                candidate_scores.append((person_name, avg_score))
                
                # Debug 印出過程
                # print(f"正在比對: {person_name} | 相似度: {avg_score:.4f}")

            # === 第二步：找出分數最高的人 (冠軍) ===
            if len(candidate_scores) > 0:
                # 透過分數 (item[1]) 排序，找出最大值
                best_person, best_score = max(candidate_scores, key=lambda item: item[1])

                # === 第三步：檢查是否通過門檻 ===
                # 注意：餘弦相似度是「越大越像」，所以要用 > (大於)
                if best_score > THRESHOLD:
                    recognized_names.append(best_person)
                    final_name = best_person # 為了畫圖用
                    print(f"✅ 判定為: {best_person} (分數: {best_score:.4f})")
                else:
                    final_name = "Unknown"
                    print(f"❌ 未知 (最高分是 {best_person}: {best_score:.4f}，但未達門檻)")
        # 記錄結果
        # if final_name != "Unknown":
        #     recognized_names.append(final_name)
        #     print(f" [MATCH] Face {i} -> {final_name} (Dist: {dist:.4f})")
        # else:
        #     print(f" [FAIL]  Face {i} -> Unknown (Dist: {dist:.4f})")

        # === 視覺化畫圖 ===
        # color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255) # 綠色成功，紅色失敗
        
        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 顯示名字與分數
        # label = f"{final_name} ({min_dist:.2f})"
        cv2.putText(img, None, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. 存檔與總結
    cv2.imwrite(OUTPUT_IMAGE, img)
    print("\n📊 點名總結:")
    print(f" - 出席名單: {list(set(recognized_names))}")
    print(f" - 結果圖檔: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    recognize()
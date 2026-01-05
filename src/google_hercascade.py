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
# ★ 修改點 1: 改用 Haar Cascade 模型路徑
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
MODEL_LANDMARKER = "models/face_landmarker.task"

DATABASE_DIR = "outputs/crops"
INPUT_IMAGE = "test_images/測試6.jpg"
OUTPUT_IMAGE = "outputs/visualized/result.jpg"

# 門檻值 (Cosine Similarity: 越大越像)
THRESHOLD = 0.6 
# =====================

# =====================
# 工具函式
# =====================ssssssssssssssssssssssssssssssss
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
    """★ 修改點 2: 載入 Haar Cascade 偵測器"""
    if not os.path.exists(CASCADE_PATH):
        raise IOError(f"找不到模型檔案: {CASCADE_PATH}")
    return cv2.CascadeClassifier(CASCADE_PATH)

def load_landmarker():
    """載入特徵提取器 (保持 MediaPipe)"""
    base_options = python.BaseOptions(model_asset_path=MODEL_LANDMARKER)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_embedding_from_image(crop_img, landmarker):
    """從裁切後的人臉圖片提取特徵"""
    if crop_img is None or crop_img.size == 0: return None
    
    # 轉為 RGB (Landmarker 依然需要 RGB)
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
    detector = load_face_detector() # 這裡現在是 Haar
    landmarker = load_landmarker()
    database = load_database()

    # 2. 讀取圖片
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        print("❌ 讀不到輸入圖片")
        return

    h, w, _ = img.shape
    
    # ★ 修改點 3: Haar 需要轉灰階才能偵測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ★ 修改點 4: 使用 detectMultiScale 偵測
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print("⚠️ 未偵測到人臉")
        return

    print(f"🔍 偵測到 {len(faces)} 張人臉，開始辨識...")
    
    recognized_names = []
    stranger_count = 0

    # 4. 針對每一張臉進行處理
    # ★ 修改點 5: 直接解包 (x, y, w, h)，不需解析 detection_result
    for i, (x, y, bw, bh) in enumerate(faces):
        
        # 邊界保護 & 裁切
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        
        # 注意：雖然是用灰階偵測，但裁切時要切「原圖(img)」給 Landmarker 用
        face_crop = img[y1:y2, x1:x2]
        
        # 提取當前人臉特徵 (Landmarker 邏輯不變)
        face_emb = get_embedding_from_image(face_crop, landmarker)

        final_name = "Unknown"
        final_score = 0.0
        is_known = False

        if face_emb is not None:
            face_emb_reshaped = face_emb.reshape(1, -1)
            candidate_scores = [] 

            # === 第一步：與資料庫所有人比對 ===
            for person_name, db_embeddings in database.items():
                if db_embeddings.ndim == 1:
                    db_embeddings = db_embeddings.reshape(1, -1)
                
                similarities = cosine_similarity(face_emb_reshaped, db_embeddings)
                avg_score = np.mean(similarities)
                candidate_scores.append((person_name, avg_score))

            # === 第二步：找出最高分 ===
            if len(candidate_scores) > 0:
                best_person, best_score = max(candidate_scores, key=lambda item: item[1])
                final_score = best_score 

                # === 第三步：陌生人判斷 ===
                if best_score > THRESHOLD:
                    final_name = best_person
                    recognized_names.append(final_name)
                    is_known = True
                    print(f"✅ [MATCH] 臉部{i} 是 {final_name} (相似度: {best_score:.4f})")
                else:
                    final_name = "Stranger"
                    stranger_count += 1
                    print(f"🚫 [UNKNOWN] 臉部{i} 是陌生人 (雖然最像 {best_person}, 但分數僅 {best_score:.4f})")
            else:
                print(f"⚠️ 資料庫為空，無法比對")
        else:
            print(f"⚠️ 臉部{i} 雖然被偵測到，但無法提取特徵 (可能太模糊或角度過大)")

        # === 第四步：視覺化 ===
        if is_known:
            color = (0, 255, 0) # 綠色
            label = f"{final_name} ({final_score:.2f})"
        else:
            color = (0, 0, 255) # 紅色
            label = f"Stranger ({final_score:.2f})" 

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
# 人臉辨識多模組實驗專案 (Face Recognition Multi-Module Project)

本專案整合了多種人臉辨識與偵測的技術路徑，包含 **MediaPipe**、**Haar Cascade** 與 **face_recognition (dlib)**。專案支援從圖片前處理（裁切）、特徵提取到最終的相似度比對與點名系統。

## 🛠️ 環境建置

建議使用 Python 3.8 以上版本。

### 1. 安裝依賴套件
```bash
pip install -r requirements.txt


### 2. 模型檔案準備
請確保專案根目錄下有 models/ 資料夾，並放入以下模型（可從 OpenCV / MediaPipe 官方 GitHub 取得）：
-haarcascade_frontalface_default.xml (OpenCV 官方 Haar 模型)
-face_landmarker.task (MediaPipe 特徵點模型)
-blaze_face_short_range.tflite (MediaPipe 偵測模型)

.
├── data/                       # 原始訓練圖片存放區 (放入含人名的圖片)
├── test_images/                # 測試用圖片 (用於辨識測試)
├── models/                     # 模型檔案 (.xml, .task, .tflite)
├── outputs/
│   ├── crops/                  # 裁切後的人臉與 embeddings.npy 資料庫
│   └── visualized/             # 辨識結果輸出圖 (含標記框)
├── requirements.txt            # 依賴套件清單
├── face_detect_and_crop.py     # [前處理] 人臉偵測並裁切存檔
├── recognize.py                # [主程式] 混合流 (Haar + MP + L1距離)
├── recognize_face_recognition.py # [主程式] dlib流 (Face_Recog + L2距離)
├── google_mediapipe.py         # [主程式] 純 Google MP 流 (MP Detect + MP Landmark)
├── google_hercascade.py        # [主程式] 混合流 (Haar Detect + MP Landmark)
└── ... (Debug scripts)



腳本名稱,偵測器 (Detector),特徵提取 (Feature),比對演算法,特點
recognize_face_recognition.py,HOG / CNN,ResNet (dlib),L2 Distance (歐式距離),"最準確。工業級標準，但速度較慢 (HOG依賴CPU, CNN需GPU)。"
google_mediapipe.py,MediaPipe (BlazeFace),MediaPipe Landmarker,Cosine Similarity,全 Google 生態。速度快，對轉向/遮擋容忍度尚可，適合行動裝置或即時應用。
google_hercascade.py,Haar Cascade,MediaPipe Landmarker,Cosine Similarity,極速方案。Haar 偵測極快但誤判率較高，後端接 MP 提取特徵。
recognize.py,Haar Cascade,MediaPipe Landmarker,L1 Mean Distance,實驗性質。使用絕對差值平均作為距離度量，運算成本最低。

準備資料：將含有人臉的照片放入 data/ (檔名為人名，如 劉毅安.jpg)。
前處理：執行 python face_detect_and_crop.py 產生裁切圖。
特徵生成：確認 outputs/crops 下已生成對應的 embeddings.npy (部分腳本整合了生成邏輯，部分需額外執行特徵提取)。
python recognize_face_recognition.py

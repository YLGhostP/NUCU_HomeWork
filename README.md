# 人臉辨識多模組實驗專案 (Face Recognition Multi-Module Project)

本專案整合了多種人臉辨識與偵測的技術路徑，包含 **MediaPipe**、**Haar Cascade** 與 **face_recognition (dlib)**。專案支援從圖片前處理（裁切）、特徵提取到最終的相似度比對與點名系統。

## 🛠️ 環境建置

建議使用 Python 3.8 以上版本。

### 1. 安裝依賴套件
```bash
pip install -r requirements.txt
```
### 2. 專案結構
```text
.
├── data/                         # 原始訓練圖片（檔名即人名）
├── test_images/                  # 測試圖片
├── models/                       # 模型檔案 (.xml / .task / .tflite)
├── outputs/
│   ├── crops/                    # 裁切後人臉 + embeddings.npy
│   └── visualized/               # 辨識結果輸出圖（含框線）
├── requirements.txt              # 套件清單
├── face_detect_and_crop.py       # 前處理：人臉偵測並裁切
├── recognize.py                  # Haar + MP + L1 距離（實驗）
├── recognize_face_recognition.py # dlib 流（ResNet + L2）
├── google_mediapipe.py           # 純 MediaPipe 流
├── google_hercascade.py          # Haar 偵測 + MP 特徵
└── ... (Debug scripts)
```

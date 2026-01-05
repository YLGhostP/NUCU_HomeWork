import face_recognition

# 讀取圖片
image = face_recognition.load_image_file("data\劉毅安1.jpg")

# 偵測人臉
face_locations = face_recognition.face_locations(image)
print("Detected faces:", len(face_locations))

# 計算 embeddings
face_encodings = face_recognition.face_encodings(image, face_locations)

for i, emb in enumerate(face_encodings):
    print(f"Face {i} embedding shape:", emb.shape)

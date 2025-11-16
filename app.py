import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import re
import cv2

# ---- FIX PYTORCH 2.9 LOADER ----
torch.serialization.add_safe_globals([__import__("ultralytics").nn.tasks.DetectionModel])

# ---- Load YOLO Model ----
MODEL_PATH = "runs/detect/train/weights/best.pt"  # ganti sesuai path
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="YOLO Food Calorie Detector", layout="wide")
st.title("🍔 YOLO Food Calorie Detector")

# ---- Upload Image ----
uploaded_file = st.file_uploader("Upload gambar makanan...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert ke numpy (OpenCV)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # ---- Inference ----
    results = model(img_bgr, verbose=False, conf=0.25)[0]

    # ---- Cek Deteksi ----
    if len(results.boxes) == 0:
        st.warning("❗ Tidak ada objek terdeteksi. Turunkan confidence atau cek gambar/model.")
    else:
        # ---- Annotasi Label ----
        annotated_image = img_bgr.copy()
        labels = []
        for cls_id, box in zip(results.boxes.cls.cpu().numpy(), results.boxes.xyxy.cpu().numpy()):
            raw_name = model.names[int(cls_id)]
            
            # Ambil angka pertama (kalori)
            m = re.search(r"(\d+)", raw_name)
            calories = m.group(1) if m else ""
            
            # Ambil nama makanan
            name = re.split(r"\d", raw_name)[0].replace("-", "").strip()
            
            label = f"{name}\n{calories} kal" if calories else name
            labels.append(label)
            
            # Gambar bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Tampilkan hasil
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="Deteksi YOLO", use_column_width=True)

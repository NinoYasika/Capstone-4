Capstone Project Module 4 — YOLO Object Detection App

Aplikasi ini merupakan bagian dari Capstone Project Modul 4 dalam kurikulum Full Data Scientist.
Project ini menerapkan Object Detection menggunakan model YOLO (Ultralytics) dan di-deploy melalui Streamlit Cloud agar dapat diakses secara online.

Aplikasi dapat mendeteksi objek pada gambar yang di-upload pengguna dan menampilkan hasil deteksi dalam bentuk visual bounding box.

🚀 Tech Stack

Python 3.x

YOLO (Ultralytics)

Streamlit

OpenCV (headless)

NumPy

Pillow (PIL)

🎯 Fitur Utama
✔ Upload gambar (JPG/PNG)

Pengguna dapat meng-upload gambar dari device mereka.

✔ YOLO Object Detection

Model akan memproses gambar dan menghasilkan bounding box, confidence score, dan label objek.

✔ Hasil deteksi langsung ditampilkan

Output berupa gambar hasil deteksi.

✔ Tombol download hasil

Pengguna dapat mengunduh gambar hasil deteksi.

✔ Model custom atau pretrained

Aplikasi mendukung:

best.pt (hasil training sendiri)

yolo11n.pt (pretrained model dari Ultralytics)

📂 Project Structure
Capstone_Empat/
│
├── app.py                # Streamlit app utama
├── requirements.txt      # Dependency environment Streamlit Cloud
├── best.pt / yolo11n.pt  # Model YOLO
├── dataset/              # (Opsional) Dataset training
├── notebooks/            # File Jupyter/Colab Notebook
└── README.md             # Dokumentasi proyek

from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("buah.pt")  # Ganti path jika berbeda

# Fungsi untuk deteksi gambar
def deteksi_gambar(image_path):
    results = model(image_path, save=True, conf=0.3)
    for r in results:
        r.show()
        print("Hasil disimpan di:", r.save_dir)

# Fungsi untuk deteksi dari kamera
def deteksi_kamera():
    cap = cv2.VideoCapture(0)  # 0 untuk webcam utama
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi dengan YOLO
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # Tampilkan
        cv2.imshow("Deteksi Kamera", annotated_frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== PILIH MODE ====
print("Pilih mode deteksi:")
print("1. Deteksi dari gambar")
print("2. Deteksi dari kamera (webcam)")
pilih = input("Masukkan pilihan (1/2): ")

if pilih == "1":
    img_path = input("Masukkan path ke gambar: ")
    if os.path.exists(img_path):
        deteksi_gambar(img_path)
    else:
        print("Gambar tidak ditemukan.")
elif pilih == "2":
    deteksi_kamera()
else:
    print("Pilihan tidak valid.")

from PIL import Image
from prediksiYOLO import predict_image, center_crop_square, deteksiBlackArea
import sys
import os
import uuid

DEBUG_DIR = "debugging"

def save_image(image: Image.Image, name: str):
    """Simpan gambar ke folder debugging"""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, name)
    image.save(path)
    print(f"🖼️ Gambar disimpan: {path}")

def main():
    # Ganti ke path gambar kamu
    image_path = "sample/ISIC_0063327.jpg"
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        sys.exit(1)

    # Buka gambar asli
    original_image = Image.open(image_path).convert("RGB")
    save_image(original_image, "1-original.jpg")

    # Simpan hasil center crop
    cropped_image = center_crop_square(original_image)
    save_image(cropped_image, "2-center_crop.jpg")

    # Simpan resize 128x128 untuk model border (debugging saja)
    resized_image = cropped_image.resize((128, 128))
    save_image(resized_image, "3-resize_for_border_model.jpg")

    # Proses deteksi dan crop border hitam
    final_image = deteksiBlackArea(original_image)  # deteksiBlackArea sudah melakukan crop+resize
    save_image(final_image, "4-after_border_crop.jpg")
    
    # Prediksi dengan model YOLO
    result = predict_image(original_image)

    # Tampilkan hasil klasifikasi
    print("\n=== HASIL PREDIKSI ===")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.2%}")  # dalam persen

if __name__ == "__main__":
    main()

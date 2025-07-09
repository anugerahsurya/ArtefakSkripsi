# debug_acgan.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical

# Konfigurasi
model_path = "weights/GeneratorACGAN.h5"
output_path = "debug_output.png"
latent_dim = 128
n_class = 8
test_class = 3  # Ganti sesuai kelas yang ingin diuji (0-7)

print("🚀 Mulai proses debugging ACGAN...")

# Cek apakah file model tersedia
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model tidak ditemukan: {model_path}")

# Load model
print(f"📦 Meload model dari: {model_path}")
generator = tf.keras.models.load_model(model_path)
print("✅ Model berhasil diload.")

# Generate input
label = np.array([test_class])
one_hot_label = to_categorical(label, num_classes=n_class)
noise = tf.random.normal(shape=(1, latent_dim))
print(f"🎯 Class: {test_class}, Noise shape: {noise.shape}")

# Generate image
print("🎨 Menghasilkan gambar...")
synthetic_image = generator.predict([noise, one_hot_label])[0]
synthetic_image = ((synthetic_image + 1) * 127.5).astype(np.uint8)

# Simpan ke file
img = Image.fromarray(synthetic_image)
img.save(output_path)
print(f"✅ Gambar disimpan ke: {output_path}")

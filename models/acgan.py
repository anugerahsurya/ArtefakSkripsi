# models/acgan.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical

# ✅ Load generator & model konfigurasi sekali saja
generator = tf.keras.models.load_model("weights/GeneratorACGAN.h5")  # Sesuaikan path file H5 kamu
latent_dim = 128
n_class = 8 

def generate_acgan(class_id: int, save_path: str):
    """Generate 1 gambar dari ACGAN untuk class tertentu dan simpan ke file."""
    # Buat 1 label dan 1 noise vector
    label = np.array([class_id])
    one_hot_label = to_categorical(label, num_classes=n_class)
    noise = tf.random.normal(shape=(1, latent_dim))

    # Generate gambar
    synthetic_image = generator.predict([noise, one_hot_label])[0]  # Ambil gambar pertama

    # Konversi ke [0,255] dan simpan
    synthetic_image = ((synthetic_image + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(synthetic_image)
    image.save(save_path)

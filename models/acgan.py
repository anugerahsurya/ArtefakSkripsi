# models/acgan.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical
import uuid

# ✅ Load generator & model konfigurasi sekali saja
generator = tf.keras.models.load_model("weights/GeneratorACGAN.h5")  # Sesuaikan path file H5 kamu
latent_dim = 128
n_class = 8 

def generate_acgan(class_id: int, output_dir: str, count: int):
    """
    Generate multiple images using ACGAN for a specific class and save them.

    Args:
        class_id (int): Index of the class (0 to n_class-1).
        output_dir (str): Folder where images will be saved.
        count (int): Number of images to generate.
    """
    # Buat batch of labels dan noise
    labels = np.array([class_id] * count)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=n_class)
    noise = tf.random.normal(shape=(count, latent_dim))

    # Generate batch gambar
    synthetic_images = generator.predict([noise, one_hot_labels], verbose=0)

    # Post-process dan simpan
    synthetic_images = ((synthetic_images + 1) * 127.5).numpy().astype(np.uint8)  # ke [0,255]

    for i in range(count):
        img = Image.fromarray(synthetic_images[i])
        img_name = f"{uuid.uuid4()}.png"
        save_path = os.path.join(output_dir, img_name)
        img.save(save_path)


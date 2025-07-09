import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from io import BytesIO
import uuid

# =============================
# 1. Parameter Model
# =============================
latent_dim = 256
img_shape = (128, 128, 3)
num_classes = 8
beta = 0.1

# =============================
# 2. Definisi Class CVAE
# =============================
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, image_shape, num_classes, beta):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.beta = beta

        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")

        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=image_shape),
            layers.Conv2D(64, 5, strides=2, activation='relu', padding='same'),
            layers.Conv2D(128, 5, strides=2, activation='relu', padding='same'),
            layers.Conv2D(256, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(512, 3, strides=2, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(shape=(latent_dim + num_classes,)),
            layers.Dense(8 * 8 * 512, activation='relu'),
            layers.Reshape((8, 8, 512)),
            layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid'),
        ])

    def decode(self, z):
        return self.decoder(z)

# =============================
# 3. Load Weights Model
# =============================
weights_path = "weights/DecoderCVAE.weights.h5"
cvae = CVAE(latent_dim, img_shape, num_classes, beta)
cvae.build(input_shape=(None, *img_shape))  # wajib untuk load_weights
cvae.load_weights(weights_path)

# =============================
# 4. Fungsi Generate Gambar
# =============================
def generate_cvae(class_id: int, output_dir: str, count: int):
    """
    Generate multiple images using CVAE for a specific class and save them.

    Args:
        class_id (int): Index of the class (0 to num_classes-1).
        output_dir (str): Folder where images will be saved.
        count (int): Number of images to generate.
    """
    # Buat one-hot labels batch
    label_vector = np.zeros((count, num_classes))
    label_vector[:, class_id] = 1

    # Random latent vector
    z = np.random.normal(size=(count, latent_dim))

    # Gabungkan latent vector dengan label
    z_with_label = np.concatenate([z, label_vector], axis=-1)

    # Decode batch gambar
    generated_images = cvae.decode(z_with_label).numpy()  # Shape: (count, H, W, C)

    for i in range(count):
        img_array = (generated_images[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_name = f"{uuid.uuid4()}.png"
        save_path = os.path.join(output_dir, img_name)
        img.save(save_path)


# Ini kode Debugging =============================================

# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras import layers

# # Parameter model
# latent_dim = 256
# img_shape = (128, 128, 3)
# num_classes = 8
# beta = 0.1

# # ====== 1. Definisi Class CVAE ======
# class CVAE(tf.keras.Model):
#     def __init__(self, latent_dim, image_shape, num_classes, beta):
#         super(CVAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.image_shape = image_shape
#         self.num_classes = num_classes
#         self.beta = beta

#         self.z_mean = layers.Dense(latent_dim, name="z_mean")
#         self.z_log_var = layers.Dense(latent_dim, name="z_log_var")

#         self.encoder = tf.keras.Sequential([
#             layers.InputLayer(shape=image_shape),
#             layers.Conv2D(64, 5, strides=2, activation='relu', padding='same'),
#             layers.Conv2D(128, 5, strides=2, activation='relu', padding='same'),
#             layers.Conv2D(256, 3, strides=2, activation='relu', padding='same'),
#             layers.Conv2D(512, 3, strides=2, activation='relu', padding='same'),
#             layers.Flatten(),
#             layers.Dense(latent_dim + latent_dim),
#         ])

#         self.decoder = tf.keras.Sequential([
#             layers.InputLayer(shape=(latent_dim + num_classes,)),
#             layers.Dense(8 * 8 * 512, activation='relu'),
#             layers.Reshape((8, 8, 512)),
#             layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'),
#             layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
#             layers.Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu'),
#             layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid'),
#         ])

#     def decode(self, z):
#         return self.decoder(z)

# # ====== 2. Load Weights ke Model ======
# try:
#     weights_path = "weights/DecoderCVAE.weights.h5"  # ← ini penyesuaian nama
#     cvae = CVAE(latent_dim, img_shape, num_classes, beta)
#     cvae.build(input_shape=(None, *img_shape))  # penting sebelum load_weights
#     cvae.load_weights(weights_path)
#     print(f"[SUKSES] Weights berhasil dimuat dari: {weights_path}")
# except Exception as e:
#     print(f"[ERROR] Gagal load weights: {e}")
#     exit()


# # ====== 3. Fungsi Generate Gambar ======
# def generate_cvae(class_id: int, save_path: str):
#     try:
#         num_images = 1
#         label_vector = np.zeros((num_images, num_classes))
#         label_vector[:, class_id] = 1

#         z = np.random.normal(size=(num_images, latent_dim))
#         z_with_label = np.concatenate([z, label_vector], axis=-1)

#         generated_image = cvae.decoder(z_with_label)[0].numpy()

#         image_array = (generated_image * 255).astype(np.uint8)
#         img = Image.fromarray(image_array)
#         if img.mode != 'RGB':
#             img = img.convert('RGB')

#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         img.save(save_path)
#         print(f"[SUKSES] Gambar berhasil disimpan di: {save_path}")
#     except Exception as e:
#         print(f"[ERROR] Gagal generate gambar untuk class {class_id}: {e}")

# # ====== 4. Debug Uji Generate ======
# if __name__ == "__main__":
#     output_file = "debug_output/class_3_sample.png"
#     generate_cvae(class_id=3, save_path=output_file)


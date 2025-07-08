from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from fastai.vision.all import load_learner, PILImage

# Load model hanya sekali
model = YOLO("weights/best.pt")
# Load model klasifikasi border hitam
learn_black = load_learner('weights/ResNet34-CroppingBlackArea-finetune.pkl')  # ubah path sesuai filemu

# Parameter cropping
CROP_RATIO_TIPIS = 0.1
CROP_RATIO_TEBAL = 0.25

def center_crop_square(image: Image.Image) -> Image.Image:
    """Crop citra ke tengah dengan bentuk persegi"""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))

def crop_black_border(image_np, crop_ratio):
    """Crop border hitam berdasarkan rasio"""
    h, w, _ = image_np.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    cropped = image_np[crop_h:h - crop_h, crop_w:w - crop_w]
    return cv2.resize(cropped, (w, h))

def deteksiBlackArea(image_pil: Image.Image) -> Image.Image:
    """
    Deteksi dan crop border hitam jika perlu.
    
    Args:
        image_pil: PIL.Image sebelum diproses
    
    Returns:
        Image setelah deteksi dan cropping border
    """
    # Step 1: center crop → persegi
    image_cropped = center_crop_square(image_pil)

    # Step 2: resize ke 128x128 → untuk input ke model border
    image_resized = image_cropped.resize((128, 128))

    # Step 3: prediksi dengan model border
    img_fastai = PILImage.create(image_resized)
    pred_border = int(learn_black.predict(img_fastai)[0])  # hasil: 0, 1, atau 2

    # Step 4: crop jika perlu
    image_np = np.array(image_cropped)
    if pred_border == 1:
        result_np = crop_black_border(image_np, CROP_RATIO_TIPIS)
        return Image.fromarray(result_np)
    elif pred_border == 2:
        result_np = crop_black_border(image_np, CROP_RATIO_TEBAL)
        return Image.fromarray(result_np)
    else:
        return image_cropped  # tidak di-crop jika border = 0


def predict_image(image: Image.Image):
    # Step 1–4: Deteksi dan crop border
    image = deteksiBlackArea(image)

    # Step 5: Resize ke ukuran yang sama saat training (224x224)
    image = image.resize((224, 224))
    print(f"[DEBUG] Ukuran akhir gambar sebelum predict: {image.size}")  # (224, 224)

    # Step 6: Prediksi klasifikasi (YOLOv8)
    result = model.predict(image, verbose=False)
    probs = result[0].probs

    if probs is None:
        return {"label": "Unknown", "confidence": 0.0}
    
    probs = probs.data.cpu().numpy()
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = model.names[top_idx]
    return {"label": label, "confidence": confidence}




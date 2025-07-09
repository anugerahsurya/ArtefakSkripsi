from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict
import os
import uuid
import shutil
import tempfile

from models.acgan import generate_acgan
from models.cvae import generate_cvae
from models.cddpm import generate_cddpm

router = APIRouter()

# Folder hasil gambar
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simpan file hasil generate terakhir
last_generated_files: List[str] = []

class GenerateRequest(BaseModel):
    model: str
    image_count: int
    classes: List[int]

@router.post("/generate")
async def generate_images(req: GenerateRequest):
    global last_generated_files
    last_generated_files = []

    model = req.model.lower()
    image_count = req.image_count
    classes = req.classes

    results: Dict[str, List[str]] = {}

    for cls in classes:
        cls_dir = os.path.join(OUTPUT_DIR, str(uuid.uuid4()))
        os.makedirs(cls_dir, exist_ok=True)

        # Panggil generator untuk menghasilkan banyak gambar sekaligus
        if model == "acgan":
            generate_acgan(cls, cls_dir, image_count)
        elif model == "cvae":
            generate_cvae(cls, cls_dir, image_count)
        elif model == "cddpm":
            generate_cddpm(cls, cls_dir, image_count)
        else:
            return JSONResponse(content={"error": "Model tidak valid."}, status_code=400)

        # Ambil semua file yang baru dihasilkan
        cls_images = []
        for fname in sorted(os.listdir(cls_dir)):
            full_path = os.path.join(cls_dir, fname)
            last_generated_files.append(full_path)
            cls_images.append(f"/generated/{os.path.basename(full_path)}")

            # Pindahkan ke OUTPUT_DIR root agar bisa diakses frontend (jika perlu)
            os.rename(full_path, os.path.join(OUTPUT_DIR, os.path.basename(full_path)))

        results[str(cls)] = cls_images

    return {"generated": results}

@router.get("/download")
async def download_zip():
    if not last_generated_files:
        return JSONResponse(content={"error": "Belum ada data untuk diunduh."}, status_code=400)

    # Buat folder sementara dan salin file
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_path in last_generated_files:
            if os.path.exists(file_path):
                shutil.copy(file_path, tmpdir)

        # Buat file zip dari isi folder sementara
        zip_path = shutil.make_archive(tmpdir, 'zip', tmpdir)

        return FileResponse(
            zip_path,
            filename="generated_images.zip",
            media_type="application/zip"
        )

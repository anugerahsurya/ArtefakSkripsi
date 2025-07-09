from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io
from prediksiYOLO import predict_image

# Import router baru
from routers import generative

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ➕ Tambahkan router untuk endpoint /generate
app.include_router(generative.router)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Static file
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/generated", StaticFiles(directory="generated_images"), name="generated")
app.mount("/", StaticFiles(directory=".", html=True), name="static")
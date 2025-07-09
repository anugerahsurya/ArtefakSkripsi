from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io

from prediksiYOLO import predict_image
from routers import generative

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router tambahan
app.include_router(generative.router)

# Endpoint utama
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Serve file statis
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/generated", StaticFiles(directory="generated_images"), name="generated")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

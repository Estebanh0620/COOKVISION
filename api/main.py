import cv2
import numpy as np
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from procesamiento.ruido import quitar_ruido
from procesamiento.segmentacion import segmentar
from procesamiento.contornos import detectar_contornos

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagenRequest(BaseModel):
    imagen: str

@app.get("/")
def health():
    return {"status": "ok", "app": "CookVision"}

@app.post("/analizar")
def analizar(body: ImagenRequest):
    img_bytes = base64.b64decode(body.imagen)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if imagen is None:
        return {"error": "No se pudo decodificar la imagen"}

    imagen_sin_ruido = quitar_ruido(imagen)
    umbral = segmentar(imagen_sin_ruido)
    imagen_contornos, cantidad = detectar_contornos(imagen, umbral)

    _, buffer = cv2.imencode(".jpg", imagen_contornos)
    resultado_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "cantidad": cantidad,
        "imagen_resultado": resultado_b64,
    }
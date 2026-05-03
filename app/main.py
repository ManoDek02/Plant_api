"""
main.py — API FastAPI pour la segmentation agricole PlantVillage
Utilise un modèle HuggingFace en attendant le modèle U-Net entraîné.

Endpoints :
  POST /predict        → image upload (multipart/form-data)
  POST /predict/url    → image via URL (JSON)
  GET  /health         → statut de l'API
  GET  /classes        → liste des maladies détectables
"""
import io
import base64
import httpx
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from app.predictor import PlantPredictor
from app.utils import image_to_base64, generate_mask_overlay

# ─── Modèle chargé au démarrage ────────────────────────────────────────────
predictor: Optional[PlantPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage."""
    global predictor
    print("Chargement du modèle...")
    predictor = PlantPredictor()
    print("Modèle prêt ✓")
    yield
    print("Arrêt de l'API")


# ─── App FastAPI ────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌿 Plant Disease API",
    description="API de détection et segmentation de maladies foliaires (PlantVillage)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — autorise toutes les origines (adapte selon ta plateforme)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Schémas de réponse                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

class PredictionResponse(BaseModel):
    # Classification
    label: str                  # ex: "Tomato Bacterial Spot"
    is_healthy: bool
    confidence: float           # 0.0 → 1.0
    disease_ratio: float        # proportion pixels malades
    severity: str               # "SAINE" | "LÉGÈRE" | "MODÉRÉE" | "SÉVÈRE"
    top3: list                  # top 3 classes avec scores

    # Images en base64 (affichage direct dans <img src="data:image/png;base64,...">)
    image_original_b64: str     # image originale
    image_mask_b64: str         # masque binaire (blanc = zone malade)
    image_overlay_b64: str      # image + masque superposé en rouge


class UrlRequest(BaseModel):
    url: str
    threshold: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════ #
#  Endpoints                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": predictor.model_name if predictor else "non chargé",
        "version": "1.0.0",
    }


@app.get("/classes")
async def get_classes():
    """Retourne la liste des maladies détectables."""
    return {
        "count": len(predictor.classes),
        "classes": predictor.classes,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(..., description="Image de feuille (jpg/png)"),
    threshold: float = 0.5,
):
    """
    Prédit depuis un fichier uploadé.
    Content-Type: multipart/form-data
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image (jpg, png)")

    contents = await file.read()
    return await _run_prediction(contents, threshold)


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url(body: UrlRequest):
    """
    Prédit depuis une URL d'image.
    Content-Type: application/json
    Body: {"url": "https://...", "threshold": 0.5}
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
            contents = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de télécharger l'image : {e}")

    return await _run_prediction(contents, body.threshold)


# ─── Fonction commune ───────────────────────────────────────────────────────

async def _run_prediction(image_bytes: bytes, threshold: float) -> PredictionResponse:
    """Traitement commun pour upload et URL."""
    try:
        # Décode l'image
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Image illisible ou corrompue")

        # Prédiction
        result = predictor.predict(bgr, threshold=threshold)

        return PredictionResponse(
            label            = result["label"],
            is_healthy       = result["is_healthy"],
            confidence       = result["confidence"],
            disease_ratio    = result["disease_ratio"],
            severity         = result["severity"],
            top3             = result["top3"],
            image_original_b64 = image_to_base64(bgr),
            image_mask_b64     = image_to_base64(result["mask"]),
            image_overlay_b64  = image_to_base64(result["overlay"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

"""
predictor.py — Modèle de prédiction utilisant HuggingFace.

Utilise le modèle de classification PlantVillage disponible gratuitement.
La segmentation (masque) est générée via traitement HSV sur l'image.

⚡ Quand ton U-Net sera entraîné :
   → Remplace juste la méthode _generate_mask() par ton modèle .pth
"""
from typing import Dict, List, Any
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification

from app.utils import generate_mask_hsv, generate_overlay


# ─── Mapping label HuggingFace → nom lisible ────────────────────────────────
LABEL_MAP = {
    "Apple___Apple_scab":                     "Tavelure du pommier",
    "Apple___Black_rot":                      "Pourriture noire du pommier",
    "Apple___Cedar_apple_rust":               "Rouille du pommier",
    "Apple___healthy":                        "Pommier sain",
    "Blueberry___healthy":                    "Myrtille saine",
    "Cherry___Powdery_mildew":                "Oïdium du cerisier",
    "Cherry___healthy":                       "Cerisier sain",
    "Corn___Cercospora_leaf_spot":            "Tache foliaire du maïs",
    "Corn___Common_rust":                     "Rouille commune du maïs",
    "Corn___Northern_Leaf_Blight":            "Brûlure nordique du maïs",
    "Corn___healthy":                         "Maïs sain",
    "Grape___Black_rot":                      "Pourriture noire de la vigne",
    "Grape___Esca":                           "Esca de la vigne",
    "Grape___Leaf_blight":                    "Brûlure foliaire de la vigne",
    "Grape___healthy":                        "Vigne saine",
    "Orange___Citrus_greening":               "Verdissement des agrumes",
    "Peach___Bacterial_spot":                 "Tache bactérienne du pêcher",
    "Peach___healthy":                        "Pêcher sain",
    "Pepper___Bacterial_spot":                "Tache bactérienne du poivron",
    "Pepper___healthy":                       "Poivron sain",
    "Potato___Early_blight":                  "Mildiou précoce de la pomme de terre",
    "Potato___Late_blight":                   "Mildiou tardif de la pomme de terre",
    "Potato___healthy":                       "Pomme de terre saine",
    "Raspberry___healthy":                    "Framboisier sain",
    "Soybean___healthy":                      "Soja sain",
    "Squash___Powdery_mildew":                "Oïdium de la courge",
    "Strawberry___Leaf_scorch":               "Brûlure foliaire du fraisier",
    "Strawberry___healthy":                   "Fraisier sain",
    "Tomato___Bacterial_spot":                "Tache bactérienne de la tomate",
    "Tomato___Early_blight":                  "Mildiou précoce de la tomate",
    "Tomato___Late_blight":                   "Mildiou tardif de la tomate",
    "Tomato___Leaf_Mold":                     "Moisissure foliaire de la tomate",
    "Tomato___Septoria_leaf_spot":            "Septoriose de la tomate",
    "Tomato___Spider_mites":                  "Acariens de la tomate",
    "Tomato___Target_Spot":                   "Tache cible de la tomate",
    "Tomato___Yellow_Leaf_Curl_Virus":        "Virus de l'enroulement jaune de la tomate",
    "Tomato___Mosaic_virus":                  "Virus de la mosaïque de la tomate",
    "Tomato___healthy":                       "Tomate saine",
}


class PlantPredictor:
    """
    Predictor principal.
    Classification  : HuggingFace (linkanjarad/mobilenet_v2_1.0_224)
    Segmentation    : HSV + morphologie (remplacé par U-Net quand prêt)
    """

    MODEL_ID   = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    IMAGE_SIZE = 224

    def __init__(self):
        print(f"Chargement du modèle : {self.MODEL_ID}")
        self._pipe = pipeline(
            "image-classification",
            model=self.MODEL_ID,
            top_k=3,
        )
        self.model_name = self.MODEL_ID
        self.classes    = list(LABEL_MAP.keys())
        print("Modèle chargé ✓")

    # ------------------------------------------------------------------ #
    def predict(self, bgr: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Prédit la maladie et génère le masque de segmentation.

        Paramètres
        ----------
        bgr       : image BGR (numpy array)
        threshold : seuil de détection pour le masque (0.0–1.0)

        Retourne
        --------
        dict avec : label, is_healthy, confidence, disease_ratio,
                    severity, top3, mask, overlay
        """
        # ── 1. Classification ────────────────────────────────────────────
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        preds = self._pipe(pil)   # liste de {label, score}

        top1       = preds[0]
        raw_label  = top1["label"]
        confidence = round(float(top1["score"]), 4)
        is_healthy = "healthy" in raw_label.lower()

        # Label lisible
        label_fr = LABEL_MAP.get(raw_label, raw_label.replace("___", " — ").replace("_", " "))

        # Top 3 avec labels lisibles
        top3 = [
            {
                "label":      LABEL_MAP.get(p["label"], p["label"]),
                "label_raw":  p["label"],
                "confidence": round(float(p["score"]), 4),
            }
            for p in preds[:3]
        ]

        # ── 2. Segmentation (masque) ──────────────────────────────────────
        mask    = self._generate_mask(bgr, is_healthy)
        overlay = generate_overlay(bgr, mask)

        # ── 3. Ratio et sévérité ──────────────────────────────────────────
        disease_ratio = float(np.sum(mask > 0)) / mask.size
        severity = _get_severity(is_healthy, disease_ratio)

        return {
            "label":         label_fr,
            "label_raw":     raw_label,
            "is_healthy":    is_healthy,
            "confidence":    confidence,
            "disease_ratio": round(disease_ratio, 4),
            "severity":      severity,
            "top3":          top3,
            "mask":          mask,
            "overlay":       overlay,
        }

    # ------------------------------------------------------------------ #
    def _generate_mask(self, bgr: np.ndarray, is_healthy: bool) -> np.ndarray:
        """
        Génère le masque de segmentation.

        ✅ Actuellement : traitement HSV (pas besoin de GPU)
        🔄 À remplacer : charger ton U-Net .pth quand entraîné
        """
        return generate_mask_hsv(bgr, is_healthy=is_healthy)

    # ── Pour brancher ton U-Net plus tard ─────────────────────────────── #
    # def _load_unet(self, checkpoint_path: str):
    #     import segmentation_models_pytorch as smp
    #     model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1)
    #     ckpt  = torch.load(checkpoint_path, map_location="cpu")
    #     model.load_state_dict(ckpt["model_state"])
    #     return model.eval()
    #
    # def _generate_mask_unet(self, bgr, model):
    #     import albumentations as A
    #     from albumentations.pytorch import ToTensorV2
    #     transform = A.Compose([A.Resize(256,256), A.Normalize(), ToTensorV2()])
    #     tensor = transform(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0)
    #     with torch.no_grad():
    #         logits = model(tensor)
    #     prob = torch.sigmoid(logits[0,0]).numpy()
    #     mask = (prob > 0.5).astype(np.uint8) * 255
    #     return cv2.resize(mask, bgr.shape[:2][::-1])


# ─── Helpers ────────────────────────────────────────────────────────────────

def _get_severity(is_healthy: bool, ratio: float) -> str:
    if is_healthy or ratio < 0.01:
        return "SAINE"
    elif ratio < 0.10:
        return "LÉGÈRE"
    elif ratio < 0.30:
        return "MODÉRÉE"
    else:
        return "SÉVÈRE"

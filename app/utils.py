"""
utils.py — Fonctions utilitaires : masque HSV, overlay, encodage base64.
"""
import base64
import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════ #
#  Génération de masque HSV                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def generate_mask_hsv(
    bgr: np.ndarray,
    is_healthy: bool = False,
    kernel_size: int = 5,
    min_area: int = 80,
) -> np.ndarray:
    """
    Génère un masque binaire des zones malades via seuillage HSV.
    Retourne un array [H, W] uint8 (0 ou 255).
    """
    h, w = bgr.shape[:2]

    if is_healthy:
        return np.zeros((h, w), dtype=np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Masque feuille (exclut le fond clair/blanc)
    leaf = cv2.inRange(hsv, np.array([20, 15, 15]), np.array([170, 255, 255]))
    k_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k_cl)

    # Zones malades : brun/ocre/jaune + zones sombres (nécrose)
    ranges = [
        (np.array([ 8, 30, 30]),  np.array([30, 255, 255])),   # brun-orangé
        (np.array([30, 20, 20]),  np.array([65, 255, 190])),   # jaune pâle
        (np.array([ 0,  0,  0]),  np.array([180, 80,  75])),   # nécrose sombre
    ]
    combined = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

    # Intersection avec la feuille
    combined = cv2.bitwise_and(combined, leaf)

    # Nettoyage morphologique
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    # Supprime micro-composantes
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    clean = np.zeros_like(combined)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return clean


# ═══════════════════════════════════════════════════════════════════════════ #
#  Overlay                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def generate_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color_bgr: Tuple[int, int, int] = (0, 0, 220),  # rouge
) -> np.ndarray:
    """
    Superpose le masque en couleur sur l'image originale.
    Retourne une image BGR.
    """
    overlay = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:] = color_bgr

    mask_bool = mask > 0
    blend = cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)
    overlay[mask_bool] = blend[mask_bool]

    # Contour pour mieux délimiter la zone
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    return overlay


# ═══════════════════════════════════════════════════════════════════════════ #
#  Encodage base64                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def image_to_base64(img: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """
    Convertit un array numpy (BGR ou grayscale) en string base64 JPEG/PNG.

    Usage côté frontend :
        <img src="data:image/jpeg;base64,{image_original_b64}" />
    """
    encode_params = []
    if fmt == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, buffer = cv2.imencode(fmt, img, encode_params)
    if not success:
        raise ValueError("Échec de l'encodage de l'image")

    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"

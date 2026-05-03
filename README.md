# 🌿 Plant Disease API

API de détection et segmentation de maladies foliaires (PlantVillage).

**Modèle actuel** : MobileNetV2 HuggingFace (classification) + HSV (masque)  
**Modèle futur** : U-Net ResNet-50 entraîné sur PlantVillage

---

## 🚀 Déploiement en 5 minutes

### Sur Render.com (recommandé)

1. Push ce dossier sur GitHub
2. Va sur [render.com](https://render.com) → New Web Service
3. Connecte ton repo GitHub
4. Render détecte automatiquement le `Dockerfile`
5. Clique **Deploy** → attends ~5 min
6. Ton URL : `https://plant-disease-api.onrender.com`

### Sur Railway.app

1. Push sur GitHub
2. Va sur [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Sélectionne ton repo
4. Railway détecte le `railway.toml` automatiquement
5. Ton URL : `https://plant-disease-api.up.railway.app`

---

## 📡 Endpoints

### `GET /health`
Vérifie que l'API fonctionne.
```json
{"status": "ok", "model": "...", "version": "1.0.0"}
```

---

### `POST /predict` — Upload d'image
```javascript
// Depuis ton frontend
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://TON-URL/predict', {
  method: 'POST',
  body: formData,
});
const result = await response.json();
```

---

### `POST /predict/url` — Via URL
```javascript
const response = await fetch('https://TON-URL/predict/url', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'https://exemple.com/feuille.jpg' }),
});
const result = await response.json();
```

---

## 📦 Format de réponse

```json
{
  "label": "Tache bactérienne de la tomate",
  "is_healthy": false,
  "confidence": 0.9423,
  "disease_ratio": 0.1823,
  "severity": "MODÉRÉE",
  "top3": [
    {"label": "Tache bactérienne de la tomate", "confidence": 0.9423},
    {"label": "Mildiou précoce de la tomate",   "confidence": 0.0312},
    {"label": "Tomate saine",                   "confidence": 0.0181}
  ],
  "image_original_b64": "data:image/jpeg;base64,...",
  "image_mask_b64":     "data:image/jpeg;base64,...",
  "image_overlay_b64":  "data:image/jpeg;base64,..."
}
```

### Affichage des images base64 dans ton frontend
```html
<img src="{{ result.image_original_b64 }}" />
<img src="{{ result.image_mask_b64 }}" />
<img src="{{ result.image_overlay_b64 }}" />
```

---

## 🔄 Remplacer par ton U-Net (quand entraîné)

Dans `app/predictor.py`, décommente la méthode `_generate_mask_unet()`
et modifie `_generate_mask()` :

```python
def _generate_mask(self, bgr, is_healthy):
    # Remplace HSV par U-Net
    return self._generate_mask_unet(bgr, self.unet_model)
```

---

## 🧪 Tester localement

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000/docs  (Swagger UI automatique)
```

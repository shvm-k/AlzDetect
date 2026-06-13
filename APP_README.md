# AlzDetect — Web App

A deployable web application that wraps the research model from this repo
(MobileNetV2 Alzheimer's MRI stage classifier) into an uploadable, shareable
product.

> ⚠️ **Research / educational use only. Not a medical device.** Do not use for
> clinical diagnosis.

## What it does

Upload an MRI brain scan → get the predicted dementia stage, per-class
confidence bars, and a Grad-CAM heatmap showing where the model "looked".

Four classes (matching the notebook): Mild Dementia, Moderate Dementia,
Very mild Dementia, Non Demented.

## Architecture

```
frontend/   Static single-page UI (HTML/CSS/JS, no build step)
backend/    FastAPI service: /api/predict, /api/health, serves frontend
  model.py        preprocessing (128x128, /255), inference, Grad-CAM
  app.py          API + static hosting
  models/         drop your trained .keras here (see models/README.md)
```

## Run locally

```bash
cd backend
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000

### Demo mode vs live

Without trained weights the app runs in **DEMO MODE** — the full
upload→predict→display pipeline works but predictions are deterministic
placeholders (clearly badged in the UI). To go live, export your model from the
notebook and drop it in `backend/models/` (see `backend/models/README.md`).

## Deploy

Any host that runs a Python web service works (Render, Railway, Fly.io, a VM).
Start command:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

(TensorFlow is only needed for live mode; for a pure demo deploy you can drop
the `tensorflow` line from requirements.txt to keep the image small.)

## Roadmap to a "real" product

This MVP is a portfolio-grade, deployable demo. To grow it further:

1. **Live model** — export and ship the trained weights.
2. **Robustness** — validate on external datasets; reject non-MRI uploads.
3. **Accounts & history** — let users save and revisit past scans.
4. **Reporting** — generate a downloadable PDF report per scan.
5. **Positioning** — keep it as clinical *decision-support / research*, not
   diagnosis. A genuine diagnostic device requires clinical validation and
   regulatory clearance (FDA / CDSCO), which is a funded, multi-year effort.

"""
AlzDetect API — FastAPI backend serving the Alzheimer's MRI classifier
and the static web frontend.

NOT FOR CLINICAL USE. This is a research / educational demonstration tool.
"""

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import model

app = FastAPI(
    title="AlzDetect API",
    description="Alzheimer's stage classification from MRI scans "
    "(MobileNetV2). Research/educational use only — not a medical device.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.on_event("startup")
def _startup():
    model.load_model()


@app.get("/api/health")
def health():
    return {"status": "ok", "demo_mode": model.is_demo_mode()}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        result = model.predict(data)
        result["heatmap"] = model.gradcam(data)
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")
    result["disclaimer"] = (
        "Research/educational tool only. Not a medical device and not a "
        "substitute for professional diagnosis."
    )
    return result


# Serve the frontend (mounted last so /api/* takes precedence).
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

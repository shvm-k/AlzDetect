# Publishing AlzDetect to Hugging Face Spaces

This gets the app live on a public URL like
`https://huggingface.co/spaces/<your-username>/AlzDetect`.

## Step 1 — Export your trained model (one time)

Your notebook trains the model but never saves it. At the end of
`Alzheimer's_Fuzzy-Logic+MobileNetV2 (1).ipynb`, add a cell and run it:

```python
model.save("alz_mobilenetv2.keras")
```

Download the resulting `alz_mobilenetv2.keras` file.

## Step 2 — Host the weights file

The Space downloads the model at startup from a `MODEL_URL`. Upload your
`.keras` file somewhere that gives a **direct download link**. Easiest option,
a Hugging Face model repo:

1. Go to https://huggingface.co/new → create a **Model** repo (e.g. `alzdetect-weights`).
2. Upload `alz_mobilenetv2.keras` via the web UI.
3. Copy the file's direct URL — it looks like:
   `https://huggingface.co/<user>/alzdetect-weights/resolve/main/alz_mobilenetv2.keras`

(Any direct-download URL works — S3, a GitHub Release asset, etc.)

## Step 3 — Create the Space

1. Go to https://huggingface.co/new-space
2. Name: `AlzDetect`  ·  SDK: **Docker**  ·  visibility: Public
3. Create it (it starts empty).

## Step 4 — Push the code to the Space

In a terminal, from your local clone of this repo:

```bash
# Add the Space as a second remote (use the URL HF shows you)
git remote add space https://huggingface.co/spaces/<your-username>/AlzDetect

# The Space needs its README.md to have HF frontmatter:
cp deploy/huggingface-space-README.md README.md   # (or merge into your README)

git add README.md && git commit -m "HF Space config"
git push space claude/upbeat-newton-f7gyub:main
```

If prompted, log in with a Hugging Face access token
(https://huggingface.co/settings/tokens, "write" scope).

## Step 5 — Set the model URL

In the Space → **Settings** → **Variables and secrets** → add:

```
MODEL_URL = https://huggingface.co/<user>/alzdetect-weights/resolve/main/alz_mobilenetv2.keras
```

The Space rebuilds, downloads the weights on boot, and the badge flips from
"DEMO MODE" to live predictions. Done — share the URL.

---

### Want demo-mode-only (no model, publish in 2 minutes)?

Skip steps 1, 2 and 5. The app deploys and works with placeholder predictions
(clearly badged). To make the image smaller, delete the `tensorflow` line from
`backend/requirements.txt` before pushing.

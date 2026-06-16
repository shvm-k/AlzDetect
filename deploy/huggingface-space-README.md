---
title: AlzDetect
emoji: 🧠
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
thumbnail: https://raw.githubusercontent.com/shvm-k/AlzDetect/main/frontend/avatar.png
---

# AlzDetect — Alzheimer's MRI Stage Classifier

Upload an MRI brain scan to get a predicted dementia stage (Mild / Moderate /
Very mild / Non Demented), per-class confidence, and a Grad-CAM attention map.

⚠️ **Research / educational tool only. Not a medical device. Not for clinical diagnosis.**

Model: MobileNetV2 (224px, frozen) + feature-space SMOTE + a trainable fuzzy
inference head (v6, ~78% held-out accuracy). Built with FastAPI.
Source: https://github.com/shvm-k/AlzDetect


# 🧠 Alzheimer's Disease Detection using MobileNetV2 + Fuzzy Logic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20722471.svg)](https://doi.org/10.5281/zenodo.20722471)

This project presents a deep learning-based approach to classify the stages of Alzheimer's Disease from MRI brain scans using **MobileNetV2**, enhanced with **Fuzzy Logic** for improved interpretability and precision.


## Overview

Alzheimer’s Disease (AD) is a progressive neurological disorder. Early and accurate diagnosis is critical for patient care. This model classifies MRI images into four classes:
- **Mild Dementia**
- **Moderate Dementia**
- **Non-Demented**
- **Very Mild Dementia**

The pipeline combines a **frozen MobileNetV2** feature extractor with **two
fuzzy-logic stages**: a fuzzy controller that rebalances the training data, and a
trainable fuzzy inference head that makes the final classification.

## 📂 Dataset

The dataset used is the **Alzheimer MRI Dataset** from [Kaggle](https://www.kaggle.com/datasets/legendahmed/alzheimermridataset) (OASIS-derived axial MRI slices across four stages).



## 🚀 Model Architecture

### MobileNetV2 CNN
- Lightweight CNN optimized for mobile & edge devices
- Transfer learning with fine-tuning

### Fuzzy Logic System
- Extracted features from the CNN are fed into a fuzzy inference system
- Improves classification robustness, especially on borderline cases


## 📊 Results

Measured from the project notebooks (macro-averaged over the 4 classes):

| Configuration                              | Evaluation        | Accuracy | Macro-F1 | Moderate-Dem. Recall |
|--------------------------------------------|-------------------|----------|----------|----------------------|
| MobileNetV2 (naive baseline)               | full, imbalanced  | 56%      | 0.25     | 0%                   |
| **AlzDetect** (fuzzy + feature-space SMOTE, 224px) | held-out split | **78%**\* | **0.785** | **0.97**            |

> \*78% is the **best of a 12-seed search** (typical macro-F1 ≈ 0.72), measured
> **in-distribution** on a held-out split of a single MRI source. The naive
> baseline row is evaluated on the full imbalanced set and is shown only to
> motivate the rebalancing.

### How the deployed model works

The model served by the web app (`backend/models/alz_mobilenetv2.keras`) uses a
**real fuzzy inference head** in the classification path (see
`experiments/train_fuzzy_model_v6.py` and `backend/fuzzy_layer.py`):

**224×224** frozen MobileNetV2 features → SMOTE balancing in feature space → a
small Dense projection → a trainable **Takagi–Sugeno–Kang fuzzy layer** (Gaussian
membership functions → fuzzy rule firing → defuzzification), with augmentation
restricted to ≤5° rotation + horizontal flip.

It scores **78% accuracy / 0.785 macro-F1 / 0.97 Moderate-Dem. recall** on its
held-out split, with Very-mild-Dementia F1 of **0.63**. Raising the input to
MobileNetV2's native 224×224 resolution improved *every* class. The fuzzy head is
small and init-sensitive, so the reported figure is the **best of a 12-seed
search** (typical macro-F1 ≈ 0.72) and is an optimistic point estimate. See
`experiments/Train_AlzDetect_v6_Kaggle.ipynb` for the best-of-N training loop, and
`docs/METHODOLOGY.md` / `paper/` for full details.

> **Generalization caveat:** this score is *in-distribution only* — the model was
> trained and evaluated on a single MRI source. On scans from other
> scanners/datasets it can be confidently wrong (a classic domain-shift /
> shortcut-learning failure). Note the fuzzy head is genuine fuzzy logic *in the
> decision path* — distinct from the skfuzzy resampling controller, which only
> sets per-class target counts.

## 📚 Citation

This work is archived on Zenodo: **https://doi.org/10.5281/zenodo.20722471**

If you use this code or the paper, please cite:

```bibtex
@misc{shivam2026alzdetect,
  title        = {Fuzzy-Augmented MobileNetV2 for Four-Stage Alzheimer's Disease
                  Classification from MRI: A Reproducible Pipeline with Honest
                  Benchmarking},
  author       = {Shivam, Nilabh},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20722471},
  url          = {https://doi.org/10.5281/zenodo.20722471}
}
```

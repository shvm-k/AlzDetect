# 🧠 Alzheimer's Disease Detection using MobileNetV2 + Fuzzy Logic

This project presents a deep learning-based approach to classify the stages of Alzheimer's Disease from MRI brain scans using **MobileNetV2**, enhanced with **Fuzzy Logic** for improved interpretability and precision.


## Overview

Alzheimer’s Disease (AD) is a progressive neurological disorder. Early and accurate diagnosis is critical for patient care. This model classifies MRI images into four classes:
- **Mild Dementia**
- **Moderate Dementia**
- **Non-Demented**
- **Very Mild Dementia**

Two models were developed:
- ✅ **MobileNetV2-based CNN**
- ✅ **MobileNetV2 + Fuzzy Logic post-processing**

Fuzzy logic is applied after the CNN to enhance decision boundaries based on feature outputs.

## 📂 Dataset

The dataset used is the **Alzheimer MRI Dataset** from [Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).



## 🚀 Model Architecture

### MobileNetV2 CNN
- Lightweight CNN optimized for mobile & edge devices
- Transfer learning with fine-tuning

### Fuzzy Logic System
- Extracted features from the CNN are fed into a fuzzy inference system
- Improves classification robustness, especially on borderline cases


## 📊 Results

Measured from the project notebooks (macro-averaged over the 4 classes):

| Model                       | Dataset            | Accuracy | Macro-F1 | Moderate-Dem. Recall |
|----------------------------|--------------------|----------|----------|----------------------|
| MobileNetV2 (baseline)     | full, imbalanced   | 56%      | 0.25     | 0%                   |
| MobileNetV2 + Fuzzy Logic  | balanced subset    | **92%**  | **0.93** | **92%**              |

> ⚠️ The two rows are **not a like-for-like comparison** (they differ in dataset
> size, evaluation distribution, and architecture). See `paper/` for the full
> manuscript and a "Threats to Validity" discussion. A controlled comparison
> script is in `experiments/`.

> 🔁 **Note on the live deployed model:** the original `sachinkumar413/alzheimer-mri-dataset`
> used above has since been removed from Kaggle. The model currently served by
> the web app (`backend/models/alz_mobilenetv2.keras`) was retrained on a mirror
> dataset (`legendahmed/alzheimermridataset`) using the same fuzzy-resampling
> pipeline, with a **real fuzzy inference head** in the classification path
> (see `experiments/train_fuzzy_model_v6.py` and `backend/fuzzy_layer.py`):
> **224×224** frozen MobileNetV2 features → SMOTE balancing in feature space →
> a small Dense projection → a trainable Takagi–Sugeno–Kang fuzzy layer
> (Gaussian membership functions → fuzzy rule firing → defuzzification), with
> augmentation restricted to ≤5° rotation + horizontal flip. It scores
> **76% accuracy / 0.75 macro-F1 / 0.97 Moderate-Dem. recall** on its held-out
> split — up from 72% at 128×128 input (raising the resolution to MobileNetV2's
> native size improved *every* class). Note the fuzzy head is genuine fuzzy
> logic *in the decision path* — distinct from the skfuzzy resampling
> controller, which only sets per-class target counts. The 92% figure above is
> not reproducible until the original dataset resurfaces or an equivalent
> replacement is found.

<p align="left">
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/65c080bc-e087-465c-9ec4-0b93c34dc389"/>
</p>

<p align="left">
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/1f136f58-f8c6-45ba-acaa-02745cf848d2" />
</p>

<p align="left">
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/b4c67d4f-9417-43eb-9da5-76379e656b84" />
</p>

# Methodology

This document describes the data-loading protocol, model architecture, training
procedure, and evaluation for the AlzDetect Alzheimer's-stage MRI classifier, as
implemented in `experiments/train_fuzzy_model_v6.py` /
`experiments/Train_AlzDetect_v6_Kaggle.ipynb` and served by `backend/`.

## 1. Dataset

We use the four-stage axial brain-MRI dataset `legendahmed/alzheimermridataset`
(a mirror of the OASIS-derived, augmented Alzheimer's MRI set; the original
`sachinkumar413/alzheimer-mri-dataset` was removed from Kaggle). The four classes
are **Mild Dementia**, **Moderate Dementia**, **Very mild Dementia**, and
**Non Demented**. Images are read from the flat `all image/` directory, in which
the class is encoded in the filename prefix.

Per-class image counts are severely imbalanced:

| Class | Count |
|---|---|
| Non Demented | 2560 |
| Very mild Dementia | 1792 |
| Mild Dementia | 717 |
| Moderate Dementia | 52 |

## 2. Data-loading protocol (deterministic label mapping)

To avoid the well-known *alphabetical-sort label drift* that occurs when a
pipeline is moved between environments or readers (e.g. `ImageFolder` /
`flow_from_directory` deriving class indices from OS-dependent directory
ordering), we **do not** infer labels from sorted directory traversal. Instead,
class membership is resolved explicitly from each filename via a regular
expression and a fixed prefix-to-class dictionary:

```python
PREFIX_TO_CLASS = {
    "mildDem":     "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem":      "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia",
               "Very mild Dementia", "Non Demented"]

m = re.match(r"^[a-zA-Z]+", filename)        # leading alphabetic prefix
cls = PREFIX_TO_CLASS.get(m.group())          # explicit class lookup
label_index = CLASS_ORDER.index(cls)          # fixed integer index
```

The integer label of every image is therefore its position in the **fixed**
`CLASS_ORDER` list, independent of filesystem ordering or platform. The inference
backend (`backend/model.py`) hard-codes the identical order in `CLASS_NAMES`, and
a startup guardrail asserts that the loaded model's output width equals
`len(CLASS_NAMES)`, refusing to serve (falling back to a clearly-badged demo
mode) if the two ever drift apart. This guarantees the index→label mapping is
consistent end-to-end between training and serving.

## 3. Fuzzy-logic resampling controller

Class imbalance is addressed first by a Mamdani fuzzy inference system
(scikit-fuzzy) that sets a per-class target count. For each class it computes an
imbalance score `1 − n_c / n_max` and maps it, through three triangular
membership functions (low/medium/high) and three rules, to a `resample_factor`
that determines the class's training target. This is the *first* of two
fuzzy-logic stages and operates only on dataset composition, not on predictions.

## 4. Image preparation and medical-safe augmentation

Minority classes are expanded to their fuzzy targets using **structure-preserving
augmentation only**: random rotation of at most ±5° (simulating slight head tilt)
and horizontal flip (exploiting approximate brain symmetry). Brightness,
contrast, zoom, and shift augmentations are deliberately excluded, as they wash
out the subtle gray-matter boundaries that distinguish adjacent dementia stages.
Images are resized to **224×224** — MobileNetV2's native input resolution — and
kept as raw 0–255 pixel values (consistent between training and serving).
Reducing resolution to 128×128 was found to materially degrade every class,
particularly the early stages, so the native resolution is retained.

## 5. Feature extraction and feature-space balancing (SMOTE)

A MobileNetV2 backbone (ImageNet weights, frozen, no top) followed by global
average pooling extracts a 1280-dimensional feature vector per image. The data is
split 80/20 (stratified). Class balance for training is then achieved by applying
**SMOTE in feature space** to the extracted training vectors — synthesizing
balanced minority-class clusters without warping the frozen backbone's gradients.
Earlier experiments using loss-side reweighting (`class_weight`) and aggressive
minority over-sampling beyond balance, combined with focal loss, were found to
*degrade* performance by inducing a lazy majority-guess bias; these were
therefore rejected in favor of clean feature-space balancing.

## 6. Fuzzy inference classification head

The balanced feature vectors are projected by a small `Dense(8, ReLU)` layer and
classified by a trainable **Takagi–Sugeno–Kang (TSK) fuzzy inference layer**
(`backend/fuzzy_layer.py`) — the *second* fuzzy-logic stage, and the one that
produces the final decision. The layer maintains `R = 16` rules; each rule `r`
has, per input dimension `d`, a Gaussian membership function with trainable
center `c[r,d]` and width `σ[r,d]`:

```
μ[r,d](x) = exp(−½ ((x_d − c[r,d]) / σ[r,d])²)
```

Rule firing strength is the product of per-dimension memberships (fuzzy AND,
computed in log-space for stability), normalized across rules. The class output
is the firing-weighted sum of per-rule consequent vectors, followed by softmax
(order-0 TSK defuzzification). Centers, widths, and consequents are all learned
by backpropagation. The layer self-registers via
`register_keras_serializable`, so the exported single end-to-end model
(frozen backbone → projection → fuzzy head) deserializes in the backend without
custom-object plumbing.

## 7. Training and seed selection

The head is trained with Adam (lr 1e-3), categorical cross-entropy, and early
stopping on validation loss. Because the fuzzy head is small and
initialization-sensitive, a single run is high-variance (observed val macro-F1
spanning ≈0.67–0.79 across seeds). We therefore train it across **N = 12 random
seeds** and select the seed with the highest validation macro-F1, reporting both
the selected (best) result and the typical (mean) result for transparency. Note
that selecting on the held-out split makes the *best* figure an optimistic point
estimate.

## 8. Results (in-distribution)

On the held-out 20% split, the selected model achieves:

| Metric | Value |
|---|---|
| Accuracy | 78% (best of 12 seeds; typical ≈72%) |
| Macro-F1 | 0.785 |
| Moderate-Dem. recall | 0.97 |
| Very-mild-Dem. F1 | 0.63 |

## 9. Threats to validity / limitations

All metrics above are **in-distribution**: training and evaluation draw from a
single MRI source. The model exploits cues localized to that distribution
(intensity histogram, contrast matrix, scanner/preprocessing characteristics)
rather than scanner-independent biological features. On external scans from other
institutions or preprocessing pipelines it can be **confidently wrong** — in one
test a textbook advanced-AD scan was classified *Non Demented* at 93%. This is
the signature of domain shift / shortcut learning, *not* a label-mapping error
(the index→label mapping is verified consistent end-to-end and in-distribution
accuracy is 78%, which would be impossible under an inverted map). Closing this
gap requires multi-source / multi-scanner training data or explicit
domain-generalization methods, not further architecture tuning. The historical
92% figure cited in earlier project notebooks was measured on a different,
now-removed dataset and is not reproducible with the current data.

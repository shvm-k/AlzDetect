# AlzDetect — The One Experiment That Unblocks the Paper

`run_experiments.py` produces **every number the paper needs**, honestly and
reproducibly, in a single run:

- baseline MobileNetV2 (no balancing) + class weights + random oversampling +
  SMOTE + focal loss + **fuzzy adaptive resampling (your method)**
- stratified **5-fold cross-validation**, mean ± std
- overall accuracy, macro-F1, per-class precision/recall/F1, confusion matrices
- exports **`alz_mobilenetv2.keras`** so the live demo uses the real model

Run it once → you get locked, trustworthy numbers AND the baseline comparison,
so the paper can *honestly* claim both. This is the thing standing between you
and a postable preprint.

## Easiest way: Google Colab (free GPU, no laptop dependency)

1. Open [colab.research.google.com](https://colab.research.google.com) → New notebook.
2. `Runtime → Change runtime type → GPU`.
3. Upload `run_experiments.py` (folder icon on the left), then run these cells:

```python
!pip install -q scikit-fuzzy imbalanced-learn opencv-python-headless

# --- get the dataset from Kaggle ---
# Upload your kaggle.json (Kaggle -> Account -> Create New API Token) first.
!pip install -q kaggle
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d sachinkumar413/alzheimer-mri-dataset -p data --unzip

import os
# Point this at the folder that directly contains the 4 class subfolders:
os.environ["DATA_DIR"] = "data"        # adjust if unzip nests it deeper
os.environ["EPOCHS"]   = "15"

!python run_experiments.py
```

4. When done, download `alz_mobilenetv2.keras` and `results/results_table.md`.

## Outputs
- `results/results.json` — full metrics (feeds the paper tables)
- `results/results_table.md` — the baseline-comparison table (Table II)
- `alz_mobilenetv2.keras` — trained model for the live demo

## Knobs (environment variables)
| Var | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data` | folder containing the 4 class subfolders |
| `EPOCHS` | `15` | training epochs per fold |
| `RUN_SMOTE` | `1` | set `0` to skip SMOTE (it's memory-heavy) |
| `GROUP_REGEX` | empty | regex capturing a subject id from filenames, for **patient-level** splits (prevents leakage) |

## ⚠️ Read this about leakage
By default the split is image-level. If filenames encode a subject/patient id
(e.g. `OAS1_0031_...`), set `GROUP_REGEX` (e.g. `(OAS\d+)`) so all of one
subject's slices stay on the same side of the split. Image-level splits on this
dataset inflate accuracy and reviewers/readers know it — even on arXiv, doing
this right protects your credibility.

## After it runs
Send me `results/results_table.md` and I'll:
1. drop the real numbers into the paper (one consistent set, everywhere),
2. fill the baseline table and reframe claims to match what actually ran,
3. make the manuscript arXiv-ready (endorsement notes, license, final pass).

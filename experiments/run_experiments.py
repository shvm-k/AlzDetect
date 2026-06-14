"""
AlzDetect — reproducible imbalance experiment.

Runs ONE clean, leakage-controlled benchmark that produces every number the
paper needs:
  * baseline MobileNetV2 with NO balancing
  * class weights
  * random oversampling
  * SMOTE (image-space)
  * focal loss
  * fuzzy adaptive resampling (our method)

For each strategy it reports, under stratified 5-fold cross-validation:
  overall accuracy, macro-F1, and per-class precision/recall/F1 (mean +/- std),
plus confusion matrices. Results are written to results/.

It also retrains the fuzzy model on ALL data at the end and saves
`alz_mobilenetv2.keras` — drop that into backend/models/ (or host it and set
MODEL_URL) to make the live demo use the real model.

IMPORTANT (anti-leakage): resampling is applied ONLY to each training fold,
never to the validation fold. If your filenames encode a patient/subject id,
set GROUP_REGEX so splits are patient-level — image-level splits on this dataset
inflate accuracy.

------------------------------------------------------------------------------
HOW TO RUN (Google Colab, free GPU — recommended):
  Runtime -> Change runtime type -> GPU, then:

    !pip install -q scikit-fuzzy imbalanced-learn
    # Option A: pull dataset from Kaggle (needs kaggle.json token)
    !pip install -q kaggle
    !kaggle datasets download -d sachinkumar413/alzheimer-mri-dataset -p data --unzip
    # Point DATA_DIR below at the folder that contains the 4 class subfolders.

    !python run_experiments.py
------------------------------------------------------------------------------
"""

import json
import os
import re
from collections import Counter

import numpy as np

# ----------------------------- CONFIG --------------------------------------
DATA_DIR   = os.environ.get("DATA_DIR", "data")  # contains 4 class subfolders
IMG_SIZE   = 128
FOLDS      = 5
EPOCHS     = int(os.environ.get("EPOCHS", "15"))
BATCH      = 32
SEED       = 42
RESULTS_DIR = "results"
GROUP_REGEX = os.environ.get("GROUP_REGEX", "")  # e.g. r"(OAS\d+)" for patient id
RUN_SMOTE  = os.environ.get("RUN_SMOTE", "1") == "1"  # SMOTE is memory-heavy

# Class order MUST match the live app (backend/model.py).
CLASS_NAMES = ["Mild Dementia", "Moderate Dementia",
               "Very mild Dementia", "Non Demented"]

np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------- DATA LOADING ----------------------------------
def find_class_dir(root, name):
    """Match a class folder tolerantly (case / spacing / underscores)."""
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    target = norm(name)
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and norm(d) == target:
            return os.path.join(root, d)
    # looser contains-match
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and target[:6] in norm(d):
            return os.path.join(root, d)
    raise FileNotFoundError(f"No folder for class '{name}' under {root}")


def load_dataset():
    import cv2
    X, y, groups = [], [], []
    for idx, cname in enumerate(CLASS_NAMES):
        cdir = find_class_dir(DATA_DIR, cname)
        for fn in sorted(os.listdir(cdir)):
            path = os.path.join(cdir, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)
            if GROUP_REGEX:
                m = re.search(GROUP_REGEX, fn)
                groups.append(m.group(1) if m else fn)
            else:
                groups.append(len(y))  # unique -> image-level split
    X = np.asarray(X, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    print(f"Loaded {len(y)} images. Class counts: {Counter(y.tolist())}")
    if not GROUP_REGEX:
        print("WARNING: image-level split (no GROUP_REGEX). Possible leakage — "
              "set GROUP_REGEX if filenames encode a subject id.")
    return X, y, groups


# --------------------------- FUZZY CONTROLLER ------------------------------
def fuzzy_target_counts(counts):
    """Return target per-class counts using the triangular-MF Mamdani system."""
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl

    imbalance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "imbalance")
    resample  = ctrl.Consequent(np.arange(0, 1.01, 0.01), "resample")
    imbalance["low"]    = fuzz.trimf(imbalance.universe, [0, 0, 0.5])
    imbalance["medium"] = fuzz.trimf(imbalance.universe, [0.3, 0.5, 0.7])
    imbalance["high"]   = fuzz.trimf(imbalance.universe, [0.5, 1, 1])
    resample["low"]     = fuzz.trimf(resample.universe, [0, 0, 0.3])
    resample["medium"]  = fuzz.trimf(resample.universe, [0.2, 0.5, 0.7])
    resample["high"]    = fuzz.trimf(resample.universe, [0.6, 1, 1])
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([
        ctrl.Rule(imbalance["low"], resample["low"]),
        ctrl.Rule(imbalance["medium"], resample["medium"]),
        ctrl.Rule(imbalance["high"], resample["high"]),
    ]))
    nmax = max(counts.values())
    targets = {}
    for c, n in counts.items():
        sim.input["imbalance"] = 1 - (n / nmax)
        sim.compute()
        targets[c] = int(n + sim.output["resample"] * nmax)
    return targets


# --------------------------- RESAMPLERS ------------------------------------
def augment(img):
    """Light label-preserving augmentation for oversampled copies."""
    import cv2
    if np.random.rand() < 0.5:
        img = img[:, ::-1]
    ang = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
    return img.astype(np.float32)


def oversample_to(Xtr, ytr, targets):
    """Oversample each class up to targets[c] with augmentation."""
    outX, outY = [Xtr], [ytr]
    for c, tgt in targets.items():
        idx = np.where(ytr == c)[0]
        need = tgt - len(idx)
        if need <= 0 or len(idx) == 0:
            continue
        pick = np.random.choice(idx, size=need, replace=True)
        outX.append(np.stack([augment(Xtr[i]) for i in pick]))
        outY.append(np.full(need, c))
    return np.concatenate(outX), np.concatenate(outY)


def resample_fold(Xtr, ytr, strategy):
    counts = dict(Counter(ytr.tolist()))
    nmax = max(counts.values())
    if strategy in ("none", "class_weights", "focal"):
        return Xtr, ytr
    if strategy == "random_oversample":
        return oversample_to(Xtr, ytr, {c: nmax for c in counts})
    if strategy == "fuzzy":
        return oversample_to(Xtr, ytr, fuzzy_target_counts(counts))
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        flat = Xtr.reshape(len(Xtr), -1)
        k = max(1, min(5, min(counts.values()) - 1))
        Xr, yr = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(flat, ytr)
        return Xr.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32), yr
    raise ValueError(strategy)


# --------------------------- MODEL -----------------------------------------
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return Model(base.input, out)


def focal_loss(gamma=2.0):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1 - y_pred, gamma) * ce, axis=-1)
    return loss


def train_eval(Xtr, ytr, Xva, yva, strategy):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.utils.class_weight import compute_class_weight

    model = build_model()
    loss = focal_loss() if strategy == "focal" else "categorical_crossentropy"
    model.compile(optimizer=Adam(1e-4), loss=loss, metrics=["accuracy"])

    cw = None
    if strategy == "class_weights":
        w = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        cw = {int(c): float(wi) for c, wi in zip(np.unique(ytr), w)}

    model.fit(Xtr, to_categorical(ytr, len(CLASS_NAMES)),
              validation_data=(Xva, to_categorical(yva, len(CLASS_NAMES))),
              epochs=EPOCHS, batch_size=BATCH, class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])
    proba = model.predict(Xva, verbose=0)
    return np.argmax(proba, axis=1), model


# --------------------------- CV DRIVER -------------------------------------
def run():
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support, confusion_matrix)

    X, y, groups = load_dataset()
    strategies = ["none", "class_weights", "random_oversample", "focal", "fuzzy"]
    if RUN_SMOTE:
        strategies.insert(3, "smote")

    use_groups = bool(GROUP_REGEX) and len(set(groups)) < len(groups)
    splitter = (StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
                if use_groups else
                StratifiedKFold(FOLDS, shuffle=True, random_state=SEED))
    splits = list(splitter.split(X, y, groups if use_groups else None))

    summary = {}
    for strat in strategies:
        accs, mf1s, cms = [], [], []
        per_class = {c: {"p": [], "r": [], "f": []} for c in range(len(CLASS_NAMES))}
        for fold, (tr, va) in enumerate(splits):
            Xr, yr = resample_fold(X[tr], y[tr], strat)
            ypred, _ = train_eval(Xr, yr, X[va], y[va], strat)
            accs.append(accuracy_score(y[va], ypred))
            mf1s.append(f1_score(y[va], ypred, average="macro"))
            p, r, f, _ = precision_recall_fscore_support(
                y[va], ypred, labels=range(len(CLASS_NAMES)), zero_division=0)
            for c in range(len(CLASS_NAMES)):
                per_class[c]["p"].append(p[c]); per_class[c]["r"].append(r[c]); per_class[c]["f"].append(f[c])
            cms.append(confusion_matrix(y[va], ypred, labels=range(len(CLASS_NAMES))))
            print(f"[{strat}] fold {fold+1}/{FOLDS}  acc={accs[-1]:.3f}  macroF1={mf1s[-1]:.3f}")
        summary[strat] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(mf1s)), "macro_f1_std": float(np.std(mf1s)),
            "per_class": {CLASS_NAMES[c]: {
                "precision": float(np.mean(v["p"])), "recall": float(np.mean(v["r"])),
                "f1": float(np.mean(v["f"]))} for c, v in per_class.items()},
            "confusion_matrix_sum": np.sum(cms, axis=0).tolist(),
        }
        print(f"==> {strat}: acc {summary[strat]['accuracy_mean']:.3f}"
              f"±{summary[strat]['accuracy_std']:.3f}  "
              f"macroF1 {summary[strat]['macro_f1_mean']:.3f}\n")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_table(summary)

    # Retrain fuzzy model on ALL data -> ship for the live demo.
    print("Retraining fuzzy model on all data for deployment...")
    Xr, yr = resample_fold(X, y, "fuzzy")
    _, final = train_eval(Xr, yr, X[:BATCH], y[:BATCH], "fuzzy")
    final.save("alz_mobilenetv2.keras")
    print("Saved alz_mobilenetv2.keras  ->  put in backend/models/ for live mode.")


def _write_table(summary):
    lines = ["| Method | Accuracy | Macro-F1 |", "|---|---|---|"]
    for s, m in summary.items():
        lines.append(f"| {s} | {m['accuracy_mean']*100:.1f}±{m['accuracy_std']*100:.1f}% "
                     f"| {m['macro_f1_mean']:.3f}±{m['macro_f1_std']:.3f} |")
    table = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "results_table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    run()

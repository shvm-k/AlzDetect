"""
Controlled cross-validation comparison of imbalance-handling strategies for
AlzDetect, for peer-review readiness.

This script replaces the single-split, best-of-N-seed protocol with stratified
k-fold cross-validation and reports mean +/- std, and it fills the paper's
"planned controlled comparison" table by varying ONLY the imbalance-handling
strategy on top of identical frozen MobileNetV2 features:

  1. No balancing        (Dense head, cross-entropy)
  2. Class weights       (Dense head, class-weighted cross-entropy)
  3. Random oversampling (RandomOverSampler on train features, Dense head)
  4. SMOTE               (SMOTE on train features, Dense head)
  5. Focal loss          (Dense head, focal loss, no resampling)
  6. Fuzzy (ours)        (SMOTE on train features + TSK fuzzy head)

Design choices that make this a FAIR comparison (and that reviewers ask for):
  * The MobileNetV2 backbone is frozen and features are extracted ONCE; feature
    extraction is not training, so there is no leakage across folds.
  * NO image-level augmentation is used here (augmentation is a confound); the
    comparison isolates the balancing/loss choice.
  * Every method shares the same Dense(8, relu) projection; rows 1-5 end in a
    softmax classifier, row 6 replaces it with the TSK fuzzy inference layer.
  * Resampling is fit on the TRAIN fold only, inside the CV loop.
  * We report per-fold accuracy and macro-F1, then mean +/- std across folds.

Usage (Kaggle/Colab, after the dataset is available):
  import os; os.environ["DATA_DIR"] = ".../all image"
  !pip install -q scikit-fuzzy imbalanced-learn opencv-python-headless
  !python experiments/crossval_baselines.py
"""

import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import fuzzy_layer  # noqa: E402  (registers FuzzyLayer)
FuzzyLayer = fuzzy_layer.FuzzyLayer

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = 32
FEATURE_BATCH = int(os.environ.get("FEATURE_BATCH", "32"))
PROJ_DIM = 8
N_RULES = 16
SEED = 42

PREFIX_TO_CLASS = {
    "mildDem": "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem": "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
np.random.seed(SEED)


def _bucket_files():
    buckets = {c: [] for c in CLASS_ORDER}
    for fn in sorted(os.listdir(DATA_DIR)):
        m = re.match(r"^[a-zA-Z]+", fn)
        if m and PREFIX_TO_CLASS.get(m.group()):
            buckets[PREFIX_TO_CLASS[m.group()]].append(fn)
    return buckets


def focal_loss(gamma=2.0):
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)

    return loss


def _dense_head(in_dim, n_classes=4):
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model
    inp = Input(shape=(in_dim,))
    x = Dense(PROJ_DIM, activation="relu")(inp)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)
    return Model(inp, out)


def _fuzzy_head(in_dim, n_classes=4):
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    inp = Input(shape=(in_dim,))
    x = Dense(PROJ_DIM, activation="relu")(inp)
    out = FuzzyLayer(n_rules=N_RULES, n_classes=n_classes)(x)
    return Model(inp, out)


def _fit_eval(Xtr, ytr, Xte, yte, method, seed):
    """Train one method on a fold; return (accuracy, macro_f1)."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    tf.keras.utils.set_random_seed(seed)

    # Standardize features per fold (fit on TRAIN only). Raw MobileNetV2 GAP
    # features are not zero-centred/unit-scaled; without this the small head
    # trained with Adam is unstable and can diverge on some folds.
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    in_dim = Xtr.shape[1]
    class_weight = None
    loss = "categorical_crossentropy"
    Xt, yt = Xtr, ytr

    if method == "No balancing":
        head = _dense_head(in_dim)
    elif method == "Class weights":
        head = _dense_head(in_dim)
        cw = compute_class_weight("balanced", classes=np.arange(4), y=ytr)
        class_weight = {i: w for i, w in enumerate(cw)}
    elif method == "Random oversampling":
        head = _dense_head(in_dim)
        Xt, yt = RandomOverSampler(random_state=seed).fit_resample(Xtr, ytr)
    elif method == "SMOTE":
        head = _dense_head(in_dim)
        k = max(1, min(5, np.bincount(ytr).min() - 1))
        Xt, yt = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)
    elif method == "Focal loss":
        head = _dense_head(in_dim)
        loss = focal_loss(2.0)
    elif method == "Fuzzy (ours)":
        head = _fuzzy_head(in_dim)
        k = max(1, min(5, np.bincount(ytr).min() - 1))
        Xt, yt = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)
    else:
        raise ValueError(method)

    # Internal stratified val split + early stopping -> stable, no fold collapse.
    Xt2, Xv, yt2, yv = train_test_split(
        Xt, yt, test_size=0.1, stratify=yt, random_state=seed)
    head.compile(optimizer=Adam(5e-4), loss=loss, metrics=["accuracy"])
    head.fit(Xt2, to_categorical(yt2, 4),
             validation_data=(Xv, to_categorical(yv, 4)),
             epochs=80, batch_size=BATCH, verbose=0, class_weight=class_weight,
             callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True)])
    yp = np.argmax(head.predict(Xte, verbose=0), axis=1)
    return accuracy_score(yte, yp), f1_score(yte, yp, average="macro")


def main():
    import cv2
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    from sklearn.model_selection import StratifiedKFold

    # ---- load ORIGINAL images only (no augmentation) ----
    buckets = _bucket_files()
    class_map = {c: i for i, c in enumerate(CLASS_ORDER)}
    imgs, labels = [], []
    for label in CLASS_ORDER:
        for fn in buckets[label]:
            im = cv2.imread(os.path.join(DATA_DIR, fn))
            if im is not None:
                imgs.append(cv2.resize(im, (IMG_SIZE, IMG_SIZE)))
                labels.append(class_map[label])
    X_img = np.array(imgs, dtype=np.uint8)
    y = np.array(labels)
    print("Loaded:", X_img.shape, {CLASS_ORDER[i]: int((y == i).sum()) for i in range(4)})

    # ---- frozen feature extraction (once) ----
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inp)
    base.trainable = False
    feat_model = Model(inp, GlobalAveragePooling2D()(base.output))
    print("Extracting frozen features ...")
    F = feat_model.predict(X_img, batch_size=FEATURE_BATCH, verbose=1)
    del X_img

    methods = ["No balancing", "Class weights", "Random oversampling",
               "SMOTE", "Focal loss", "Fuzzy (ours)"]
    results = {m: {"acc": [], "f1": []} for m in methods}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(F, y)):
        print(f"\n=== Fold {fold + 1}/{N_SPLITS} ===")
        for m in methods:
            acc, f1 = _fit_eval(F[tr], y[tr], F[te], y[te], m, seed=SEED + fold)
            results[m]["acc"].append(acc)
            results[m]["f1"].append(f1)
            print(f"  {m:<20s} acc={acc:.4f}  macroF1={f1:.4f}")

    print("\n================  CROSS-VALIDATED RESULTS  ================")
    print(f"{'Method':<22s}{'Accuracy (mean+/-std)':<26s}{'Macro-F1 (mean+/-std)'}")
    print("-" * 70)
    for m in methods:
        a = np.array(results[m]["acc"]); f = np.array(results[m]["f1"])
        print(f"{m:<22s}{a.mean():.3f} +/- {a.std():.3f}           "
              f"{f.mean():.3f} +/- {f.std():.3f}")

    # CSV for the paper table
    with open("crossval_results.csv", "w") as fh:
        fh.write("method,acc_mean,acc_std,f1_mean,f1_std\n")
        for m in methods:
            a = np.array(results[m]["acc"]); f = np.array(results[m]["f1"])
            fh.write(f"{m},{a.mean():.4f},{a.std():.4f},{f.mean():.4f},{f.std():.4f}\n")
    print("\nSaved crossval_results.csv")


if __name__ == "__main__":
    main()

"""
Run 4 / v3 training for alz_mobilenetv2.keras -- the "medical-safe" plan.

Run 2 (full pool + file-repetition oversampling) is the true 68% baseline.
Run 3 (heavy augmentation + class_weight + fine-tuning) dropped to 66% because
each of those three levers fought the fuzzy rebalancing:
  - aggressive augmentation washed out subtle gray-matter boundaries,
  - class_weight warped the gradients the fuzzy targets had already balanced,
  - unfreezing layers caused catastrophic forgetting of ImageNet edge filters.

This script keeps what worked and changes only the imbalance handling:

  1. MEDICAL-SAFE AUGMENTATION ONLY: <=5 deg rotation + horizontal flip.
     No brightness / contrast / zoom / shift. Applied once to materialise a
     modest, structure-preserving expansion of the minority classes.
  2. SMOTE IN FEATURE SPACE, NOT class_weight: freeze MobileNetV2, extract the
     pooled feature vectors, then SMOTE-balance those vectors. Clean, balanced
     clusters for the head to separate without touching the CNN gradients.
  3. FROZEN BASE + TINY ADAPTER: MobileNetV2 stays fully frozen (keeps the fast
     ~3-min train and its edge detectors). A single Dense(32, swish) adapter
     translates frozen vision features before the softmax head.

The exported alz_mobilenetv2.keras is a single end-to-end Keras model
(frozen MobileNetV2 -> GAP -> Dense(32, swish) -> Dropout -> Dense(4, softmax))
so it stays a drop-in replacement for backend/model.py -- the SMOTE step only
shapes the *training* set, it is not part of the served graph.

Usage (same Colab session, after downloading legendahmed/alzheimermridataset):
  import os
  os.environ["DATA_DIR"] = "data/Alzheimer_s Dataset/all image"
  !pip install -q imbalanced-learn
  !python train_fuzzy_model_v3.py
"""

import os
import re
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = 32
SEED = 42

PREFIX_TO_CLASS = {
    "mildDem": "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem": "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]

np.random.seed(SEED)


def fuzzy_targets(counts):
    """Same skfuzzy Mamdani controller as the notebook / Run 2."""
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl

    imbalance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "imbalance")
    resample = ctrl.Consequent(np.arange(0, 1.1, 0.1), "resample_factor")
    imbalance["low"] = fuzz.trimf(imbalance.universe, [0, 0, 0.5])
    imbalance["medium"] = fuzz.trimf(imbalance.universe, [0.3, 0.5, 0.7])
    imbalance["high"] = fuzz.trimf(imbalance.universe, [0.5, 1, 1])
    resample["low"] = fuzz.trimf(resample.universe, [0, 0, 0.3])
    resample["medium"] = fuzz.trimf(resample.universe, [0.2, 0.5, 0.7])
    resample["high"] = fuzz.trimf(resample.universe, [0.6, 1, 1])
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([
        ctrl.Rule(imbalance["low"], resample["low"]),
        ctrl.Rule(imbalance["medium"], resample["medium"]),
        ctrl.Rule(imbalance["high"], resample["high"]),
    ]))
    nmax = max(counts.values())
    out = {}
    for c, n in counts.items():
        sim.input["imbalance"] = 1 - (n / nmax)
        sim.compute()
        out[c] = int(n + sim.output["resample_factor"] * nmax)
    return out


def _bucket_files():
    buckets = {c: [] for c in CLASS_ORDER}
    for fn in sorted(os.listdir(DATA_DIR)):
        m = re.match(r"^[a-zA-Z]+", fn)
        if not m:
            continue
        cls = PREFIX_TO_CLASS.get(m.group())
        if cls:
            buckets[cls].append(fn)
    return buckets


def _medical_safe_aug(img, rng):
    """<=5 deg rotation + optional horizontal flip. Nothing else."""
    import cv2

    angle = rng.uniform(-5, 5)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    return img


def main():
    import cv2
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from imblearn.over_sampling import SMOTE

    rng = np.random.default_rng(SEED)

    buckets = _bucket_files()
    counts = {c: len(files) for c, files in buckets.items()}
    print("Real counts:", counts)

    # Fuzzy controller still decides how far to push each class, but we cap the
    # synthetic blow-up: read real images once, add only medical-safe variants
    # up to the fuzzy target (SMOTE handles the rest in feature space later).
    targets = fuzzy_targets(counts)
    print("Fuzzy target counts:", targets)

    class_map = {label: i for i, label in enumerate(CLASS_ORDER)}
    data, labels = [], []
    for label in CLASS_ORDER:
        files = buckets[label]
        target = targets[label]
        # Load every real image once.
        imgs = []
        for fn in files:
            img = cv2.imread(os.path.join(DATA_DIR, fn))
            if img is None:
                continue
            imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        for im in imgs:
            data.append(im)
            labels.append(class_map[label])
        # Top up minority classes with medical-safe augmented copies (cap at the
        # fuzzy target). Majority classes already exceed target -> no aug.
        i = 0
        while len(imgs) and (len([l for l in labels if l == class_map[label]]) < target):
            data.append(_medical_safe_aug(imgs[i % len(imgs)], rng))
            labels.append(class_map[label])
            i += 1

    data = np.array(data, dtype=np.float32)  # RAW 0-255 (matches backend/model.py)
    y_int = np.array(labels)
    print("Image-level counts after safe aug:",
          {CLASS_ORDER[i]: int((y_int == i).sum()) for i in range(len(CLASS_ORDER))})

    Xtr_img, Xte_img, ytr_int, yte_int = train_test_split(
        data, y_int, test_size=0.2, stratify=y_int, random_state=SEED)

    # ---- Frozen MobileNetV2 feature extractor (kept as the served base) ----
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inp)
    base.trainable = False
    pooled = GlobalAveragePooling2D()(base.output)
    feature_model = Model(inp, pooled)

    print("\nExtracting frozen features ...")
    Ftr = feature_model.predict(Xtr_img, batch_size=BATCH, verbose=1)
    Fte = feature_model.predict(Xte_img, batch_size=BATCH, verbose=1)

    # ---- SMOTE in feature space (train only) ----
    # k must be < smallest class count; Moderate is tiny, so clamp defensively.
    min_class = min(int((ytr_int == i).sum()) for i in range(len(CLASS_ORDER)))
    k = max(1, min(5, min_class - 1))
    print(f"SMOTE k_neighbors={k} (smallest train class={min_class})")
    Ftr_bal, ytr_bal = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Ftr, ytr_int)
    print("Balanced feature counts:",
          {CLASS_ORDER[i]: int((ytr_bal == i).sum()) for i in range(len(CLASS_ORDER))})

    ytr_cat = to_categorical(ytr_bal, num_classes=len(CLASS_ORDER))
    yte_cat = to_categorical(yte_int, num_classes=len(CLASS_ORDER))

    # ---- Tiny adapter head trained on balanced feature vectors ----
    feat_in = Input(shape=(Ftr.shape[1],))
    h = Dense(32, activation="swish")(feat_in)
    h = Dropout(0.3)(h)
    head_out = Dense(len(CLASS_ORDER), activation="softmax")(h)
    head = Model(feat_in, head_out)
    head.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy",
                 metrics=["accuracy"])

    head.fit(Ftr_bal, ytr_cat, validation_data=(Fte, yte_cat),
             epochs=EPOCHS, batch_size=BATCH,
             callbacks=[EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True)])

    yp = np.argmax(head.predict(Fte), axis=1)
    print("\nClassification report:")
    print(classification_report(yte_int, yp, target_names=CLASS_ORDER))

    # ---- Stitch frozen base + trained head into one served model ----
    full_out = head(feature_model(inp))
    full = Model(inp, full_out)
    full.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras  ->  send it back / host it for MODEL_URL.")


if __name__ == "__main__":
    main()

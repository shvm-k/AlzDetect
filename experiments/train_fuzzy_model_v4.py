"""
Run 5 / v4 training -- adds a REAL fuzzy inference head to the pipeline.

Up to v3 the only "fuzzy" component was the skfuzzy resampling controller that
picks per-class target counts; the actual classifier was a plain Dense head.
This version keeps the winning v3 recipe (medical-safe aug + frozen MobileNetV2
features + SMOTE balancing) but replaces the Dense head with a trainable
Takagi-Sugeno-Kang fuzzy inference layer (Gaussian membership functions ->
fuzzy rule firing -> defuzzification). See backend/fuzzy_layer.py for the math.

Pipeline:
  frozen MobileNetV2 -> GAP -> Dense(PROJ_DIM, relu)   [feature projection]
                            -> FuzzyLayer(n_rules=R)    [fuzzy inference head]
                            -> class probabilities

The exported alz_mobilenetv2.keras is a single drop-in model. Because
FuzzyLayer self-registers (@register_keras_serializable), backend/model.py
loads it with no custom_objects -- it just imports backend/fuzzy_layer.

Usage (clone the repo in Colab so this script can import the shared layer):
  !git clone -b claude/upbeat-newton-f7gyub https://github.com/shvm-k/AlzDetect.git
  %cd AlzDetect
  import os; os.environ["DATA_DIR"] = "data/Alzheimer_s Dataset/all image"
  !pip install -q scikit-fuzzy imbalanced-learn opencv-python-headless
  !python experiments/train_fuzzy_model_v4.py
"""

import os
import re
import sys
import numpy as np

# Import the SAME fuzzy layer the backend serves with (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import fuzzy_layer  # noqa: E402
FuzzyLayer = fuzzy_layer.FuzzyLayer

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BATCH = 32
SEED = 42
PROJ_DIM = int(os.environ.get("PROJ_DIM", "8"))   # fuzzy input dim (keep small)
N_RULES = int(os.environ.get("N_RULES", "16"))    # number of fuzzy rules

PREFIX_TO_CLASS = {
    "mildDem": "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem": "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]

np.random.seed(SEED)


def fuzzy_targets(counts):
    """skfuzzy Mamdani controller: imbalance -> resample_factor (per-class targets)."""
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
        if m and PREFIX_TO_CLASS.get(m.group()):
            buckets[PREFIX_TO_CLASS[m.group()]].append(fn)
    return buckets


def _medical_safe_aug(img, rng):
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
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from imblearn.over_sampling import SMOTE

    rng = np.random.default_rng(SEED)
    buckets = _bucket_files()
    counts = {c: len(f) for c, f in buckets.items()}
    print("Real counts:", counts)
    targets = fuzzy_targets(counts)
    print("Fuzzy target counts:", targets)

    class_map = {label: i for i, label in enumerate(CLASS_ORDER)}
    data, labels = [], []
    for label in CLASS_ORDER:
        files, target = buckets[label], targets[label]
        imgs = []
        for fn in files:
            im = cv2.imread(os.path.join(DATA_DIR, fn))
            if im is not None:
                imgs.append(cv2.resize(im, (IMG_SIZE, IMG_SIZE)))
        for im in imgs:
            data.append(im); labels.append(class_map[label])
        i = 0
        while imgs and sum(1 for l in labels if l == class_map[label]) < target:
            data.append(_medical_safe_aug(imgs[i % len(imgs)], rng))
            labels.append(class_map[label]); i += 1

    data = np.array(data, dtype=np.float32)  # RAW 0-255 (matches backend/model.py)
    y_int = np.array(labels)
    print("After safe aug:", {CLASS_ORDER[i]: int((y_int == i).sum()) for i in range(4)})

    Xtr_img, Xte_img, ytr_int, yte_int = train_test_split(
        data, y_int, test_size=0.2, stratify=y_int, random_state=SEED)

    # Frozen MobileNetV2 feature extractor (served as-is).
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inp)
    base.trainable = False
    pooled = GlobalAveragePooling2D()(base.output)
    feature_model = Model(inp, pooled)

    print("\nExtracting frozen features ...")
    Ftr = feature_model.predict(Xtr_img, batch_size=BATCH, verbose=1)
    Fte = feature_model.predict(Xte_img, batch_size=BATCH, verbose=1)

    min_class = min(int((ytr_int == i).sum()) for i in range(4))
    k = max(1, min(5, min_class - 1))
    print(f"SMOTE k_neighbors={k} (smallest train class={min_class})")
    Ftr_bal, ytr_bal = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Ftr, ytr_int)
    print("Balanced:", {CLASS_ORDER[i]: int((ytr_bal == i).sum()) for i in range(4)})

    ytr_cat = to_categorical(ytr_bal, num_classes=4)
    yte_cat = to_categorical(yte_int, num_classes=4)

    # ---- Fuzzy inference head: projection -> FuzzyLayer ----
    feat_in = Input(shape=(Ftr.shape[1],))
    proj = Dense(PROJ_DIM, activation="relu", name="fuzzy_projection")(feat_in)
    head_out = FuzzyLayer(n_rules=N_RULES, n_classes=4, name="fuzzy_head")(proj)
    head = Model(feat_in, head_out)
    head.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy",
                 metrics=["accuracy"])
    print(f"\nFuzzy head: PROJ_DIM={PROJ_DIM}, N_RULES={N_RULES}")

    head.fit(Ftr_bal, ytr_cat, validation_data=(Fte, yte_cat),
             epochs=EPOCHS, batch_size=BATCH,
             callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True)])

    yp = np.argmax(head.predict(Fte), axis=1)
    print("\nClassification report:")
    print(classification_report(yte_int, yp, target_names=CLASS_ORDER))

    # Stitch frozen base + fuzzy head into one served model.
    full = Model(inp, head(feature_model(inp)))
    full.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras (fuzzy head) -> send it back / host for MODEL_URL.")


if __name__ == "__main__":
    main()

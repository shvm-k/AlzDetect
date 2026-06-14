"""
Variant of train_fuzzy_model.py for the "all image" flat-folder mirror of the
Alzheimer MRI dataset (legendahmed/alzheimermridataset), where the original
sachinkumar413 dataset's per-class folders are unavailable.

Files live directly under DATA_DIR with class encoded in the filename prefix:
  mildDem*, moderateDem*, verymildDem*, nonDem*

Real per-class counts in that folder:
  mildDem: 717, moderateDem: 52, verymildDem: 1792, nonDem: 2560

Unlike train_fuzzy_model.py (which truncates to the fuzzy target), this script
OVERSAMPLES minority classes by cycling through their files with repetition
when the fuzzy target exceeds the available count -- this is what actually
balances classes like Moderate Dementia (52 real images).

Usage (same Colab session, after downloading legendahmed/alzheimermridataset):
  import os
  os.environ["DATA_DIR"] = "data/Alzheimer_s Dataset/all image"
  !python train_fuzzy_model_allimages.py
"""

import os
import re
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BATCH = 32
SEED = 42

# Maps to the notebook's class order/names (must match backend/model.py CLASS_NAMES).
PREFIX_TO_CLASS = {
    "mildDem": "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem": "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]

np.random.seed(SEED)


def fuzzy_targets(counts):
    """Same skfuzzy Mamdani controller as the notebook."""
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
    """Group filenames in DATA_DIR by class, using the filename prefix."""
    buckets = {c: [] for c in CLASS_ORDER}
    for fn in sorted(os.listdir(DATA_DIR)):
        m = re.match(r"^[a-zA-Z]+", fn)
        if not m:
            continue
        cls = PREFIX_TO_CLASS.get(m.group())
        if cls:
            buckets[cls].append(fn)
    return buckets


def main():
    import cv2
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    buckets = _bucket_files()
    counts = {c: len(files) for c, files in buckets.items()}
    print("Real counts:", counts)

    targets = fuzzy_targets(counts)
    print("Fuzzy target counts:", targets)

    class_map = {label: i for i, label in enumerate(CLASS_ORDER)}
    data, labels = [], []
    for label in CLASS_ORDER:
        files = buckets[label]
        target = targets[label]
        # Cycle through the available files with repetition to hit the target
        # (oversampling for minority classes; truncation for majority classes).
        if len(files) >= target:
            selected = files[:target]
        else:
            reps = (target + len(files) - 1) // len(files)
            selected = (files * reps)[:target]
        for fn in selected:
            img = cv2.imread(os.path.join(DATA_DIR, fn))
            if img is None:
                continue
            data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(class_map[label])

    data = np.array(data, dtype=np.float32)  # RAW 0-255 (matches the notebook)
    labels = to_categorical(np.array(labels), num_classes=len(CLASS_ORDER))
    print("Loaded:", data.shape, labels.shape)

    Xtr, Xte, ytr, yte = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=SEED)

    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(len(CLASS_ORDER), activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(Xtr, ytr, validation_data=(Xte, yte),
              epochs=EPOCHS, batch_size=BATCH)

    yp = np.argmax(model.predict(Xte), axis=1)
    yt = np.argmax(yte, axis=1)
    print("\nClassification report:")
    print(classification_report(yt, yp, target_names=CLASS_ORDER))

    model.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras  ->  send this file back / host it for MODEL_URL.")


if __name__ == "__main__":
    main()

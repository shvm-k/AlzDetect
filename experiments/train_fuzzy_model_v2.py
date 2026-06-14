"""
Improved training script for alz_mobilenetv2.keras (v2).

Builds on train_fuzzy_model_allimages.py with the highest-impact upgrades:
  1. Real data augmentation (rotation/zoom/brightness/flip) via
     ImageDataGenerator, instead of relying on raw file-repetition oversampling.
  2. class_weight in model.fit() to reinforce the fuzzy rebalancing.
  3. Two-phase training: frozen base (5 epochs) then fine-tune the last 40
     layers of MobileNetV2 at a low LR (up to 20 more epochs).
  4. EarlyStopping + ReduceLROnPlateau on val_loss.

Usage (same as train_fuzzy_model_allimages.py):
  import os
  os.environ["DATA_DIR"] = "data/Alzheimer_s Dataset/all image"
  !python train_fuzzy_model_v2.py
"""

import os
import re
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = 128
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


def main():
    import cv2
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.utils.class_weight import compute_class_weight

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
    y_int = np.array(labels)
    labels_cat = to_categorical(y_int, num_classes=len(CLASS_ORDER))
    print("Loaded:", data.shape, labels_cat.shape)

    Xtr, Xte, ytr, yte = train_test_split(
        data, labels_cat, test_size=0.2, stratify=labels_cat, random_state=SEED)

    # class_weight reinforces the fuzzy rebalancing during training.
    ytr_int = np.argmax(ytr, axis=1)
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_ORDER)), y=ytr_int)
    class_weight = {i: w for i, w in enumerate(cw)}
    print("Class weights:", class_weight)

    train_gen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=(0.9, 1.1),
        horizontal_flip=True,
    )
    train_flow = train_gen.flow(Xtr, ytr, batch_size=BATCH, seed=SEED)

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

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\n=== Phase 1: frozen base ===")
    model.fit(train_flow, validation_data=(Xte, yte),
              epochs=5, class_weight=class_weight, callbacks=callbacks)

    print("\n=== Phase 2: fine-tune last 40 layers ===")
    for layer in base.layers[-40:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_flow, validation_data=(Xte, yte),
              epochs=20, class_weight=class_weight, callbacks=callbacks)

    yp = np.argmax(model.predict(Xte), axis=1)
    yt = np.argmax(yte, axis=1)
    print("\nClassification report:")
    print(classification_report(yt, yp, target_names=CLASS_ORDER))

    model.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras  ->  send this file back / host it for MODEL_URL.")


if __name__ == "__main__":
    main()

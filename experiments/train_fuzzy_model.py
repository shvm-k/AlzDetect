"""
Faithful reproduction of the FUZZY notebook
(Alzheimer's_Fuzzy-Logic+MobileNetV2) that produces the deployable model file
`alz_mobilenetv2.keras` (~92% accuracy on the balanced subset).

This mirrors the notebook EXACTLY so the saved model matches the live app:
  * class-capped load (500/200/100/1200) adjusted by the fuzzy controller
  * images resized to 128x128, kept as RAW 0-255 (NO /255 normalization) --
    the notebook defines rescale=1/255 but never applies it in model.fit()
  * MobileNetV2 (imagenet), base fully frozen, GAP -> Dense(128) -> Dropout(0.3)
    -> Dense(4, softmax)
  * Adam(lr=1e-4), categorical cross-entropy, 15 epochs, batch 32, 80/20 split

The class order is the notebook's dict order and MUST match backend/model.py:
  Mild Dementia, Moderate Dementia, Very mild Dementia, Non Demented

------------------------------------------------------------------------------
RUN IT (Google Colab, free GPU, ~10-15 min) -- the data lives on Kaggle:

  Runtime -> Change runtime type -> GPU, then:

    !pip install -q scikit-fuzzy
    !pip install -q kaggle
    # upload your kaggle.json (Kaggle -> Settings -> Create New API Token):
    from google.colab import files; files.upload()
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d sachinkumar413/alzheimer-mri-dataset -p data --unzip

    import os
    # Point at the folder that DIRECTLY contains the 4 class subfolders:
    os.environ["DATA_DIR"] = "data"     # adjust if unzip nests it (check with !ls data)

    !python train_fuzzy_model.py

  Then download alz_mobilenetv2.keras (left file panel) and send it to me,
  OR upload it to a Hugging Face model repo and give me the direct link.
------------------------------------------------------------------------------
"""

import os
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data")
IMG_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BATCH = 32
SEED = 42

# Notebook's dict order — keep identical to backend/model.py CLASS_NAMES.
DATASET_CLASSES = {
    "Mild Dementia": 500,
    "Moderate Dementia": 200,
    "Very mild Dementia": 100,
    "Non Demented": 1200,
}

np.random.seed(SEED)


def _find_class_dir(root, name):
    import re
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    target = norm(name)
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and norm(d) == target:
            return os.path.join(root, d)
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and target[:6] in norm(d):
            return os.path.join(root, d)
    raise FileNotFoundError(
        f"No folder for class '{name}' under {root}. Found: {os.listdir(root)}")


def fuzzy_targets(counts):
    """Reproduce the notebook's fuzzy resampling controller exactly."""
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


def main():
    import cv2
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    targets = fuzzy_targets(dict(DATASET_CLASSES))
    print("Fuzzy target counts:", targets)

    class_map = {label: i for i, label in enumerate(DATASET_CLASSES.keys())}
    data, labels = [], []
    for label in DATASET_CLASSES.keys():
        cdir = _find_class_dir(DATA_DIR, label)
        files = sorted(os.listdir(cdir))[: targets[label]]
        for fn in files:
            img = cv2.imread(os.path.join(cdir, fn))
            if img is None:
                continue
            data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(class_map[label])
    data = np.array(data, dtype=np.float32)  # RAW 0-255 (matches the notebook)
    labels = to_categorical(np.array(labels), num_classes=len(DATASET_CLASSES))
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
    out = Dense(len(DATASET_CLASSES), activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(Xtr, ytr, validation_data=(Xte, yte),
              epochs=EPOCHS, batch_size=BATCH)

    yp = np.argmax(model.predict(Xte), axis=1)
    yt = np.argmax(yte, axis=1)
    print("\nClassification report:")
    print(classification_report(yt, yp, target_names=list(DATASET_CLASSES.keys())))

    model.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras  ->  send this file back / host it for MODEL_URL.")


if __name__ == "__main__":
    main()

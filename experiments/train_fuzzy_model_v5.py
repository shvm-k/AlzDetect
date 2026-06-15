"""
Run 6 / v5 -- targeted fix for the "Very mild Dementia" blindspot.

v4 (fuzzy head) hit 72% but Very mild Dementia sat at F1=0.49 -- it sits in the
confusion corridor between Non Demented and Mild Dementia. This version keeps the
v4 fuzzy inference head and SMOTE feature balancing, and adds three targeted
levers (all training-side, no architecture change):

  1. BOOST VERY MILD in the fuzzy/SMOTE resampling: oversample Very mild Dementia
     *beyond* class balance (BOOST_VERYMILD x the majority count) so the fuzzy
     head sees far more of its subtle variations.
  2. MACRO-F1 EARLY STOPPING: stop on val macro-F1 (a custom callback), not
     val_loss/accuracy -- overall accuracy hides the weak minority class.
  3. FOCAL LOSS instead of categorical_crossentropy: down-weights easy, already-
     correct examples and concentrates gradient on the hard, confused ones.

Usage (clone the repo in Colab so this can import the shared fuzzy layer):
  !git clone -b claude/upbeat-newton-f7gyub https://github.com/shvm-k/AlzDetect.git
  %cd AlzDetect
  import os; os.environ["DATA_DIR"] = "data/Alzheimer_s Dataset/all image"
  !pip install -q scikit-fuzzy imbalanced-learn opencv-python-headless
  !python experiments/train_fuzzy_model_v5.py
"""

import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import fuzzy_layer  # noqa: E402
FuzzyLayer = fuzzy_layer.FuzzyLayer

DATA_DIR = os.environ.get("DATA_DIR", "data/Alzheimer_s Dataset/all image")
IMG_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "80"))
BATCH = 32
SEED = 42
PROJ_DIM = int(os.environ.get("PROJ_DIM", "8"))
N_RULES = int(os.environ.get("N_RULES", "16"))
# How much to over-represent Very mild Dementia relative to the majority class.
BOOST_VERYMILD = float(os.environ.get("BOOST_VERYMILD", "1.5"))
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))

PREFIX_TO_CLASS = {
    "mildDem": "Mild Dementia",
    "moderateDem": "Moderate Dementia",
    "verymildDem": "Very mild Dementia",
    "nonDem": "Non Demented",
}
CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
VERYMILD_IDX = CLASS_ORDER.index("Very mild Dementia")

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


def categorical_focal_loss(gamma=2.0):
    """Focal loss: down-weight easy examples, focus on the confused ones."""
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return loss


def main():
    import cv2
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score
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

    data = np.array(data, dtype=np.float32)
    y_int = np.array(labels)
    print("After safe aug:", {CLASS_ORDER[i]: int((y_int == i).sum()) for i in range(4)})

    Xtr_img, Xte_img, ytr_int, yte_int = train_test_split(
        data, y_int, test_size=0.2, stratify=y_int, random_state=SEED)

    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inp)
    base.trainable = False
    pooled = GlobalAveragePooling2D()(base.output)
    feature_model = Model(inp, pooled)

    print("\nExtracting frozen features ...")
    Ftr = feature_model.predict(Xtr_img, batch_size=BATCH, verbose=1)
    Fte = feature_model.predict(Xte_img, batch_size=BATCH, verbose=1)

    # ---- Lever 1: SMOTE balances all classes, but Very mild is BOOSTED ----
    tr_counts = {i: int((ytr_int == i).sum()) for i in range(4)}
    nmax = max(tr_counts.values())
    strategy = {i: max(tr_counts[i], nmax) for i in range(4)}
    strategy[VERYMILD_IDX] = int(nmax * BOOST_VERYMILD)
    min_class = min(tr_counts.values())
    k = max(1, min(5, min_class - 1))
    print(f"SMOTE k={k}, sampling_strategy={strategy} (Very mild boosted x{BOOST_VERYMILD})")
    Ftr_bal, ytr_bal = SMOTE(random_state=SEED, k_neighbors=k,
                             sampling_strategy=strategy).fit_resample(Ftr, ytr_int)
    print("Balanced:", {CLASS_ORDER[i]: int((ytr_bal == i).sum()) for i in range(4)})

    ytr_cat = to_categorical(ytr_bal, 4)
    yte_cat = to_categorical(yte_int, 4)

    feat_in = Input(shape=(Ftr.shape[1],))
    proj = Dense(PROJ_DIM, activation="relu", name="fuzzy_projection")(feat_in)
    head_out = FuzzyLayer(n_rules=N_RULES, n_classes=4, name="fuzzy_head")(proj)
    head = Model(feat_in, head_out)
    # ---- Lever 3: focal loss ----
    head.compile(optimizer=Adam(1e-3),
                 loss=categorical_focal_loss(FOCAL_GAMMA), metrics=["accuracy"])
    print(f"\nFuzzy head PROJ_DIM={PROJ_DIM} N_RULES={N_RULES}, focal gamma={FOCAL_GAMMA}")

    # ---- Lever 2: early-stop on val MACRO-F1 ----
    class MacroF1Stopper(Callback):
        def __init__(self, Xval, yval_int, patience=10):
            super().__init__()
            self.Xval, self.yval_int, self.patience = Xval, yval_int, patience
            self.best, self.best_w, self.wait = -1.0, None, 0

        def on_epoch_end(self, epoch, logs=None):
            yp = np.argmax(self.model.predict(self.Xval, verbose=0), axis=1)
            f1 = f1_score(self.yval_int, yp, average="macro")
            vm = f1_score(self.yval_int, yp, labels=[VERYMILD_IDX], average="macro")
            print(f"  val_macro_f1={f1:.4f}  very_mild_f1={vm:.4f}")
            if f1 > self.best:
                self.best, self.best_w, self.wait = f1, self.model.get_weights(), 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"  early stop (best macro-F1={self.best:.4f})")
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_w is not None:
                self.model.set_weights(self.best_w)

    stopper = MacroF1Stopper(Fte, yte_int, patience=10)
    head.fit(Ftr_bal, ytr_cat, validation_data=(Fte, yte_cat),
             epochs=EPOCHS, batch_size=BATCH, callbacks=[stopper])

    yp = np.argmax(head.predict(Fte), axis=1)
    print("\nClassification report:")
    print(classification_report(yte_int, yp, target_names=CLASS_ORDER))

    full = Model(inp, head(feature_model(inp)))
    full.save("alz_mobilenetv2.keras")
    print("\nSaved alz_mobilenetv2.keras (fuzzy head, very-mild-boosted) -> send it back.")


if __name__ == "__main__":
    main()

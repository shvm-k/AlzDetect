# Trained model goes here

The app loads the first `.keras` or `.h5` file it finds in this folder.
Until you add one, the app runs in **DEMO MODE** (pipeline works, predictions
are placeholder values).

## How to export your model from the notebook

At the end of `Alzheimer's_Fuzzy-Logic+MobileNetV2 (1).ipynb`, after training,
add one cell:

```python
model.save("alz_mobilenetv2.keras")
```

Then download `alz_mobilenetv2.keras` and drop it into this folder:

```
backend/models/alz_mobilenetv2.keras
```

Restart the server and the badge will switch from "DEMO MODE" to live.
The class order the app expects (must match training) is:

1. Mild Dementia
2. Moderate Dementia
3. Very mild Dementia
4. Non Demented

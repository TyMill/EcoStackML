# 💾 Save & Load Utilities

Utility functions for saving and loading models, metrics, and predictions.

---

## 📁 Functions

### `save_model(model, path)`

Saves a trained model to a `.pkl` file using `joblib`.

### `load_model(path) -> model`

Loads a model from a `.pkl` file.

### `save_stacker(stacker, path)`

Saves the entire `ModelStacker` object using `pickle`.

### `load_stacker(path) -> ModelStacker`

Loads a saved `ModelStacker` object.

### `save_predictions(y_true, y_pred, path)`

Exports a CSV with true vs. predicted labels.

### `save_metrics(metrics: Dict, path)`

Exports a `.json` file with evaluation metrics.

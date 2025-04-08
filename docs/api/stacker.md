# 🧠 ModelStacker

`ModelStacker` is the core class for training and predicting using stacked machine learning models. It supports flexible base model configuration and a meta-model for final predictions.

---

## 🔧 Constructor

```python
ModelStacker(
    base_models_config: List[Dict[str, Any]],
    meta_model_name: str = "logistic",
    model_type: str = "classification"
)
```

- `base_models_config`: list of dictionaries with model names and parameters
- `meta_model_name`: name of the meta-model ("logistic", "xgboost", etc.)
- `model_type`: "classification" or "regression"

---

## ⚙️ Methods

### `fit(X, y)`

Trains all base models and the meta-model using cross-validation.

### `predict(X)`

Returns final prediction from the stacked model.

### `meta_model_input(X)`

Returns base model predictions for use in meta-model.

### `explain_base_models(X)`

Generates SHAP visualizations for each base model.

### `explain_meta_model(X)`

Generates SHAP visualization for the meta-model.

# 🌿 EcoStackML


[![PyPI](https://img.shields.io/pypi/v/ecostackml?color=green)](https://pypi.org/project/ecostackml/)
[![Downloads](https://static.pepy.tech/badge/ecostackml)](https://pepy.tech/project/ecostackml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15173335.svg)](https://doi.org/10.5281/zenodo.15173335)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/TyMill/EcoStackML)](https://github.com/TyMill/EcoStackML/LICENCE)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://tymill.github.io/EcoStackML/)
[![Notebook](https://img.shields.io/badge/Tutorial-Notebook-yellow)](https://github.com/TyMill/EcoStackML/tree/main/notebooks)


**Stacked Machine Learning Framework for Environmental and Tabular Data**

EcoStackML is a modular and production-ready Python framework that leverages stacked machine learning techniques to deliver robust and explainable models for classification and regression tasks. Designed for environmental researchers, data scientists, and ML engineers.

---

## 🚀 Features

- ✅ Supports multiple base models (Random Forest, XGBoost, SVM, etc.)
- 🧠 Meta-learner for model stacking (e.g., Logistic Regression, Gradient Boosting)
- 📊 Built-in evaluation: ROC-AUC, PR Curve, SHAP plots, confusion matrix
- 🧽 Preprocessing pipeline with anomaly removal, scaling, imputation
- 📅 Automatic datetime feature extraction
- 💾 Save and load models, predictions, and metrics
- 📓 Includes Jupyter notebooks (01–07) with step-by-step tutorials
- 🔧 YAML-based configuration & logging setup

---

## 🛠 Installation

```bash
pip install .
```

For development:

```bash
pip install .[dev]
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
data:
  path: "data/raw/sample.csv"
  target_column: "target"

preprocessing:
  missing_strategy: "median"
  scaling: "standard"
  anomaly_method: "iqr"
  datetime_cols: []

model:
  base_models:
    - name: "random_forest"
    - name: "xgboost"
  meta_model: "logistic"
  model_type: "classification"

split:
  test_size: 0.2
  stratify: true
  random_state: 42

output:
  model_dir: "models/"
  results_dir: "results/"
```

---

## 🧪 Quickstart

```python
from ecostackml.data.loader import DataLoader
from ecostackml.preprocessing.cleaner import Cleaner
from ecostackml.data.splitter import split_data
from ecostackml.models.stacker import ModelStacker
from ecostackml.models.evaluator import evaluate_classification

df = DataLoader.from_csv("data/raw/sample.csv")
df["target"] = [0, 0, 1, 0, 1]

cleaner = Cleaner(strategy="median", scaling="standard", anomaly_method="iqr")
df_clean = cleaner.fit_transform(df)

X_train, X_test, _, y_train, y_test, _ = split_data(df_clean, target_column="target")

stacker = ModelStacker(
    base_models_config=[{"name": "random_forest"}, {"name": "xgboost"}],
    meta_model_name="logistic",
    model_type="classification"
)

stacker.fit(X_train, y_train)
y_pred = stacker.predict(X_test)
```

---

## 📈 Evaluation & SHAP

```python
from ecostackml.models.evaluator import evaluate_classification
metrics = evaluate_classification(y_test, y_pred, plot=True)

stacker.explain_base_models(X_test)
stacker.explain_meta_model(X_test)
```

---

## 💾 Save / Load

```python
from ecostackml.utils.save_load import save_model, load_model, save_stacker, load_stacker

save_model(stacker.meta_model.model, "models/meta_model.pkl")
save_stacker(stacker, "models/full_stacker.pkl")

restored = load_stacker("models/full_stacker.pkl")
restored.predict(X_test)
```

---

## 📁 Project Structure

```
EcoStackML/
├── src/ecostackml/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   └── utils/
├── notebooks/
├── main.py
├── config.yaml
├── pyproject.toml
└── README.md
```

---

## 📚 Notebooks

- `01_data_loading.ipynb` – loading CSV, JSON, Parquet, Hive
- `02_cleaning_and_preprocessing.ipynb` – full preprocessing
- `03_model_training.ipynb` – base + stacking models
- `04_model_evaluation.ipynb` – metrics & visualization
- `05_shap_explainer.ipynb` – explainability
- `06_full_pipeline.ipynb` – complete pipeline
- `07_save_and_load.ipynb` – serialization demo

---

## 🤝 Contributing

Feel free to fork, contribute, and suggest improvements!

---

## 📜 License

MIT © 2025 Tymoteusz Miller

# 🧪 CLI Example – `main.py`

This script executes the full EcoStackML pipeline using a YAML configuration file.

---

## ▶️ Usage

```bash
python main.py
```

Make sure `config.yaml` is correctly set.

---

## ⚙️ What it does

- Loads and cleans data
- Splits into train/test
- Trains base models and meta-model (stacking)
- Evaluates performance
- Generates SHAP plots
- Saves models, metrics, and predictions

---

## 🛠 Configuration

See: [`config.yaml`](../config.yaml)

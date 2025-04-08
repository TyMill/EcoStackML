# 🤖 Tutorial: Model Training

This tutorial shows how to train individual base models and a meta-model for stacking.

## Step-by-Step

1. Generate or load your dataset
2. Split into train/test
3. Define base models
4. Fit the stacker

## Example

```python
from sklearn.datasets import make_classification
import pandas as pd
from ecostackml.data.splitter import split_data
from ecostackml.models.stacker import ModelStacker

# Generate toy data
X, y = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df["target"] = y

# Split
X_train, X_test, _, y_train, y_test, _ = split_data(df, target_column="target", stratify=True)

# Train
stacker = ModelStacker(
    base_models_config=[{"name": "random_forest"}, {"name": "xgboost"}],
    meta_model_name="logistic",
    model_type="classification"
)
stacker.fit(X_train, y_train)
```

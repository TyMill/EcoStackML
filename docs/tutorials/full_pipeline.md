# 🔄 Tutorial: Full ML Pipeline

This tutorial walks through the full ML pipeline using EcoStackML.

## Steps Covered

1. Load and clean data
2. Split dataset
3. Train stacked model
4. Evaluate results
5. SHAP explainability
6. Save models and predictions

## Example

```python
from ecostackml.data.loader import DataLoader
from ecostackml.preprocessing.cleaner import Cleaner
from ecostackml.models.stacker import ModelStacker
from ecostackml.models.evaluator import evaluate_classification

df = DataLoader.from_csv("sample.csv")
cleaner = Cleaner(strategy="median")
df_clean = cleaner.fit_transform(df)
# ... continue as shown in notebook 06 ...
```

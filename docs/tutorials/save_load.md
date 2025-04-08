# 💾 Tutorial: Saving and Loading

Save your models and results for future reuse.

## Save models

```python
from ecostackml.utils.save_load import save_model

save_model(stacker.meta_model.model, "models/meta.pkl")
```

## Save metrics and predictions

```python
from ecostackml.utils.save_load import save_metrics, save_predictions

save_predictions(y_test, y_pred, "results/preds.csv")
save_metrics(metrics, "results/metrics.json")
```

## Save / Load full stacker

```python
from ecostackml.utils.save_load import save_stacker, load_stacker

save_stacker(stacker, "models/full_stacker.pkl")
restored = load_stacker("models/full_stacker.pkl")
```

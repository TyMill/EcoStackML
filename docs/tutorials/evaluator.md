# 📈 Tutorial: Model Evaluation

Learn how to evaluate your classification model with built-in tools.

## Example

```python
from ecostackml.models.evaluator import evaluate_classification

metrics = evaluate_classification(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    plot=True
)

print(metrics)
```

This function generates:
- Confusion matrix
- ROC Curve + AUC
- Precision-Recall Curve
- Returns dictionary with accuracy, precision, recall, F1, etc.

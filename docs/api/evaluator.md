# 📊 evaluate_classification

This function evaluates classification models using common metrics and visualizations.

---

## 🧠 Signature

```python
evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    plot: bool = True
) -> Dict[str, float]
```

- `y_true`: true labels
- `y_pred`: predicted labels
- `y_proba`: predicted probabilities (for ROC/PR curves)
- `plot`: whether to display plots

---

## 📈 Returns

Dictionary of metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
- pr_auc

And displays:
- ROC curve
- Precision-Recall curve
- Confusion matrix

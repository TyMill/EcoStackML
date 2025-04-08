# 🔍 Tutorial: SHAP Explainability

This tutorial demonstrates how to use SHAP to explain model predictions.

## Base Model SHAP

```python
stacker.explain_base_models(X_test)
```

Generates SHAP summary plots for each base model individually.

## Meta Model SHAP

```python
stacker.explain_meta_model(X_test)
```

Shows the influence of each base model on the final stacked prediction.

## Notes

- Ensure models support `predict_proba` and SHAP is installed.
- Visualizations open in notebook or interactive window.

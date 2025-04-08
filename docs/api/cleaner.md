# 🧽 Cleaner

The `Cleaner` class provides a unified interface for cleaning tabular data, handling missing values, anomalies, scaling, and datetime features.

---

## 🔧 Constructor

```python
Cleaner(
    strategy: str = "mean",
    scaling: str = "standard",
    anomaly_method: Optional[str] = None,
    datetime_cols: Optional[List[str]] = None
)
```

- `strategy`: how to fill missing values ("mean", "median", "most_frequent")
- `scaling`: "standard" or "minmax"
- `anomaly_method`: "iqr" or "isolation" or None
- `datetime_cols`: list of datetime columns to process

---

## ⚙️ Methods

### `fit(df)`

Fits transformers to the dataset.

### `transform(df)`

Transforms a dataset using the previously fitted strategies.

### `fit_transform(df)`

Combined version of `fit` and `transform`.

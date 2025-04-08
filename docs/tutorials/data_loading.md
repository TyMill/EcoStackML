# 📘 Tutorial: Data Loading

This tutorial shows how to load data in various formats using `DataLoader`.

## Supported Formats

- CSV (`from_csv`)
- JSON (`from_json`)
- Parquet (`from_parquet`)
- Hive/Beeline (experimental)

## Example

```python
from ecostackml.data.loader import DataLoader

df_csv = DataLoader.from_csv("sample.csv")
df_json = DataLoader.from_json("sample.json")
df_parquet = DataLoader.from_parquet("sample.parquet")
```


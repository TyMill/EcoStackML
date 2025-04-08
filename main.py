import yaml
import pandas as pd
from ecostackml.utils.logging_setup import setup_logger
from ecostackml.data.loader import DataLoader
from ecostackml.preprocessing.cleaner import Cleaner
from ecostackml.data.splitter import split_data
from ecostackml.models.stacker import ModelStacker
from ecostackml.models.evaluator import evaluate_classification
from ecostackml.models.shap_explainer import SHAPExplainer
from ecostackml.utils.save_load import save_model, save_metrics, save_predictions

logger = setup_logger(__name__)

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load data
    df = DataLoader.from_csv(config["data"]["path"])
    if config["data"]["target_column"] not in df.columns:
        raise ValueError("Target column not found in dataset.")
    
    # 2. Clean data
    cleaner = Cleaner(
        strategy=config["preprocessing"]["missing_strategy"],
        scaling=config["preprocessing"]["scaling"],
        anomaly_method=config["preprocessing"]["anomaly_method"],
        datetime_cols=config["preprocessing"]["datetime_cols"]
    )
    df_clean = cleaner.fit_transform(df)

    # 3. Split data
    X_train, X_test, _, y_train, y_test, _ = split_data(
        df_clean,
        target_column=config["data"]["target_column"],
        test_size=config["split"]["test_size"],
        stratify=config["split"]["stratify"],
        random_state=config["split"]["random_state"]
    )

    # 4. Train model
    stacker = ModelStacker(
        base_models_config=config["model"]["base_models"],
        meta_model_name=config["model"]["meta_model"],
        model_type=config["model"]["model_type"]
    )
    stacker.fit(X_train, y_train)

    # 5. Predict and evaluate
    y_pred = stacker.predict(X_test)
    meta_input = pd.DataFrame([
        model.model.predict(X_test) for model in stacker.base_models
    ]).T
    y_proba = stacker.meta_model.model.predict_proba(meta_input)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba)
    logger.info(f"Evaluation metrics: {metrics}")

    # 6. SHAP
    stacker.explain_base_models(X_test)
    stacker.explain_meta_model(X_test)

    # 7. Save everything
    model_dir = config["output"]["model_dir"]
    results_dir = config["output"]["results_dir"]
    save_model(stacker.meta_model.model, f"{model_dir}/stacked_meta_model.pkl")
    for i, base in enumerate(stacker.base_models):
        save_model(base.model, f"{model_dir}/base_model_{i}.pkl")

    save_predictions(y_test, y_pred, f"{results_dir}/predictions.csv")
    save_metrics(metrics, f"{results_dir}/metrics.json")


if __name__ == "__main__":
    main()

import os
import joblib
import json
import pickle
import pandas as pd
from typing import Any, Dict
from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)


def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str) -> Any:
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def save_metrics(metrics: Dict[str, float], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {path}")


def save_predictions(y_true, y_pred, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.to_csv(path, index=False)
    logger.info(f"Predictions saved to {path}")


def save_stacker(stacker_obj: Any, path: str):
    """
    Saves the entire ModelStacker object (meta + base models + config) to a single file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stacker_obj, f)
    logger.info(f"Full ModelStacker saved to {path}")


def load_stacker(path: str) -> Any:
    """
    Loads the full ModelStacker object from file.
    """
    with open(path, 'rb') as f:
        stacker = pickle.load(f)
    logger.info(f"Full ModelStacker loaded from {path}")
    return stacker

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression

from ecostackml.models.base_models import BaseModelTrainer
from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)


class ModelStacker:
    def __init__(self,
                 base_models_config: List[Dict],
                 meta_model_name: str = 'logistic',
                 model_type: Literal['classification', 'regression'] = 'classification',
                 k_folds: int = 5,
                 random_state: int = 42):
        self.base_models_config = base_models_config
        self.meta_model_name = meta_model_name
        self.model_type = model_type
        self.k_folds = k_folds
        self.random_state = random_state

        self.base_models: List[BaseModelTrainer] = []
        self.meta_model: Optional[BaseModelTrainer] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting stacked model...")
        self.base_models = [BaseModelTrainer(model_type=self.model_type,
                                             model_name=cfg["name"],
                                             params=cfg.get("params"))
                            for cfg in self.base_models_config]

        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)

        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i + 1}/{len(self.base_models)}: {model.model_name}")
            preds = np.zeros(X.shape[0])

            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]

                model_fold = BaseModelTrainer(model_type=self.model_type,
                                              model_name=model.model_name,
                                              params=model.params)
                model_fold.fit(X_train_fold, y_train_fold)
                preds[val_idx] = model_fold.predict(X_val_fold)

            meta_features[:, i] = preds

        logger.info("Training meta-model...")
        self.meta_model = BaseModelTrainer(model_type=self.model_type,
                                           model_name=self.meta_model_name)
        self.meta_model.fit(pd.DataFrame(meta_features), y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.info("Predicting using stacked model...")
        meta_features = np.column_stack([
            model.model.predict(X) for model in self.base_models
        ])
        return self.meta_model.predict(pd.DataFrame(meta_features))

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating stacked model...")
        y_pred = self.predict(X)
        return self.meta_model.evaluate(pd.DataFrame(y_pred).T, y_true)

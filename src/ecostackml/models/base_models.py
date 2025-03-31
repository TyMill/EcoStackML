from typing import Dict, Optional, Literal, Any
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None


class BaseModelTrainer:
    def __init__(self,
                 model_type: Literal['classification', 'regression'] = 'classification',
                 model_name: str = 'random_forest',
                 params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.model_name = model_name
        self.params = params or {}
        self.model = self._init_model()

    def _init_model(self):
        logger.info(f"Initializing model: {self.model_name} ({self.model_type})")
        if self.model_type == 'classification':
            return self._init_classification_model()
        else:
            return self._init_regression_model()

    def _init_classification_model(self):
        models = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier if XGBClassifier else None,
            'lightgbm': LGBMClassifier if LGBMClassifier else None,
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'logistic': LogisticRegression
        }
        cls = models.get(self.model_name)
        if cls is None:
            raise ValueError(f"Unknown classification model: {self.model_name}")
        return cls(**self.params)

    def _init_regression_model(self):
        models = {
            'random_forest': RandomForestRegressor,
            'xgboost': XGBRegressor if XGBRegressor else None,
            'lightgbm': LGBMRegressor if LGBMRegressor else None,
            'svm': SVR,
            'linear': LinearRegression
        }
        cls = models.get(self.model_name)
        if cls is None:
            raise ValueError(f"Unknown regression model: {self.model_name}")
        return cls(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Training base model...")
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        if self.model_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
        else:
            return {
                'rmse': mean_squared_error(y_true, y_pred, squared=False),
                'r2': r2_score(y_true, y_pred)
            }

    def get_model(self):
        return self.model

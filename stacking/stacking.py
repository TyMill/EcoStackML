import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def create_base_models():
    """Create a dictionary of base models."""
    base_models = {
        'decision_tree': DecisionTreeRegressor(),
        'mlp': MLPRegressor(),
        'xgb': XGBRegressor()
    }
    return base_models

def create_stacking_model(base_models):
    """Create a stacking model using the base models."""
    estimators = [(name, model) for name, model in base_models.items()]
    stacking_model = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression())
    return stacking_model

def train_stacking_model(X, y):
    """Train the stacking model."""
    base_models = create_base_models()
    stacking_model = create_stacking_model(base_models)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    stacking_model.fit(X_train, y_train)
    
    val_predictions = stacking_model.predict(X_val)
    return stacking_model, val_predictions, y_val

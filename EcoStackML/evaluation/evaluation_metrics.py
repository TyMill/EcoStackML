from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """Evaluate the model using different metrics."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def print_evaluation_metrics(y_true, y_pred):
    """Print the evaluation metrics."""
    metrics = evaluate_model(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

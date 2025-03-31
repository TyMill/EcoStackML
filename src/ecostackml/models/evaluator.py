import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Literal
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    mean_squared_error, r2_score
)

from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)


def evaluate_classification(y_true: pd.Series,
                            y_pred: np.ndarray,
                            y_proba: np.ndarray,
                            plot: bool = True) -> Dict[str, float]:
    logger.info("Evaluating classification performance...")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    results = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

    # If binary classification and proba available
    if y_proba is not None and len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_vals, precision_vals)

        results["roc_auc"] = roc_auc
        results["pr_auc"] = pr_auc

        if plot:
            # ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Precision-Recall Curve
            plt.figure()
            plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.2f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Confusion matrix
    if plot:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    return results


def evaluate_regression(y_true: pd.Series,
                        y_pred: np.ndarray,
                        plot: bool = True) -> Dict[str, float]:
    logger.info("Evaluating regression performance...")
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    if plot:
        # Scatter plot
        plt.figure()
        sns.scatterplot(x=y_true, y=y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs True")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Residuals
        residuals = y_true - y_pred
        plt.figure()
        sns.histplot(residuals, bins=30, kde=True)
        plt.title("Residuals")
        plt.xlabel("Error")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "rmse": rmse,
        "r2": r2
    }

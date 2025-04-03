import pandas as pd
from typing import Tuple, Optional, Literal
from sklearn.model_selection import train_test_split
from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)


def split_data(df: pd.DataFrame,
               target_column: str,
               test_size: float = 0.2,
               val_size: Optional[float] = None,
               stratify: bool = False,
               shuffle: bool = True,
               random_state: int = 42,
               time_series: bool = False
               ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame],
                          pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Splits data into train/test/(val) sets.

    Returns: X_train, X_test, (X_val), y_train, y_test, (y_val)
    """
    logger.info("Splitting data...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if time_series:
        logger.info("Time series split enabled.")
        split_index = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        X_val = y_val = None
    else:
        stratify_col = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_col,
            shuffle=shuffle, random_state=random_state
        )
        X_val = y_val = None

        if val_size:
            logger.info("Performing additional train/val split...")
            stratify_col = y_train if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size,
                stratify=stratify_col, shuffle=shuffle, random_state=random_state
            )

    logger.info(f"Data split complete. Sizes — Train: {len(X_train)}, Test: {len(X_test)}" +
                (f", Val: {len(X_val)}" if X_val is not None else ""))

    return X_train, X_test, X_val, y_train, y_test, y_val

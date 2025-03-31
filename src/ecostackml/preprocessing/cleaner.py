import pandas as pd
from typing import Optional, List, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from ecostackml.utils.logging_setup import setup_logger

logger = setup_logger(__name__)


class Cleaner:
    def __init__(self,
                 strategy: Literal['mean', 'median', 'mode', 'ffill', 'bfill'] = 'mean',
                 scaling: Optional[Literal['standard', 'minmax', 'robust']] = 'standard',
                 anomaly_method: Optional[Literal['isolation', 'iqr']] = None,
                 datetime_cols: Optional[List[str]] = None):
        self.strategy = strategy
        self.scaling = scaling
        self.scaler = None
        self.anomaly_method = anomaly_method
        self.datetime_cols = datetime_cols

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Handling missing values using strategy: {self.strategy}")
        if self.strategy == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif self.strategy == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif self.strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif self.strategy == 'ffill':
            return df.fillna(method='ffill')
        elif self.strategy == 'bfill':
            return df.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown missing data strategy: {self.strategy}")

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scaling:
            logger.info("Scaling skipped.")
            return df

        logger.info(f"Applying scaling method: {self.scaling}")
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical variables using LabelEncoder.")
        obj_cols = df.select_dtypes(include='object').columns.tolist()
        for col in obj_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def _handle_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_cols:
            return df
        logger.info(f"Parsing datetime columns: {self.datetime_cols}")
        for col in self.datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_weekday"] = df[col].dt.weekday
            df.drop(columns=col, inplace=True)
        return df

    def _remove_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.anomaly_method:
            return df

        logger.info(f"Removing anomalies using method: {self.anomaly_method}")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if self.anomaly_method == 'isolation':
            clf = IsolationForest(contamination=0.01, random_state=42)
            y_pred = clf.fit_predict(df[numeric_cols])
            df = df[y_pred == 1]
        elif self.anomaly_method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        else:
            raise ValueError(f"Unknown anomaly method: {self.anomaly_method}")
        return df

    def fit(self, df: pd.DataFrame):
        logger.info("Fitting cleaner...")
        df = self._handle_datetime(df)
        df = self._handle_missing(df)
        df = self._remove_anomalies(df)
        df = self._encode_categorical(df)
        self._scale_features(df)
        logger.info("Fitting complete.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming data...")
        df = self._handle_datetime(df)
        df = self._handle_missing(df)
        df = self._remove_anomalies(df)
        df = self._encode_categorical(df)
        df = self._scale_features(df)
        logger.info("Transformation complete.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

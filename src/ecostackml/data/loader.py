import pandas as pd
import json
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def from_csv(path: str, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, **kwargs)
            logger.info(f"CSV file loaded successfully from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise

    @staticmethod
    def from_json(path: str, orient: str = 'records', **kwargs) -> pd.DataFrame:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"JSON file loaded successfully from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise

    @staticmethod
    def from_parquet(path: str, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path, **kwargs)
            logger.info(f"Parquet file loaded successfully from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            raise

    @staticmethod
    def from_hive(query: str, connection: Optional[Any] = None, output_format: str = 'pandas') -> pd.DataFrame:
        """
        Execute Hive query using pyhive or system beeline (as fallback).
        """
        try:
            if connection is not None:
                # Option 1: Using pyhive
                df = pd.read_sql(query, connection)
                logger.info("Hive query executed via pyhive.")
            else:
                # Option 2: Beeline fallback (must have Hadoop env)
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                    tmp_path = tmp.name

                beeline_cmd = f"beeline --outputformat=csv2 -e \"{query}\" > {tmp_path}"
                subprocess.run(beeline_cmd, shell=True, check=True)

                df = pd.read_csv(tmp_path)
                os.remove(tmp_path)
                logger.info("Hive query executed via beeline.")

            return df

        except Exception as e:
            logger.error(f"Failed to execute Hive query: {e}")
            raise

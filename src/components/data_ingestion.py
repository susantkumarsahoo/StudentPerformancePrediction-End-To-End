import os
import sys
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
            logger.info("DataIngestion initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion process...")

            # Load raw data
            df = pd.read_csv(self.config.dataset_path)
            logger.info(f"Raw dataset loaded with shape: {df.shape}")

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved at: {self.config.raw_data_path}")

            # Process (simple clean)
            df_processed = df.dropna()
            df_processed.to_csv(self.config.processed_data_path, index=False)
            logger.info(f"Processed data saved at: {self.config.processed_data_path}, shape: {df_processed.shape}")

            # Split train-test
            train_df, test_df = train_test_split(
                df_processed,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(f"Train data saved at: {self.config.train_data_path}, shape: {train_df.shape}")
            logger.info(f"Test data saved at: {self.config.test_data_path}, shape: {test_df.shape}")

            # Metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "raw_shape": df.shape,
                "processed_shape": df_processed.shape,
                "train_shape": train_df.shape,
                "test_shape": test_df.shape
            }
            with open(self.config.metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved at: {self.config.metadata_path}")

            # Schema
            schema = {"type": "object", "properties": {}}
            for col in df.columns:
                if pd.api.types.is_integer_dtype(df[col]):
                    col_type = "integer"
                elif pd.api.types.is_float_dtype(df[col]):
                    col_type = "number"
                else:
                    col_type = "string"

                unique_vals = df[col].nunique() if col_type == "string" else None
                example_vals = df[col].dropna().unique()[:5].tolist()

                schema["properties"][col] = {
                    "type": col_type,
                    "unique_values": unique_vals,
                    "example_values": example_vals
                }

            with open(self.config.schema_path, "w") as f:
                json.dump(schema, f, indent=4)
            logger.info(f"Schema saved at: {self.config.schema_path}")

            logger.info("Data ingestion completed successfully.")
            return DataIngestionArtifact(
                raw_data_path=self.config.raw_data_path,
                processed_data_path=self.config.processed_data_path,
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
                metadata_path=self.config.metadata_path,
                schema_path=self.config.schema_path
            )

        except Exception as e:
            logger.error("Error occurred during data ingestion.", exc_info=True)
            raise CustomException(e, sys)




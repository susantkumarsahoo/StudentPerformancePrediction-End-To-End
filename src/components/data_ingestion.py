import os
import json
import sys
import pandas as pd
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.constants.constants import*

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config

            # create dirs
            os.makedirs(os.path.join(self.config.artifact_dir, RAW_DATA_DIR), exist_ok=True)
            os.makedirs(os.path.join(self.config.artifact_dir, PROCESSED_DATA_DIR), exist_ok=True)
            os.makedirs(os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR), exist_ok=True)
            os.makedirs(os.path.join(self.config.artifact_dir, TIMESTAMP), exist_ok=True)


            logger.info("DataIngestion initialized. Directories created successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion process...")

            # Load raw data
            df = pd.read_csv(self.config.dataset_path)
            logger.info(f"Raw dataset loaded with shape: {df.shape}")

            # Save raw
            raw_data_file = os.path.join(self.config.artifact_dir, RAW_DATA_DIR, RAW_FILE_NAME)
            df.to_csv(raw_data_file, index=False)
            logger.info(f"Raw data saved at: {raw_data_file}")

            # Process data (simple clean)
            df_processed = df.dropna()
            processed_file = os.path.join(self.config.artifact_dir, PROCESSED_DATA_DIR, PROCESSED_FILE_NAME)
            df_processed.to_csv(processed_file, index=False)
            logger.info(f"Processed data saved at: {processed_file} with shape: {df_processed.shape}")

            # Split
            train_df, test_df = train_test_split(
                df_processed,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            train_file = os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR, TRAIN_FILE_NAME)
            test_file = os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR, TEST_FILE_NAME)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            logger.info(f"Train data saved at: {train_file}, shape: {train_df.shape}")
            logger.info(f"Test data saved at: {test_file}, shape: {test_df.shape}")

            # Metadata
            metadata_file = os.path.join(self.config.artifact_dir, METADATA_FILE_NAME)
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "raw_shape": df.shape,
                "processed_shape": df_processed.shape,
                "train_shape": train_df.shape,
                "test_shape": test_df.shape
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved at: {metadata_file}")

            # Schema
            schema_file = os.path.join(self.config.artifact_dir, SCHEMA_FILE_NAME)
            schema = {"type": "object", "properties": {}}

            for col in df.columns:
                # Determine type
                if pd.api.types.is_integer_dtype(df[col]):
                    col_type = "integer"
                elif pd.api.types.is_float_dtype(df[col]):
                    col_type = "number"
                else:
                    col_type = "string"

                # Unique values only for categorical columns
                unique_vals = df[col].nunique() if col_type == "string" else None

                # Example values: first 5 unique non-null values
                example_vals = df[col].dropna().unique()[:5].tolist()

                schema["properties"][col] = {
                    "type": col_type,
                    "unique_values": unique_vals,
                    "example_values": example_vals
                }

            # Save schema to JSON
            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=4)
            logger.info(f"Schema saved at: {schema_file}")

            # Save artifact paths
            logger.info("Data ingestion completed successfully.")
            return DataIngestionArtifact(
                raw_data_path=raw_data_file,
                processed_data_path=processed_file,
                train_data_path=train_file,
                test_data_path=test_file,
                metadata_path=metadata_file,
                schema_path=schema_file
            )

        except Exception as e:
            logger.error("Error occurred during data ingestion.", exc_info=True)
            raise CustomException(e, sys)





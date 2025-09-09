import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.constants.constants import (RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLIT_DATA_DIR,
                       RAW_FILE_NAME, PROCESSED_FILE_NAME, TRAIN_FILE_NAME,
                       TEST_FILE_NAME, METADATA_FILE_NAME, SCHEMA_FILE_NAME)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

        # create dirs
        os.makedirs(os.path.join(self.config.artifact_dir, RAW_DATA_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.config.artifact_dir, PROCESSED_DATA_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR), exist_ok=True)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        # Load raw data
        df = pd.read_csv(self.config.dataset_path)

        # Save raw
        raw_data_file = os.path.join(self.config.artifact_dir, RAW_DATA_DIR, RAW_FILE_NAME)
        df.to_csv(raw_data_file, index=False)

        # Process data (simple clean)
        df_processed = df.dropna()
        processed_file = os.path.join(self.config.artifact_dir, PROCESSED_DATA_DIR, PROCESSED_FILE_NAME)
        df_processed.to_csv(processed_file, index=False)

        # Split
        train_df, test_df = train_test_split(df_processed, test_size=self.config.test_size,
                                             random_state=self.config.random_state)
        train_file = os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR, TRAIN_FILE_NAME)
        test_file = os.path.join(self.config.artifact_dir, SPLIT_DATA_DIR, TEST_FILE_NAME)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

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
        # Schema (example schema)
        schema_file = os.path.join(self.config.artifact_dir, SCHEMA_FILE_NAME) 
        schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"},
                "target": {"type": "number"}
            }
        }

        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=4)

        # Save artifact paths
        return DataIngestionArtifact(
            raw_data_path=raw_data_file,
            processed_data_path=processed_file,
            train_data_path=train_file,
            test_data_path=test_file,
            metadata_path=metadata_file,
            schema_path=schema_file
        )



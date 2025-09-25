import os
import os
from src.constants.constants import*

# Configuration Ingestion entity classes
# Configuration Ingestion entity classes
class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state

        # Main artifact directory for this run
        self.ingestion_dir = os.path.join(ARTIFACTS_DIR, INGESTION_DATA_DIR, TIMESTAMP)
        os.makedirs(self.ingestion_dir, exist_ok=True)

        # Subdirectories
        self.raw_data_dir = os.path.join(self.ingestion_dir, RAW_DATA_DIR)
        self.processed_data_dir = os.path.join(self.ingestion_dir, PROCESSED_DATA_DIR)
        self.split_data_dir = os.path.join(self.ingestion_dir, SPLIT_DATA_DIR)

        # Create subdirectories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.split_data_dir, exist_ok=True)

        # File paths
        self.raw_data_path = os.path.join(self.raw_data_dir, RAW_FILE_NAME)
        self.processed_data_path = os.path.join(self.processed_data_dir, PROCESSED_FILE_NAME)
        self.train_data_path = os.path.join(self.split_data_dir, TRAIN_FILE_NAME)
        self.test_data_path = os.path.join(self.split_data_dir, TEST_FILE_NAME)
        self.metadata_path = os.path.join(self.ingestion_dir, METADATA_FILE_NAME)
        self.schema_path = os.path.join(self.ingestion_dir, SCHEMA_FILE_NAME)


# Configuration Validation entity classes
class DataValidationConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)
        self.max_missing_threshold = MAX_MISSING_THRESHOLD
        self.drift_threshold = DRIFT_THRESHOLD
        self.numerical_column_threshold = NUMERICAL_COLUMN_THRESHOLD
        self.categorical_column_threshold = CATEGORICAL_COLUMN_THRESHOLD


# Configuration Preprocessing entity classes
class DataPreprocessingConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)
        self.preprocessing_data_dir = PREPROCESSING_DATA_DIR
        self.preprocessing_train_file = PREPROCESSING_TRAIN_FILE_NAME
        self.preprocessing_test_file = PREPROCESSING_TEST_FILE_NAME
        self.preprocessing_report_file = PREPROCESSING_REPORT_FILE_NAME

# Configuration Feature Engineering entity classes
class FeatureEngineeringConfig:
    def __init__(self):
        # Main artifact directory
        self.artifact_dir = ARTIFACTS_DIR
        
        # Directory specific to feature engineering
        self.feature_engineering_dir = os.path.join(self.artifact_dir, FEATURE_ENGINEERING_DATA_DIR)
        
        # Make sure the directory exists
        os.makedirs(self.feature_engineering_dir, exist_ok=True)
        
        # File paths
        self.feature_engineering_train_file = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TRAIN_FILE_NAME)
        self.feature_engineering_test_file = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TEST_FILE_NAME)
        self.feature_engineering_report_file = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_REPORT_FILE_NAME)

# Configuration Transformation entity classes
class DataTransformationConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)
        self.transformer_data_dir = TRANSFORMER_DATA_DIR
        self.transformer_train_file = TRANSFORMER_TRAIN_FILE_NAME
        self.transformer_test_file = TRANSFORMER_TEST_FILE_NAME
        self.transformer_object_file = TRANSFORMER_OBJECT_FILE_NAME
        self.transformer_report_file = TRANSFORMER_REPORT_FILE_NAME


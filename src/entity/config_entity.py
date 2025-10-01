import os
import sys
from src.constants.constants import*

# Configuration Ingestion entity classes
class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state

        # Main artifact directory for this run
        self.ingestion_dir = os.path.join(ARTIFACTS_DIR,DATA_INGESTION_DIR)
        
        # Subdirectories - Define paths first
        self.timestamp_dir = os.path.join(self.ingestion_dir, TIMESTAMP)
        self.raw_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_RAW_DIR)
        self.processed_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_PROCESSED_DIR)
        self.split_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_SPLIT_DIR)

        # Create directories after defining paths
        os.makedirs(self.ingestion_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.split_data_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.raw_data_path = os.path.join(self.raw_data_dir, DATA_INGESTION_RAW_FILE)
        self.processed_data_path = os.path.join(self.processed_data_dir, DATA_INGESTION_PROCESSED_FILE)
        self.train_data_path = os.path.join(self.split_data_dir, DATA_INGESTION_TRAIN_FILE)
        self.test_data_path = os.path.join(self.split_data_dir, DATA_INGESTION_TEST_FILE)
        self.metadata_path = os.path.join(self.ingestion_dir, DATA_INGESTION_METADATA_FILE)
        self.schema_path = os.path.join(self.ingestion_dir, DATA_INGESTION_SCHEMA_FILE)


# Configuration Validation entity classes
class DataValidationConfig:
    def __init__(self):
        # Main artifact directory for validation
        self.validation_dir = os.path.join(ARTIFACTS_DIR, DATA_VALIDATION_DIR)
        self.timestamp_dir = os.path.join(self.validation_dir, TIMESTAMP)
        os.makedirs(self.validation_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.validation_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_REPORT_FILE)
        self.data_drift_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_DRIFT_REPORT)
        self.missing_columns_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_MISSING_FILE)
        self.data_type_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_TYPES_FILE)
        self.validation_status_path = os.path.join(self.validation_dir, DATA_VALIDATION_STATUS_FILE)

        # Thresholds
        self.max_missing_threshold = MAX_MISSING_THRESHOLD  
        self.drift_threshold = DRIFT_THRESHOLD
        self.numerical_column_threshold = NUMERICAL_COLUMN_THRESHOLD
        self.categorical_column_threshold = CATEGORICAL_COLUMN_THRESHOLD


# Configuration class for preprocessing
class DataPreprocessingConfig:
    def __init__(self):
        # Main artifact directory
        
        self.preprocessing_dir = os.path.join(ARTIFACTS_DIR, DATA_PREPROCESSING_DIR)
        self.timestamp_dir = os.path.join(self.preprocessing_dir, TIMESTAMP)

        # Preprocessing subdirectory
        os.makedirs(self.preprocessing_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.preprocessing_train_path = os.path.join(self.preprocessing_dir, DATA_PREPROCESSING_TRAIN_FILE)
        self.preprocessing_test_path = os.path.join(self.preprocessing_dir, DATA_PREPROCESSING_TEST_FILE)
        self.preprocessing_report_path = os.path.join(self.preprocessing_dir, DATA_PREPROCESSING_REPORT_FILE)


# Configuration Feature Engineering entity classes
class FeatureEngineeringConfig:
    def __init__(self):
        # Main artifact directory
    
        self.feature_engineering_dir = os.path.join(ARTIFACTS_DIR, FEATURE_ENGINEERING_DIR)
        self.timestamp_dir = os.path.join(self.feature_engineering_dir, TIMESTAMP)
        
        # Make sure the directory exists
        os.makedirs(self.feature_engineering_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.feature_engineering_train_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TRAIN_FILE)
        self.feature_engineering_test_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TEST_FILE)
        self.feature_engineering_report_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_REPORT_FILE)


# Configuration class for data transformation
class DataTransformationConfig:
    def __init__(self):
                
            # Transformation subdirectory
            self.transformer_data_dir = os.path.join(ARTIFACTS_DIR,DATA_TRANSFORMATION_DIR)
            self.timestamp_dir = os.path.join(self.transformer_data_dir, TIMESTAMP)
            # Make sure the directory exists
            os.makedirs(self.transformer_data_dir, exist_ok=True)
            os.makedirs(self.timestamp_dir, exist_ok=True)

            # File paths
            self.transformer_train_path = os.path.join(self.transformer_data_dir, DATA_TRANSFORMATION_X_FILE)
            self.transformer_test_path = os.path.join(self.transformer_data_dir, DATA_TRANSFORMATION_Y_FILE)
            self.transformer_object_path = os.path.join(self.transformer_data_dir, DATA_TRANSFORMATION_OBJECT_FILE)
            self.transformer_report_path = os.path.join(self.transformer_data_dir, DATA_TRANSFORMATION_REPORT_FILE)


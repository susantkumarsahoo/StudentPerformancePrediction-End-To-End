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
        self.ingestion_dir = os.path.join(ARTIFACTS_DIR, INGESTION_DATA_DIR, TIMESTAMP)
        
        # Subdirectories - Define paths first
        self.raw_data_dir = os.path.join(self.ingestion_dir, RAW_DATA_DIR)
        self.processed_data_dir = os.path.join(self.ingestion_dir, PROCESSED_DATA_DIR)
        self.split_data_dir = os.path.join(self.ingestion_dir, SPLIT_DATA_DIR)

        # Create directories after defining paths
        os.makedirs(self.ingestion_dir, exist_ok=True)
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
        # Main artifact directory for validation
        self.validation_dir = os.path.join(ARTIFACTS_DIR, VALIDATION_DATA_DIR, TIMESTAMP)
        os.makedirs(self.validation_dir, exist_ok=True)

        # File paths
        self.validation_report_path = os.path.join(self.validation_dir, VALIDATION_REPORT_FILE_NAME)
        self.data_drift_report_path = os.path.join(self.validation_dir, DATA_DRIFT_REPORT_FILE_NAME)
        self.missing_columns_report_path = os.path.join(self.validation_dir, MISSING_COLUMNS_FILE_NAME)
        self.data_type_report_path = os.path.join(self.validation_dir, DATA_TYPE_FILE_NAME)
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
        
        self.preprocessing_dir = os.path.join(ARTIFACTS_DIR, PREPROCESSING_DATA_DIR, TIMESTAMP)

        # Preprocessing subdirectory
        os.makedirs(self.preprocessing_dir, exist_ok=True)

        # File paths
        self.preprocessing_train_path = os.path.join(self.preprocessing_dir, PREPROCESSING_TRAIN_FILE_NAME)
        self.preprocessing_test_path = os.path.join(self.preprocessing_dir, PREPROCESSING_TEST_FILE_NAME)
        self.preprocessing_report_path = os.path.join(self.preprocessing_dir, PREPROCESSING_REPORT_FILE_NAME)


# Configuration Feature Engineering entity classes
class FeatureEngineeringConfig:
    def __init__(self):
        # Main artifact directory
    
        self.feature_engineering_dir = os.path.join(ARTIFACTS_DIR, FEATURE_ENGINEERING_DATA_DIR, TIMESTAMP)
        
        # Make sure the directory exists
        os.makedirs(self.feature_engineering_dir, exist_ok=True)
        
        # File paths
        self.feature_engineering_train_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TRAIN_FILE_NAME)
        self.feature_engineering_test_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TEST_FILE_NAME)
        self.feature_engineering_report_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_REPORT_FILE_NAME)


# Configuration class for data transformation
class DataTransformationConfig:
    def __init__(self):
                
            # Transformation subdirectory
            self.transformer_data_dir = os.path.join(ARTIFACTS_DIR, TRANSFORMER_DATA_DIR, TIMESTAMP)
            os.makedirs(self.transformer_data_dir, exist_ok=True)

            # File paths
            self.transformer_train_path = os.path.join(self.transformer_data_dir, TRANSFORMER_TRAIN_FILE_NAME)
            self.transformer_test_path = os.path.join(self.transformer_data_dir, TRANSFORMER_TEST_FILE_NAME)
            self.transformer_object_path = os.path.join(self.transformer_data_dir, TRANSFORMER_OBJECT_FILE_NAME)
            self.transformer_report_path = os.path.join(self.transformer_data_dir, TRANSFORMER_REPORT_FILE_NAME)


# Configuration class for model training
class ModelTrainingConfig:
    def __init__(self):
        # Main artifact directory for model training
        self.model_training_dir = os.path.join(ARTIFACTS_DIR, MODEL_TRAINING_DIR, TIMESTAMP)
        os.makedirs(self.model_training_dir, exist_ok=True)

        # File paths
        self.best_model_path = os.path.join(self.model_training_dir, BEST_MODEL_FILE_NAME)
        self.model_evaluation_report_path = os.path.join(self.model_training_dir, MODEL_REPORT_FILE_NAME)


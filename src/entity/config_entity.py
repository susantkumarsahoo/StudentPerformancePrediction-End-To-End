import os
import os
from src.constants.constants import (
    ARTIFACTS_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    SPLIT_DATA_DIR, 
    VALIDATION_DATA_DIR,
    TIMESTAMP,
    MAX_MISSING_THRESHOLD,
    DRIFT_THRESHOLD,
    NUMERICAL_COLUMN_THRESHOLD,
    CATEGORICAL_COLUMN_THRESHOLD)

class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_path = dataset_path
        self.artifact_dir = os.path.join(ARTIFACTS_DIR,"data_ingestion",TIMESTAMP)
        self.test_size = test_size
        self.random_state = random_state


class DataValidationConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)
        self.max_missing_threshold = MAX_MISSING_THRESHOLD
        self.drift_threshold = DRIFT_THRESHOLD
        self.numerical_column_threshold = NUMERICAL_COLUMN_THRESHOLD
        self.categorical_column_threshold = CATEGORICAL_COLUMN_THRESHOLD




class DataPreprocessingConfig:
    """Configuration class for data preprocessing"""
    artifacts_dir: str = ARTIFACTS_DIR
    preprocessing_dir: str = "data_preprocessing"
    preprocessor_file_name: str = "preprocessor.pkl"
    transformed_train_file_name: str = "transformed_train.csv"
    transformed_test_file_name: str = "transformed_test.csv"
    preprocessing_report_file_name: str = "preprocessing_report.json"
    feature_engineering_report_file_name: str = "feature_engineering_report.json"





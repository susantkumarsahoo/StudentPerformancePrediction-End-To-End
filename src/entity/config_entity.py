import os
import os
from src.constants.constants import*


class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_path = dataset_path
        self.artifact_dir = os.path.join(ARTIFACTS_DIR, INGESTION_DATA_DIR)
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
    def __init__(self):
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)





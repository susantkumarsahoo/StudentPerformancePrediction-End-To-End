import os
from src.constants.constants import ARTIFACTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLIT_DATA_DIR

class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_path = dataset_path
        self.artifact_dir = os.path.join(ARTIFACTS_DIR, "data_ingestion")
        self.test_size = test_size
        self.random_state = random_state

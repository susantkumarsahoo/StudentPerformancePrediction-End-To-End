import os
from datetime import datetime

ARTIFACTS_DIR = os.path.join("artifacts")

# Data Ingestion Constants
RAW_DATA_DIR = "raw"
PROCESSED_DATA_DIR = "processed"
SPLIT_DATA_DIR = "split"
RAW_FILE_NAME = "raw_data.csv"
PROCESSED_FILE_NAME = "processed_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
METADATA_FILE_NAME = "metadata.json"
SCHEMA_FILE_NAME = "schema.json"
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data validation Constants














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



# Data Validation Constants
VALIDATION_DATA_DIR = "validation"
DATA_DRIFT_REPORT_FILE_NAME = "data_drift_report.json"
DATA_VALIDATION_STATUS_FILE = "validation_status.json"
MISSING_COLUMNS_FILE_NAME = "missing_columns.json"
MISSING_NUMERICAL_COLUMNS_FILE_NAME = "missing_numerical_columns.json"
MISSING_CATEGORICAL_COLUMNS_FILE_NAME = "missing_categorical_columns.json"
DATA_TYPE_FILE_NAME = "data_types.json"
VALIDATION_REPORT_FILE_NAME = "validation_report.json"


# Validation Thresholds
MAX_MISSING_THRESHOLD = 0.7
DRIFT_THRESHOLD = 0.05
NUMERICAL_COLUMN_THRESHOLD = 0.1
CATEGORICAL_COLUMN_THRESHOLD = 0.1
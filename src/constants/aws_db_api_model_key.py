import os

# Base artifact folder
ARTIFACT_DIR = "artifact"

# Raw data path
RAW_DATA_DIR = os.path.join(ARTIFACT_DIR, "raw")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "data.csv")

# Data ingestion output
DATA_INGESTION_DIR = os.path.join(ARTIFACT_DIR, "data_ingestion")
TRAIN_FILE = os.path.join(DATA_INGESTION_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_INGESTION_DIR, "test.csv")

# Test size split
TEST_SIZE = 0.2
RANDOM_STATE = 42

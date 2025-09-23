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
VALIDATION_DATA_DIR = "data_validation"
DATA_DRIFT_REPORT_FILE_NAME = "data_drift_report.json"
DATA_VALIDATION_STATUS_FILE = "validation_status.json"
MISSING_COLUMNS_FILE_NAME = "missing_columns.json"
MISSING_NUMERICAL_COLUMNS_FILE_NAME = "missing_numerical_columns.json"
MISSING_CATEGORICAL_COLUMNS_FILE_NAME = "missing_categorical_columns.json"
DATA_TYPE_FILE_NAME = "data_types.json"
VALIDATION_REPORT_FILE_NAME = "validation_report.json"

# Validation Thresholds
MAX_MISSING_THRESHOLD = 0.3
DRIFT_THRESHOLD = 0.05
NUMERICAL_COLUMN_THRESHOLD = 0.5
CATEGORICAL_COLUMN_THRESHOLD = 0.5


# Data Preprocessing Constants
PREPROCESSING_DIR = "data_preprocessing"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
TRANSFORMED_TRAIN_FILE_NAME = "transformed_train.csv"
TRANSFORMED_TEST_FILE_NAME = "transformed_test.csv"
PREPROCESSING_REPORT_FILE_NAME = "preprocessing_report.json"

# Feature Engineering Constants
FEATURE_ENGINEERING_DIR = "feature_engineering"
FEATURE_ENGINEERED_TRAIN_FILE_NAME = "feature_engineered_train.csv"
FEATURE_ENGINEERED_TEST_FILE_NAME = "feature_engineered_test.csv"
FEATURE_ENGINEERING_REPORT_FILE_NAME = "feature_engineering_report.json"


# Data Transformation Constants
TRANSFORMATION_DIR = "data_transformation"
TRANSFORMED_TRAIN_FILE_NAME = "transformed_train.csv"
TRANSFORMED_TEST_FILE_NAME = "transformed_test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TRANSFORMATION_REPORT_FILE_NAME = "transformation_report.json"


# Model Training Constants
MODEL_TRAINING_DIR = "model_training"
BEST_MODEL_FILE_NAME = "best_model.pkl"
MODEL_REPORT_FILE_NAME = "model_rport.json"


# Model Evaluation Constants
MODEL_EVALUATION_DIR = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME = "model_evaluation_report.json"


# Model Deployment Constants
MODEL_DEPLOYMENT_DIR = "model_deployment"
DEPLOYED_MODEL_DIR = "deployed_model"
DEPLOYED_MODEL_FILE_NAME = "deployed_model.pkl"
DEPLOYED_PREPROCESSOR_FILE_NAME = "deployed_preprocessor.pkl"
DEPLOYED_METRICS_FILE_NAME = "deployed_model_metrics.json"
DEPLOYED_ARTIFACTS_DIR = os.path.join("deployed_artifacts")


# DATABASE CONSTANTS
DATABASE_NAME = "student_performance_db"
COLLECTION_NAME = "student_performance_collection"

# LOGGING CONSTANTS
LOG_DIR = "logs"
LOG_FILE_NAME = "application.log"

















import os
from datetime import datetime

ARTIFACTS_DIR = os.path.join("artifacts")
DEPLOYED_ARTIFACTS_DIR = os.path.join("deployed_artifacts")

# Data Ingestion Constants
INGESTION_DATA_DIR  = "data_ingestion"
RAW_DATA_DIR        = "raw"
PROCESSED_DATA_DIR  = "processed"
SPLIT_DATA_DIR      = "split"
RAW_FILE_NAME       = "raw_data.csv"
PROCESSED_FILE_NAME = "processed_data.csv"
TRAIN_FILE_NAME     = "train.csv"
TEST_FILE_NAME      = "test.csv"
METADATA_FILE_NAME  = "metadata.json"
SCHEMA_FILE_NAME    = "schema.json"
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Data Validation Constants
VALIDATION_DATA_DIR         = "data_validation"
DATA_DRIFT_REPORT_FILE_NAME = "data_drift_report.json"
DATA_VALIDATION_STATUS_FILE = "validation_status.json"
MISSING_COLUMNS_FILE_NAME   = "missing_columns.json"
DATA_TYPE_FILE_NAME         = "data_types.json"
VALIDATION_REPORT_FILE_NAME = "validation_report.json"


# Validation Thresholds
MAX_MISSING_THRESHOLD        = 0.3
DRIFT_THRESHOLD              = 0.05
NUMERICAL_COLUMN_THRESHOLD   = 0.5
CATEGORICAL_COLUMN_THRESHOLD = 0.5


# Data Preprocessing Constants
PREPROCESSING_DATA_DIR           = "data_preprocessing"
PREPROCESSING_TRAIN_FILE_NAME    = "preprocessing_train.csv"
PREPROCESSING_TEST_FILE_NAME     = "preprocessing_test.csv"
PREPROCESSING_REPORT_FILE_NAME   = "preprocessing_report.json"


# Feature Engineering Constants
FEATURE_ENGINEERIN_DATA_DIR           = "feature_engineering"
FEATURE_ENGINEERING_TRAIN_FILE_NAME   = "feature_engineering_train.csv"
FEATURE_ENGINEERING_TEST_FILE_NAME    = "feature_engineering_test.csv"
FEATURE_ENGINEERING_REPORT_FILE_NAME  = "feature_engineering_report.json"


# Data Transformation Constants
TRANSFORMER_DATA_DIR            = "data_transformation"
TRANSFORMER_TRAIN_FILE_NAME     = "transformer_train.csv"
TRANSFORMER_TEST_FILE_NAME      = "transformer_test.csv"
TRANSFORMER_OBJECT_FILE_NAME    = "transformer_preprocessor.pkl"
TRANSFORMER_REPORT_FILE_NAME    = "transformer_report.json"


# Model Training Constants
MODEL_TRAINING_DIR     = "model_training"
BEST_MODEL_FILE_NAME   = "best_model.pkl"
MODEL_REPORT_FILE_NAME = "model_rport.json"


# Model Evaluation Constants
MODEL_EVALUATION_DIR              = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME = "model_evaluation_report.json"


# Model Deployment Constants
MODEL_DEPLOYMENT_DIR            = "model_deployment"
DEPLOYMENT_MODEL_DIR              = "deployed_model"
DEPLOYMENT_MODEL_FILE_NAME        = "deployed_model.pkl"
DEPLOYMENT_PREPROCESSOR_FILE_NAME = "deployed_preprocessor.pkl"
DEPLOYMENT_MODEL_REPORT_FILE_NAME      = "deployed_model_report.json"


# DATABASE CONSTANTS
DATABASE_NAME             = "student_performance_db"
COLLECTION_NAME           = "student_performance_collection"
DATABASE_REPORT_FILE_NAME = "database_report.json"


# LOGGING CONSTANTS
LOG_DIR       = "logs"
LOG_FILE_NAME = "application.log"

















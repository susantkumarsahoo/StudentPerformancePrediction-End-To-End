import os
import sys
from src.constants.constants import *


# Configuration class for model training
class ModelTrainingConfig:
    def __init__(self):
        # Main artifact directory for model training
        self.model_training_dir = os.path.join(ARTIFACTS_DIR, MODEL_TRAINING_DIR)
        self.timestamp_dir = os.path.join(self.model_training_dir, TIMESTAMP)

        # Make sure the directory exists
        os.makedirs(self.model_training_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.best_model_path = os.path.join(self.model_training_dir, BEST_MODEL_FILE_NAME)
        self.model_evaluation_report_path = os.path.join(self.model_training_dir, MODEL_REPORT_FILE_NAME)

# Configuration class for model evaluation
class ModelEvaluationConfig:
    def __init__(self):
        # Main artifact directory for model evaluation
        self.model_evaluation_dir = os.path.join(ARTIFACTS_DIR, MODEL_EVALUATION_DIR)
        self.timestamp_dir = os.path.join(self.model_evaluation_dir, TIMESTAMP)

        # Make sure the directory exists
        os.makedirs(self.model_evaluation_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.model_evaluation_report_path = os.path.join(self.model_evaluation_dir, MODEL_EVALUATION_REPORT_FILE_NAME)

# Configuration class for model deployment
class ModelDeploymentConfig:
    def __init__(self):
        # Main artifact directory for model deployment
        self.model_deployment_dir = os.path.join(ARTIFACTS_DIR, MODEL_DEPLOYMENT_DIR)
        self.timestamp_dir = os.path.join(self.model_deployment_dir, TIMESTAMP)

        # Make sure the directory exists
        os.makedirs(self.model_deployment_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.deployment_model_path = os.path.join(self.model_deployment_dir, DEPLOYMENT_MODEL_FILE_NAME)
        self.deployment_preprocessor_path = os.path.join(self.model_deployment_dir, DEPLOYMENT_PREPROCESSOR_FILE_NAME)
        self.deployment_report_path = os.path.join(self.model_deployment_dir, DEPLOYMENT_MODEL_REPORT_FILE_NAME)

       
# Configuration class for database
class DatabaseConfig:
    def __init__(self):
        # Main artifact directory for database
        self.database_dir = os.path.join(ARTIFACTS_DIR, DATABASE_DIR)
        self.timestamp_dir = os.path.join(self.database_dir, TIMESTAMP)

        # Make sure the directory exists
        os.makedirs(self.database_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.database_file_path = os.path.join(self.database_dir, DATABASE_FILE_NAME)
        self.collection_path = os.path.join(self.database_dir, COLLECTION_NAME)
        self.database_report_path = os.path.join(self.database_dir, DATABASE_REPORT_FILE_NAME)


# Configuration class for logging
class LoggingConfig:
    def __init__(self):
        # Main artifact directory for logging
        self.log_dir = os.path.join(ARTIFACTS_DIR, LOG_DIR)
        self.timestamp_dir = os.path.join(self.log_dir, TIMESTAMP)

        # Make sure the directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # File paths
        self.log_file_path = os.path.join(self.log_dir, LOG_DIR)
        self.log_report_path = os.path.join(self.log_dir, LOG_REPORT_FILE_NAME)
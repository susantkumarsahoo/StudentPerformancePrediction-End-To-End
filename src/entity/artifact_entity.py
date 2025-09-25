from dataclasses import dataclass

# Artifact entity classes
@dataclass
class DataIngestionArtifact:
    raw_data_path: str
    processed_data_path: str
    train_data_path: str
    test_data_path: str
    metadata_path: str
    schema_path: str

# Data Validation Artifact
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str
    data_drift_report_path: str
    missing_columns_report_path: str
    data_type_report_path: str
    validation_status_path: str

# Data Preprocessing Artifact
@dataclass
class DataPreprocessingArtifact:
    preprocessing_train_path: str
    preprocessing_test_path: str
    preprocessing_report_path: str

# Feature Engineering Artifact
@dataclass
class FeatureEngineeringArtifact:
    feature_engineering_train_path: str
    feature_engineering_test_path: str
    feature_engineering_report_path: str

# Data Transformation Artifact
@dataclass
class DataTransformationArtifact:
    transformer_train_path: str
    transformer_test_path: str
    transformer_object_path: str
    transformer_report_path: str

# Model Training Artifact
@dataclass
class ModelTrainingArtifact:
    best_model_path: str
    model_evaluation_report_path: str

# Model Evaluation Artifact
@dataclass
class ModelEvaluationArtifact:
    model_evaluation_report_path: str
    is_model_accepted: bool

# Model Deployment Artifact
@dataclass
class ModelDeploymentArtifact:
    deployed_model_path: str
    deployment_status: bool
    deployment_preprocessor_path: str
    deployment_report_path: str

# Database Artifact
@dataclass
class DatabaseArtifact:
    database_file_path: str
    collection_path: str
    database_report_path: str

# Logging Artifact
@dataclass
class LoggingArtifact:
    log_file_path: str
    log_report_path: str












    
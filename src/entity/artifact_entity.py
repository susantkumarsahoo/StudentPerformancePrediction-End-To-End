from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str
    processed_data_path: str
    train_data_path: str
    test_data_path: str
    metadata_path: str
    schema_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str
    data_drift_report_path: str
    missing_columns_report_path: str
    data_type_report_path: str
    validation_status_path: str
    
@dataclass
class DataPreprocessingArtifact:
    preprocessor_path: str
    transformed_train_path: str
    transformed_test_path: str
    preprocessing_report_path: str
    is_preprocessing_successful: bool
     
@dataclass
class FeatureEngineeringArtifact:
    feature_engineered_train_path: str
    feature_engineered_test_path: str
    feature_engineering_report_path: str
    is_feature_engineering_successful: bool

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    transformer_object_path: str
    target_encoder_object_path: str
    transformation_report_path: str
    is_transformation_successful: bool
    
@dataclass
class ModelTrainingArtifact:
    model_path: str
    training_report_path: str
    is_model_trained: bool
    model_accuracy: float 

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    model_evaluation_report_path: str
@dataclass
class ModelPusherArtifact:
    deployed_model_path: str
    deployed_transformer_path: str
    deployed_target_encoder_path: str
    is_model_pushed: bool
    model_pusher_report_path: str
@dataclass
class TrainingPipelineArtifact:
    data_ingestion_artifact: DataIngestionArtifact
    data_validation_artifact: DataValidationArtifact
    data_preprocessing_artifact: DataPreprocessingArtifact
    feature_engineering_artifact: FeatureEngineeringArtifact











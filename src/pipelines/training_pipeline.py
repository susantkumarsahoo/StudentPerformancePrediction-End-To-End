import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.feature_transformer import FeatureTransformer
from src.models.trainer import ModelTrainer
from src.models.model_registry import ModelRegistry



from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, 
                                        DataPreprocessingArtifact, FeatureEngineeringArtifact, 
                                        DataTransformationArtifact,ModelTrainingArtifact )



from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, 
                                      DataPreprocessingConfig, FeatureEngineeringConfig, 
                                      DataTransformationConfig)

from src.entity.model_entity import ModelTrainingConfig, ModelDeploymentConfig


class TrainingPipeline:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_pipeline(self):
        # Step 1: Data Ingestion
        ingestion_config = DataIngestionConfig(dataset_path=self.dataset_path)
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()


        # Step 2: Data Validation
        validation_config = DataValidationConfig()
        validation = DataValidation(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_config=validation_config
        )
        validation_artifact = validation.initiate_data_validation()
        
        # Step 3: Data Preprocessing
        preprocessing_config = DataPreprocessingConfig()
        preprocessing = DataPreprocessing(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_artifact=validation_artifact,
            data_preprocessing_config=preprocessing_config
        )
        preprocessing_artifact = preprocessing.initiate_data_preprocessing()
        # Step 4: Feature Engineering
        feature_engineering_config = FeatureEngineeringConfig()
        feature_engineering = FeatureEngineering(
            data_preprocessing_artifact=preprocessing_artifact,
            feature_engineering_config=feature_engineering_config
        )
        feature_engineering_artifact = feature_engineering.initiate_feature_engineering()
        # Further steps like Data Transformation
        transformation_config = DataTransformationConfig()
        feature_transformation = FeatureTransformer(
            feature_engineering_artifact=feature_engineering_artifact,
            data_transformation_config=transformation_config
        )
        data_transformation_artifact = feature_transformation.initiate_data_transformation()


        # Step 5: Model Training
        model_training_config = ModelTrainingConfig()
        model_trainer = ModelTrainer(
            model_training_config=model_training_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_training_artifact = model_trainer.initiate_model_trainer()
        
        
        # Step 5: Model Training
        model_training_config = ModelTrainingConfig()
        model_trainer = ModelTrainer(
            model_training_config=model_training_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_training_artifact = model_trainer.initiate_model_trainer()
        
        # model deployment
        model_deployment_config = ModelDeploymentConfig()
        model_registry = ModelRegistry(
            model_deployment_config=model_deployment_config,
            data_transformation_artifact=data_transformation_artifact,
            model_training_artifact=model_training_artifact
        )
        model_registry.save_model_and_preprocessor()
        model_deployment_artifact = model_registry.initiate_model_registry()
        return model_deployment_artifact


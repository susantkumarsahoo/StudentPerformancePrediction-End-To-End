import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.feature_transformer import FeatureTransformer
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluation
from src.models.model_registry import ModelRegistry
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException

from src.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact, 
    DataPreprocessingArtifact, 
    FeatureEngineeringArtifact, 
    DataTransformationArtifact,
    ModelTrainingArtifact, 
    ModelDeploymentArtifact
)

from src.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataPreprocessingConfig, 
    FeatureEngineeringConfig, 
    DataTransformationConfig
)

from src.entity.model_config_entity import (
    ModelTrainingConfig, 
    ModelDeploymentConfig,
    ModelEvaluationConfig
)

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, dataset_path):
        try:
            self.dataset_path = dataset_path
            self.model_deployment_artifact = None
            logger.info("✅ TrainingPipeline initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logger.info("🚀 Starting Training Pipeline...")
            
            # Step 1: Data Ingestion
            logger.info("🔹 Step 1: Data Ingestion")
            ingestion_config = DataIngestionConfig(dataset_path=self.dataset_path)
            ingestion = DataIngestion(ingestion_config)
            ingestion_artifact = ingestion.initiate_data_ingestion()
            logger.info("✅ Data Ingestion completed")

            # Step 2: Data Validation
            logger.info("🔹 Step 2: Data Validation")
            validation_config = DataValidationConfig()
            validation = DataValidation(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_config=validation_config
            )
            validation_artifact = validation.initiate_data_validation()
            logger.info("✅ Data Validation completed")
            
            # Step 3: Data Preprocessing
            logger.info("🔹 Step 3: Data Preprocessing")
            preprocessing_config = DataPreprocessingConfig()
            preprocessing = DataPreprocessing(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_artifact=validation_artifact,
                data_preprocessing_config=preprocessing_config
            )
            preprocessing_artifact = preprocessing.initiate_data_preprocessing()
            logger.info("✅ Data Preprocessing completed")
            
            # Step 4: Feature Engineering
            logger.info("🔹 Step 4: Feature Engineering")
            feature_engineering_config = FeatureEngineeringConfig()
            feature_engineering = FeatureEngineering(
                data_preprocessing_artifact=preprocessing_artifact,
                feature_engineering_config=feature_engineering_config
            )
            feature_engineering_artifact = feature_engineering.initiate_feature_engineering()
            logger.info("✅ Feature Engineering completed")
            
            # Step 5: Data Transformation
            logger.info("🔹 Step 5: Data Transformation")
            transformation_config = DataTransformationConfig()
            feature_transformation = FeatureTransformer(
                feature_engineering_artifact=feature_engineering_artifact,
                data_transformation_config=transformation_config
            )
            data_transformation_artifact = feature_transformation.initiate_data_transformation()
            logger.info("✅ Data Transformation completed")

            # Step 6: Model Training
            logger.info("🔹 Step 6: Model Training")
            model_training_config = ModelTrainingConfig()
            model_trainer = ModelTrainer(
                model_training_config=model_training_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_training_artifact = model_trainer.initiate_model_trainer()
            logger.info("✅ Model Training completed")
            
            # model evaluation
            logger.info("🔹 Step 7: Model Evaluation")
            model_evaluation_config = ModelEvaluationConfig()
            model_eval = ModelEvaluation(
                model_evaluation_config=model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_training_artifact=model_training_artifact
            )
            model_evaluation_artifact = model_eval.initiate_model_evaluation()
            logger.info("✅ Model Evaluation completed")
        
            # Step 7: Model Deployment/Registry
            logger.info("🔹 Step 7: Model Deployment")
            model_deployment_config = ModelDeploymentConfig()
            model_registry = ModelRegistry(
                model_deployment_config=model_deployment_config,
                data_transformation_artifact=data_transformation_artifact,
                model_training_artifact=model_training_artifact
            )
            model_registry.save_model_and_preprocessor()
            self.model_deployment_artifact = model_registry.initiate_model_registry()
            logger.info("✅ Model Deployment completed")

            logger.info("🎉 Training Pipeline completed successfully!")
            
            # Return the deployment artifact
            return self.model_deployment_artifact
            
        except Exception as e:
            logger.error("❌ Error in Training Pipeline", exc_info=True)
            raise CustomException(e, sys)


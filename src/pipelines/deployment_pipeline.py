import os
import sys
import json
import joblib
from src.pipelines.training_pipeline import TrainingPipeline
from src.models.model_registry import ModelRegistry
import os
import sys
import json
import joblib
from datetime import datetime
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import ModelTrainingArtifact
from src.entity.artifact_entity import ModelDeploymentArtifact
from src.entity.model_entity import ModelDeploymentConfig

from src.logging.logger import get_logger
from src.exceptions.exception import CustomException


logger = get_logger(__name__)



class DeploymentPipeline:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model_deployment_config = ModelDeploymentConfig()

    def run_deployment_pipeline(self):
        """
        Run the deployment pipeline after training is complete.
        """
        try:
            # Step 1: Run training pipeline
            pipeline = TrainingPipeline(dataset_path=self.dataset_path)
            artifacts_pipeline = pipeline.run_pipeline()



            # Extract artifacts from the returned object
            data_transformation_artifact = artifacts_pipeline.data_transformation_artifact
            model_training_artifact = artifacts_pipeline.model_training_artifact



            # Step 2: Initialize Model Registry and deploy
            deploy_registry = ModelRegistry(
                model_deployment_config=self.model_deployment_config,
                data_transformation_artifact=data_transformation_artifact,
                model_training_artifact=model_training_artifact
            )

            # Initiate registry
            model_deployment_artifact = deploy_registry.initiate_model_registry()

            return model_deployment_artifact

        except Exception as e:
            raise CustomException(e, sys)
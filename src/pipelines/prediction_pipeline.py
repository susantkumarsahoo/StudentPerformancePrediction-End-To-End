import os
import sys
import json
import joblib
from src.exceptions.exception import CustomException
from src.pipelines.training_pipeline import TrainingPipeline
from src.entity.artifact_entity import ModelDeploymentArtifact
from src.models.predictor import ModelPredictor

class PredictionPipeline:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_prediction_pipeline(self):
        try:
            # ---- OPTION 1: Run training pipeline ----
            training_pipeline = TrainingPipeline(dataset_path=self.dataset_path)
            model_deployment_artifact = training_pipeline.run_pipeline()

            # Ensure we got the right artifact type
            if not isinstance(model_deployment_artifact, ModelDeploymentArtifact):
                raise ValueError("Training pipeline did not return ModelDeploymentArtifact")

            # ---- OPTION 2: Load from saved JSON ----
            # with open('deployment_artifacts/model/deployed_model_report.json', 'r') as f:
            #     artifact_data = json.load(f)
            # model_deployment_artifact = ModelDeploymentArtifact(**artifact_data)

            # Run prediction
            model_predictor = ModelPredictor(model_deployment_artifact=model_deployment_artifact)
            model_predictor()

        except Exception as e:
            raise CustomException(e, sys)






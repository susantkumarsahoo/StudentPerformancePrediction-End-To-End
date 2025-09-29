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



class ModelRegistry:
    """
    Handles saving trained model, preprocessor, and deployment report.
    """

    def __init__(
        self,
        model_deployment_config: ModelDeploymentConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_artifact: ModelTrainingArtifact,
    ):
        try:
            self.model_deployment_config = model_deployment_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_training_artifact = model_training_artifact
            logger.info("ModelRegistry initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def save_model_and_preprocessor(self):
        """
        Save trained model and preprocessor to deployment paths.
        """
        try:
            # Save model
            model_obj = self.model_training_artifact.best_model_path
            joblib.dump(model_obj, self.model_deployment_config.deployment_model_path)
            logger.info(f"Model saved successfully at: {self.model_deployment_config.deployment_model_path}")

            # Save preprocessor
            preprocessor_obj = self.data_transformation_artifact.transformer_object_path
            joblib.dump(preprocessor_obj, self.model_deployment_config.deployment_preprocessor_path)
            logger.info(f"Preprocessor saved at: {self.model_deployment_config.deployment_preprocessor_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def save_deployment_report(self):
        """
        Save metadata report about deployment.
        """
        try:
            report = {
                "model_path": self.model_deployment_config.deployment_model_path,
                "preprocessor_path": self.model_deployment_config.deployment_preprocessor_path,
                "metrics": self.model_deployment_config.deployment_report_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(self.model_deployment_config.deployment_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Deployment report saved at: {self.model_deployment_config.deployment_report_path}")
            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_registry(self) -> ModelDeploymentArtifact:
        """
        Run the full model registry process:
        - Save model & preprocessor
        - Save deployment report
        - Return deployment artifact
        """
        try:
            self.save_model_and_preprocessor()
            report = self.save_deployment_report()

            model_deployment_artifact = ModelDeploymentArtifact(
                deployment_model_path=self.model_deployment_config.deployment_model_path,
                deployment_preprocessor_path=self.model_deployment_config.deployment_preprocessor_path,
                deployment_report_path=self.model_deployment_config.deployment_report_path,
                deployment_status=True
            )

            logger.info(f"ModelDeploymentArtifact created: {model_deployment_artifact}")
            logger.info(f"ModelDeploymentArtifact saved at: {os.path.abspath(model_deployment_artifact.deployment_report_path)}")
            logger.info(f"ModelDeploymentArtifact saved successfully")
            return model_deployment_artifact

        except Exception as e:
            raise CustomException(e, sys)


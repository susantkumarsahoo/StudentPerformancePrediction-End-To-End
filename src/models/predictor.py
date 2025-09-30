import os
import sys
import joblib
import pandas as pd
from src.exceptions.exception import CustomException
from src.logging.logger import get_logger
from src.entity.artifact_entity import ModelDeploymentArtifact

logger = get_logger(__name__)

class ModelPredictor:
    def __init__(self, model_deployment_artifact: ModelDeploymentArtifact):
        try:
            if not isinstance(model_deployment_artifact, ModelDeploymentArtifact):
                raise ValueError(
                    f"Expected ModelDeploymentArtifact instance, got {type(model_deployment_artifact)}"
                )

            self.model_deployment_artifact = model_deployment_artifact

            # Check model + preprocessor existence
            if not os.path.exists(self.model_deployment_artifact.deployment_model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_deployment_artifact.deployment_model_path}"
                )

            if not os.path.exists(self.model_deployment_artifact.deployment_preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found: {self.model_deployment_artifact.deployment_preprocessor_path}"
                )

            # Load model and preprocessor once during initialization
            self.model = joblib.load(self.model_deployment_artifact.deployment_model_path)
            self.preprocessor = joblib.load(self.model_deployment_artifact.deployment_preprocessor_path)

            logger.info("✅ ModelPredictor initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            # Convert input dict to DataFrame
            input_df = pd.DataFrame([input_data])
            logger.info("Input data converted to DataFrame for prediction.")

            # Transform input
            X_processed = self.preprocessor.transform(input_df)

            # Make prediction
            prediction = self.model.predict(X_processed)
            logger.info(f"Prediction: {prediction[0]}")
            print(f"🎯 Predicted value: {prediction[0]}")
            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)



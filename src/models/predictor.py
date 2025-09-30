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

            # Check existence
            if not os.path.exists(self.model_deployment_artifact.deployment_model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_deployment_artifact.deployment_model_path}"
                )

            if not os.path.exists(self.model_deployment_artifact.deployment_preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found: {self.model_deployment_artifact.deployment_preprocessor_path}"
                )

            # Load model and preprocessor
            self.model = joblib.load(self.model_deployment_artifact.deployment_model_path)
            self.preprocessor = joblib.load(self.model_deployment_artifact.deployment_preprocessor_path)

            # Save training columns for feature consistency
            if hasattr(self.preprocessor, "get_feature_names_out"):
                self.feature_columns = self.preprocessor.get_feature_names_out()
            else:
                # fallback if preprocessor is older
                self.feature_columns = None

            logger.info("✅ ModelPredictor initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            # Convert input dict to DataFrame
            input_df = pd.DataFrame([input_data])
            logger.info("Input data converted to DataFrame.")

            # Drop target column if exists
            input_features = input_df.drop(columns=['math_score'], errors='ignore')

            # Transform input
            X_processed = self.preprocessor.transform(input_features)

            # Handle feature mismatch
            if self.feature_columns is not None and X_processed.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Feature mismatch: expected {len(self.feature_columns)} features, got {X_processed.shape[1]}"
                )

            # Predict
            prediction = self.model.predict(X_processed)
            logger.info(f"Prediction: {prediction[0]}")
            print(f"🎯 Predicted value: {prediction[0]}")
            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)




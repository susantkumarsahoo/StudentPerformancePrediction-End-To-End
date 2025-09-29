import os
import sys
import joblib
import pandas as pd
from src.exceptions.exception import CustomException
from src.logging.logger import get_logger
from src.entity.artifact_entity import ModelDeploymentArtifact
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)

class ModelPredictor:
    def __init__(self, model_deployment_artifact: ModelDeploymentArtifact):
        try:
            if not isinstance(model_deployment_artifact, ModelDeploymentArtifact):
                raise ValueError(f"Expected ModelDeploymentArtifact instance, got {type(model_deployment_artifact)}")
            
            self.model_deployment_artifact = model_deployment_artifact
            logger.info("ModelPredictor initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
        
    def __call__(self) -> None:
        try:
            # Load model and preprocessor
            logger.info(f"Loading model from: {self.model_deployment_artifact.deployment_model_path}")
            logger.info(f"Loading preprocessor from: {self.model_deployment_artifact.deployment_preprocessor_path}")

            # Check if files exist
            if not os.path.exists(self.model_deployment_artifact.deployment_model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_deployment_artifact.deployment_model_path}")
            
            if not os.path.exists(self.model_deployment_artifact.deployment_preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.model_deployment_artifact.deployment_preprocessor_path}")

            # Load the actual objects
            model = joblib.load(self.model_deployment_artifact.deployment_model_path)
            preprocessor = joblib.load(self.model_deployment_artifact.deployment_preprocessor_path)

            logger.info(f"Model type: {type(model)}")
            logger.info(f"Preprocessor type: {type(preprocessor)}")

            # Example user input
            user_input = pd.DataFrame([{
                "gender": "female",
                "race_ethnicity": "group B",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
                "reading_score": 72,
                "writing_score": 74
            }])

            logger.info("Transforming user input...")
            # Transform input
            X_user = preprocessor.transform(user_input)

            logger.info("Making prediction...")
            # Predict
            predicted_score = model.predict(X_user)
            print("🎯 Predicted Math Score:", predicted_score[0])

            logger.info(f"Predicted Math Score: {predicted_score[0]}")
            
            return predicted_score[0]

        except Exception as e:
            raise CustomException(e, sys)

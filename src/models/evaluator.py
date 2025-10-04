import os
import sys
import json
import joblib
import numpy as np
from typing import Dict

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import (
    ModelEvaluationArtifact, 
    ModelTrainingArtifact,
    DataTransformationArtifact
)
from src.entity.model_config_entity import ModelEvaluationConfig
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


class ModelEvaluation:
    """
    Model Evaluation Component
    Evaluates a trained model against test data and generates evaluation report.
    """

    def __init__(self, 
                 model_evaluation_config: ModelEvaluationConfig,
                 model_training_artifact: ModelTrainingArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logger.info("Initializing Model Evaluation Component")
            self.model_evaluation_config = model_evaluation_config
            self.model_training_artifact = model_training_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self, model_path: str):
        """Load trained model from joblib file"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_transformed_data(self, data_path: str) -> np.ndarray:
        """Load transformed numpy array"""
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")

            data = np.load(data_path)
            logger.info(f"Transformed data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        try:
            y_pred = model.predict(X_test)

            metrics = {
                "R2_Score": float(r2_score(y_test, y_pred)),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "MSE": float(mean_squared_error(y_test, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def generate_report(self, metrics: Dict[str, float], is_accepted: bool) -> str:
        """Save evaluation report as JSON"""
        try:
            report = {
                "model_path": self.model_training_artifact.best_model_path,
                "is_model_accepted": is_accepted,
                "metrics": metrics
            }

            os.makedirs(os.path.dirname(self.model_evaluation_config.model_evaluation_report_path),
                        exist_ok=True)

            with open(self.model_evaluation_config.model_evaluation_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Evaluation report saved at {self.model_evaluation_config.model_evaluation_report_path}")
            return self.model_evaluation_config.model_evaluation_report_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Main method to evaluate model"""
        try:
            logger.info("Starting Model Evaluation")

            # Load model
            model = self.load_model(self.model_training_artifact.best_model_path)

            # Load transformed test data
            train_data = np.load(self.data_transformation_artifact.transformer_train_path)
            test_data = np.load(self.data_transformation_artifact.transformer_test_path)

            X_train, X_test, y_train, y_test = train_test_split(train_data,test_data,test_size=0.2,random_state=42)

            # Split features and target (assuming last column is target)
            
            logger.info(f"Test data shape - X: {X_test.shape}, y: {y_test.shape}")

            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)

            # Check acceptance criteria
            is_accepted = metrics["R2_Score"] >= 0.6

            # Save report
            report_path = self.generate_report(metrics, is_accepted)

            # Create artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                model_evaluation_report_path=report_path,
                is_model_accepted=is_accepted
            )

            logger.info("Model Evaluation Completed")
            return model_evaluation_artifact

        except Exception as e:
            logger.error("Error in Model Evaluation")
            raise CustomException(e, sys)


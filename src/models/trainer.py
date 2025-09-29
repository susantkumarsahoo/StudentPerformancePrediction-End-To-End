import os
import sys
import joblib
import json
import numpy as np
from src.constants.constants import *


from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, DataPreprocessingArtifact, 
                                        FeatureEngineeringArtifact, DataTransformationArtifact, ModelTrainingArtifact, 
                                        ModelDeploymentArtifact, ModelEvaluationArtifact)

from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig, 
                                      FeatureEngineeringConfig, DataTransformationConfig)

from src.entity.model_config_entity import (ModelTrainingConfig,ModelEvaluationConfig,ModelDeploymentConfig)

from src.exceptions.exception import CustomException
from src.logging.logger import get_logger
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, 
                 model_training_config: ModelTrainingConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
            logger.info("ModelTrainer initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Lasso:
        try:
            logger.info("Training Lasso regression model.")
            model = Lasso(alpha=0.01, random_state=42)  # Configurable
            model.fit(X_train, y_train)
            logger.info("Model training completed.")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model: Lasso, X: np.ndarray, y: np.ndarray) -> dict:
        try:
            logger.info("Evaluating the model.")
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            return {"mse": mse, "r2": r2}
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainingArtifact:
        try:
            logger.info("Loading transformed training and testing data.")
            train_data = np.load(self.data_transformation_artifact.transformer_train_path)
            test_data = np.load(self.data_transformation_artifact.transformer_test_path)

            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            logger.info("Starting model training process.")
            model = self.train_model(X_train, y_train)

            # Evaluate on both train and test
            train_metrics = self.evaluate_model(model, X_train, y_train)
            test_metrics = self.evaluate_model(model, X_test, y_test)

            logger.info(f"Train Metrics: {train_metrics}")
            logger.info(f"Test Metrics: {test_metrics}")

            # Performance check
            if test_metrics["r2"] < 0.6:
                raise CustomException("Trained model does not meet the performance criteria.", sys)

            logger.info("Model meets performance criteria. Proceeding to save the model.")

            # Save model
            joblib.dump(model, self.model_training_config.best_model_path)
            logger.info(f"Trained model saved at {self.model_training_config.best_model_path}")

            # Save combined evaluation report
            evaluation_report = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            }
            with open(self.model_training_config.model_evaluation_report_path, "w") as f:
                json.dump(evaluation_report, f, indent=4)

            logger.info(f"Model evaluation report saved at {self.model_training_config.model_evaluation_report_path}")
            
            logger.info(f"Best model created successfuly")

            return ModelTrainingArtifact(
                best_model_path=self.model_training_config.best_model_path,
                model_evaluation_report_path=self.model_training_config.model_evaluation_report_path
            )
        except Exception as e:
            raise CustomException(e, sys)

import os
import sys
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline, StudentDataInput
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException

logger = get_logger(__name__)


class DeploymentPipeline:
    def __init__(self, dataset_path):
        """Initialize Deployment Pipeline with dataset path."""
        try:
            self.dataset_path = dataset_path
            self.model_deployment_artifact = None
            logger.info("✅ DeploymentPipeline initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_deployment_pipeline(self):
        """
        Runs the full deployment pipeline:
        1. Train the model using TrainingPipeline
        2. Setup PredictionPipeline for future predictions
        """
        try:
            logger.info("🚀 Starting Deployment Pipeline...")

            # Step 1: Training
            logger.info("🔹 Running Training Pipeline...")
            training_pipeline = TrainingPipeline(dataset_path=self.dataset_path)
            self.model_deployment_artifact = training_pipeline.run_pipeline()
            logger.info("✅ Training Pipeline completed successfully.")

            # Step 2: Initialize Prediction Pipeline with the trained model artifact
            logger.info("🔹 Initializing Prediction Pipeline...")
            self.prediction_pipeline = PredictionPipeline(
                model_deployment_artifact=self.model_deployment_artifact
            )
            logger.info("✅ Prediction Pipeline initialized successfully.")

            logger.info("🎉 Deployment Pipeline executed successfully.")
            
            return self.model_deployment_artifact

        except Exception as e:
            logger.error("❌ Error occurred during Deployment Pipeline execution.", exc_info=True)
            raise CustomException(e, sys)
    
    def make_prediction(self, student_data: StudentDataInput):
        """
        Make a prediction using the deployed model.
        
        Args:
            student_data: StudentDataInput object with student information
            
        Returns:
            Predicted math score
        """
        try:
            if self.prediction_pipeline is None:
                raise ValueError("Prediction pipeline not initialized. Run deployment pipeline first.")
            
            logger.info("🔹 Making prediction...")
            prediction = self.prediction_pipeline.predict(student_data)
            logger.info(f"✅ Prediction completed: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error("❌ Error during prediction", exc_info=True)
            raise CustomException(e, sys)






        
import sys
from pandas import DataFrame
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.models.predictor import ModelPredictor
from src.entity.artifact_entity import ModelDeploymentArtifact

logger = get_logger(__name__)


class StudentDataInput:
    """
    Represents the input features for student performance prediction.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int) -> None:
        try:
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        except Exception as e:
            raise CustomException(e, sys)

    def to_dict(self) -> dict:
        try:
            return {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score,
            }
        except Exception as e:
            raise CustomException(e, sys)

    def to_dataframe(self) -> DataFrame:
        try:
            return DataFrame([self.to_dict()])
        except Exception as e:
            raise CustomException(e, sys)


class PredictionPipeline:
    """
    Service class for making predictions using trained model.
    """
    def __init__(self, model_deployment_artifact: ModelDeploymentArtifact = None):
        try:
            # Use provided artifact or create default one
            if model_deployment_artifact is None:
                model_deployment_artifact = ModelDeploymentArtifact()
            
            # Initialize ModelPredictor with the artifact
            self.model_predictor = ModelPredictor(
                model_deployment_artifact=model_deployment_artifact
            )
            logger.info("✅ PredictionPipeline initialized successfully.")
        except Exception as e:
            logger.error("❌ Error initializing PredictionPipeline", exc_info=True)
            raise CustomException(e, sys)

    def predict(self, input_data: StudentDataInput) -> float:
        try:
            logger.info("🔹 Running prediction...")
            # Convert input_data to dict for ModelPredictor
            prediction = self.model_predictor.predict(input_data.to_dict())
            logger.info(f"✅ Prediction result: {prediction}")
            return prediction
        except Exception as e:
            logger.error("❌ Error during prediction", exc_info=True)
            raise CustomException(e, sys)

    









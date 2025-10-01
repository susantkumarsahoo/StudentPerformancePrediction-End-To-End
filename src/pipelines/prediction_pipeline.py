import os
import sys
import joblib
import pandas as pd
from pandas import DataFrame
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import ModelDeploymentArtifact
from src.entity.model_config_entity import ModelDeploymentConfig
from models.predictor import ModelPredictor

logger = get_logger(__name__)

class StudentDataInput:
    """
    Represents the input features for student performance prediction.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_education: str,
                 lunch: str,
                 test_prep_course: str,
                 reading_score: int,
                 writing_score: int) -> None:
        try:
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_education = parental_education
            self.lunch = lunch
            self.test_prep_course = test_prep_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        except Exception as e:
            raise CustomException(e, sys)

    def to_dict(self) -> dict:
        """
        Convert the input features into a dictionary.
        """
        try:
            logger.info("Converting StudentDataInput to dictionary.")
            input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_prep_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            logger.info(f"Generated input dictionary: {input_dict}")
            return input_dict
        except Exception as e:
            raise CustomException(e, sys)

    def dataframe(self) -> DataFrame:
        """
        Convert the input features into a Pandas DataFrame.
        """
        try:
            logger.info("Converting StudentDataInput to DataFrame.")
            return DataFrame(self.to_dict())
        except Exception as e:
            raise CustomException(e, sys)

    def user_input(self) -> dict:
        """
        Return the raw input as a dictionary (without lists).
        """
        try:
            logger.info("Returning StudentDataInput as raw dictionary.")
            return {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_prep_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score,
            }
        except Exception as e:
            raise CustomException(e, sys)







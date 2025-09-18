import os
import pandas as pd
import numpy as np
from src.logging.logger import logger
from src.exceptions.exception import CustomException
from src.config.config import DATA_PATH

log = logger(__name__)

class DataValidation:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV"""
        if not os.path.exists(self.file_path):
            raise CustomException(f"File not found: {self.file_path}", sys)
        
        try:
            self.df = pd.read_csv(self.file_path)
            log.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            raise CustomException(str(e), sys)

    def validate_schema(self, expected_columns: list) -> bool:
        """Validate required columns exist in dataset"""
        missing_cols = [col for col in expected_columns if col not in self.df.columns]
        if missing_cols:
            log.error(f"Missing columns: {missing_cols}")
            return False
        log.info("Schema validation passed ✅")
        return True

    def check_missing_values(self) -> pd.Series:
        """Check missing values"""
        missing = self.df.isnull().sum()
        log.info(f"Missing values:\n{missing}")
        return missing

    def check_duplicates(self) -> int:
        """Check duplicate rows"""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            log.warning(f"Found {duplicates} duplicate rows")
        else:
            log.info("No duplicate rows found ✅")
        return duplicates

    def check_outliers(self, numeric_cols: list) -> dict:
        """Check for outliers using IQR method"""
        outlier_report = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)].shape[0]
            outlier_report[col] = outliers
            log.info(f"Outliers in {col}: {outliers}")
        return outlier_report

    def target_validation(self, target_col: str) -> bool:
        """Check target column for regression (must be numeric)"""
        if target_col not in self.df.columns:
            log.error(f"Target column '{target_col}' not found")
            return False
        if not np.issubdtype(self.df[target_col].dtype, np.number):
            log.error(f"Target column '{target_col}' must be numeric for regression")
            return False
        log.info(f"Target column '{target_col}' is valid ✅")
        return True

    def data_summary(self) -> pd.DataFrame:
        """Generate basic statistics"""
        summary = self.df.describe(include="all")
        log.info("Generated dataset summary statistics")
        return summary


if __name__ == "__main__":
    import sys
    try:
        # Example usage for Student Performance dataset
        validator = DataValidation(file_path=os.path.join(DATA_PATH, "student_performance.csv"))

        df = validator.load_data()

        expected_cols = ["gender", "parental_level_of_education", "lunch", 
                         "test_preparation_course", "math_score", "reading_score", "writing_score"]

        validator.validate_schema(expected_cols)
        validator.check_missing_values()
        validator.check_duplicates()
        validator.check_outliers(["math_score", "reading_score", "writing_score"])
        validator.target_validation("math_score")
        print(validator.data_summary())

    except CustomException as ce:
        log.error(ce)

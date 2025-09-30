import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import DataPreprocessingArtifact, FeatureEngineeringArtifact
from src.entity.config_entity import FeatureEngineeringConfig
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold



logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self, 
                 data_preprocessing_artifact: DataPreprocessingArtifact,
                 feature_engineering_config: FeatureEngineeringConfig):
        try:
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.feature_engineering_config = feature_engineering_config
            logger.info("Advanced FeatureEngineering initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_feature_report(self, df: pd.DataFrame) -> dict:
        """Generate advanced summary report of features."""
        report = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                report[col] = {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_values": int(df[col].nunique()),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurt())
                }
            else:
                report[col] = {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_values": int(df[col].nunique())
                }
        return report

#    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Generate polynomial interaction features for numeric columns."""
        """""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) <= 1:
            return df
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[num_cols])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols))
        df = pd.concat([df.drop(columns=num_cols), poly_df], axis=1)
        return df
        """

#    def apply_variance_threshold(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove low-variance features."""
        """
        selector = VarianceThreshold(threshold)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
        df_numeric = df[numeric_cols]
        df_selected = selector.fit_transform(df_numeric)
        selected_cols = numeric_cols[selector.get_support()]
        df_selected = pd.DataFrame(df_selected, columns=selected_cols)
        df = pd.concat([df.drop(columns=numeric_cols), df_selected], axis=1)
        return df 
        """
    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logger.info("Loading preprocessed train and test data.")
            train_df = pd.read_csv(self.data_preprocessing_artifact.preprocessing_train_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.preprocessing_test_path)

            logger.info("Generating polynomial features.")

            logger.info("Saving feature-engineered data.")
            train_df.to_csv(self.feature_engineering_config.feature_engineering_train_path, index=False)
            test_df.to_csv(self.feature_engineering_config.feature_engineering_test_path, index=False)

            feature_report = {
                "train_features": self.generate_feature_report(train_df),
                "test_features": self.generate_feature_report(test_df),
            }


            with open(self.feature_engineering_config.feature_engineering_report_path, "w") as f:
                json.dump(feature_report, f, indent=4)

            logger.info("Advanced feature engineering completed successfully.")

            return FeatureEngineeringArtifact(
                feature_engineering_train_path=self.feature_engineering_config.feature_engineering_train_path,
                feature_engineering_test_path=self.feature_engineering_config.feature_engineering_test_path,
                feature_engineering_report_path=self.feature_engineering_config.feature_engineering_report_path
            )

        except Exception as e:
            raise CustomException(e, sys)


















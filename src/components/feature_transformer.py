import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from src.constants.constants import *
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from src.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact, 
    DataPreprocessingArtifact, 
    FeatureEngineeringArtifact, 
    DataTransformationArtifact, 
    ModelTrainingArtifact, 
    ModelDeploymentArtifact, 
    ModelEvaluationArtifact
)

from src.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataPreprocessingConfig, 
    FeatureEngineeringConfig, 
    DataTransformationConfig
)

from src.entity.model_config_entity import (
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelDeploymentConfig
)


logger = get_logger(__name__)


class FeatureTransformer:
    def __init__(self, 
                 feature_engineering_artifact: FeatureEngineeringArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.feature_engineering_artifact = feature_engineering_artifact
            self.data_transformation_config = data_transformation_config
            logger.info("Advanced FeatureTransformer initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_transformation_report(self, df: pd.DataFrame) -> dict:
        """Advanced report: shape, nulls, top categories for categorical features."""
        report = {
            "shape": df.shape,
            "null_values": int(df.isnull().sum().sum()),
            "columns": []
        }
        for col in df.columns:
            info = {"column": col, "dtype": str(df[col].dtype)}
            if df[col].dtype == "object":
                info["top_categories"] = df[col].value_counts().head(5).to_dict()
            report["columns"].append(info)
        return report 

    def save_columns_report(self, num_features, cat_features, file_path):
        """
        Save numerical and categorical column names to a JSON file.
        """
        columns_report = {
            "numerical_features": list(num_features),
            "categorical_features": list(cat_features),
        }
        with open(file_path, "w") as f:
            json.dump(columns_report, f, indent=4)
        return columns_report

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Loading feature-engineered train and test data.")
            train_df = pd.read_csv(self.feature_engineering_artifact.feature_engineering_train_path)
            test_df = pd.read_csv(self.feature_engineering_artifact.feature_engineering_test_path)
            
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            X_df = combined_df.drop(columns=[TARGET_COLUMN])
            y_df = combined_df[TARGET_COLUMN]        

            num_features = X_df.select_dtypes(exclude="object").columns
            cat_features = X_df.select_dtypes(include="object").columns   

            # Define transformers
            nume_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot", oh_transformer, cat_features),
                    ("scaler", nume_transformer, num_features)
                ]
            )

            logger.info("Fitting transformer on training features only.")
            X_transformed = preprocessor.fit_transform(X_df)

            # Save transformed arrays
            np.save(self.data_transformation_config.transformer_train_path, X_transformed)
            np.save(self.data_transformation_config.transformer_test_path, y_df.to_numpy())

            # Save transformer object
            joblib.dump(preprocessor, self.data_transformation_config.transformer_object_path)

            # Generate transformation report using original DataFrames
            transformation_report = {
                "train": self.generate_transformation_report(X_df),
                "test": self.generate_transformation_report(X_df)
            }

            with open(self.data_transformation_config.transformer_report_path, "w") as f:
                json.dump(transformation_report, f, indent=4)
                
            # Save columns report
            self.save_columns_report(num_features, cat_features, self.data_transformation_config.transformer_report_path)

            logger.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                transformer_train_path=self.data_transformation_config.transformer_train_path,
                transformer_test_path=self.data_transformation_config.transformer_test_path,
                transformer_object_path=self.data_transformation_config.transformer_object_path,
                transformer_report_path=self.data_transformation_config.transformer_report_path
            )

        except Exception as e:
            raise CustomException(e, sys)
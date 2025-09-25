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
from src.constants.constants import *

logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self, data_preprocessing_artifact: DataPreprocessingArtifact,
                 feature_engineering_config: FeatureEngineeringConfig):
        try:
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.feature_engineering_config = feature_engineering_config

            # create dirs
            os.makedirs(
                os.path.join(self.feature_engineering_config.artifact_dir, FEATURE_ENGINEERING_DATA_DIR, TIMESTAMP),
                exist_ok=True
            )
            logger.info("FeatureEngineering initialized. Directories created successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_feature_engineering_report(self) -> str:
        try:
            report = {
                "numerical_columns": self.data_preprocessing_artifact.numerical_columns,
                "categorical_columns": self.data_preprocessing_artifact.categorical_columns,
                "feature_engineering_steps": [
                    "Created interaction feature: feature1_feature2_interaction",
                    "Encoded categorical variables using OneHotEncoder",
                    "Scaled numerical features using StandardScaler"
                ]
            }
            report_path = os.path.join(
                self.feature_engineering_config.artifact_dir,
                FEATURE_ENGINEERING_DATA_DIR,
                TIMESTAMP,
                FEATURE_ENGINEERING_REPORT_FILE_NAME
            )
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Feature engineering report saved at: {report_path}")
            return report_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logger.info("Starting feature engineering...")

            # Load preprocessed data
            train_df = pd.read_csv(self.data_preprocessing_artifact.preprocessing_train_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.preprocessing_test_path)
            logger.info(f"Preprocessed Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Example: create interaction feature if exists
            if 'feature1' in train_df.columns and 'feature2' in train_df.columns:
                train_df['feature1_feature2_interaction'] = train_df['feature1'] * train_df['feature2']
                test_df['feature1_feature2_interaction'] = test_df['feature1'] * test_df['feature2']
                logger.info("Created interaction feature: feature1_feature2_interaction")

            # Identify columns
            categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
                ]
            )

            # Fit and transform
            X_train = preprocessor.fit_transform(train_df.drop(columns=['target']))
            X_test = preprocessor.transform(test_df.drop(columns=['target']))

            # Convert to DataFrame
            feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)

            # Save feature engineered data
            train_file_path = os.path.join(self.feature_engineering_config.artifact_dir, FEATURE_ENGINEERING_DATA_DIR, TIMESTAMP, "feature_engineered_train.csv")
            test_file_path = os.path.join(self.feature_engineering_config.artifact_dir, FEATURE_ENGINEERING_DATA_DIR, TIMESTAMP, "feature_engineered_test.csv")
            X_train_df.to_csv(train_file_path, index=False)
            X_test_df.to_csv(test_file_path, index=False)

            logger.info("Feature engineering completed successfully.")

            # Generate report
            report_path = self.generate_feature_engineering_report()

            return FeatureEngineeringArtifact(
                feature_engineering_train_path=train_file_path,
                feature_engineering_test_path=test_file_path,
                feature_engineering_report_path=report_path
            )

        except Exception as e:
            raise CustomException(e, sys)
















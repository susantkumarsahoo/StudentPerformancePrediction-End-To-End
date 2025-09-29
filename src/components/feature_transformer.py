import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, DataPreprocessingArtifact, 
                                        FeatureEngineeringArtifact, DataTransformationArtifact, ModelTrainingArtifact, 
                                        ModelDeploymentArtifact, ModelEvaluationArtifact)

from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig, 
                                      FeatureEngineeringConfig, DataTransformationConfig)

from src.entity.model_config_entity import (ModelTrainingConfig,ModelEvaluationConfig,ModelDeploymentConfig)

from src.exceptions.exception import CustomException
from src.logging.logger import get_logger

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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Loading feature-engineered train and test data.")
            train_df = pd.read_csv(self.feature_engineering_artifact.feature_engineering_train_path)
            test_df = pd.read_csv(self.feature_engineering_artifact.feature_engineering_test_path)

            num_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_features = train_df.select_dtypes(include=[object]).columns.tolist()

            logger.info("Building advanced preprocessing pipeline.")
            num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, num_features),
                    ("cat", cat_transformer, cat_features)
                ]
            )

            logger.info("Fitting transformer on training data.")
            X_train = preprocessor.fit_transform(train_df)
            X_test = preprocessor.transform(test_df)

            # ✅ Already NumPy arrays, no need to call .to_numpy()
            X_train_arr = X_train
            X_test_arr = X_test

            # ✅ Save the NumPy Arrays
            np.save(self.data_transformation_config.transformer_train_path, X_train_arr)
            np.save(self.data_transformation_config.transformer_test_path, X_test_arr)

            logger.info("Saving transformed datasets and transformer object.")
            joblib.dump(preprocessor, self.data_transformation_config.transformer_object_path)

            # ⚠️ Report should be generated from original DataFrames, not arrays
            transformation_report = {
                "train": self.generate_transformation_report(train_df),
                "test": self.generate_transformation_report(test_df)
            }

            with open(self.data_transformation_config.transformer_report_path, "w") as f:
                json.dump(transformation_report, f, indent=4)

            logger.info("Advanced data transformation completed successfully.")

            return DataTransformationArtifact(
                transformer_train_path=self.data_transformation_config.transformer_train_path,
                transformer_test_path=self.data_transformation_config.transformer_test_path,
                transformer_object_path=self.data_transformation_config.transformer_object_path,
                transformer_report_path=self.data_transformation_config.transformer_report_path
            )

        except Exception as e:
            raise CustomException(e, sys)







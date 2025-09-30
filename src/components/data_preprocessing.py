import os
import json
import sys
import pandas as pd
import scipy as sp
import numpy as np
from datetime import datetime
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataPreprocessingArtifact
from src.entity.config_entity import DataValidationConfig, DataPreprocessingConfig


logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_preprocessing_config: DataPreprocessingConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_preprocessing_config = data_preprocessing_config
      
            logger.info("DataPreprocessing initialized. Directories created successfully.")
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_preprocessing(self) -> DataPreprocessingArtifact:
        try:

            if not self.data_validation_artifact.validation_status:
                error_msg = "Data validation failed. Cannot proceed to preprocessing."
                logger.error(error_msg)
                raise CustomException(error_msg, sys)

            logger.info("Data validation passed. Proceeding with preprocessing...")
            logger.info("Starting data preprocessing...")

            # Load train and test data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_path)
            logger.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

            # Basic preprocessing: Fill missing values with mean for numerical columns
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_cols:
                mean_value = train_df[col].mean()
                train_df[col] = train_df[col].fillna(mean_value)
                test_df[col] = test_df[col].fillna(mean_value)
                logger.info(f"Filled missing values in column '{col}' with mean value {mean_value}")

            # Save preprocessed data
            preprocessed_train_file = self.data_preprocessing_config.preprocessing_train_path
            preprocessed_test_file = self.data_preprocessing_config.preprocessing_test_path
        
            train_df.to_csv(preprocessed_train_file, index=False)
            test_df.to_csv(preprocessed_test_file, index=False)
            logger.info(f"Preprocessed train data saved at: {preprocessed_train_file}")
            logger.info(f"Preprocessed test data saved at: {preprocessed_test_file}")

            # Generate preprocessing report
            report = {
                "numerical_columns": numerical_cols,
                "missing_values_filled": {col: "mean" for col in numerical_cols}
            }
            report_file = self.data_preprocessing_config.preprocessing_report_path
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)

            logger.info(f"Preprocessing report saved at: {report_file}")

            # Return Artifact (make sure names match DataPreprocessingArtifact in artifact_entity.py)
            logger.info("Data preprocessing completed successfully.")

            return DataPreprocessingArtifact(
                preprocessing_train_path=preprocessed_train_file,
                preprocessing_test_path=preprocessed_test_file,
                preprocessing_report_path=report_file)

        except Exception as e:
            raise CustomException(e, sys)

        
        




            
            
            
            
            
            
            
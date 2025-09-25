import os
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataPreprocessingArtifact
from src.entity.config_entity import DataValidationConfig, DataPreprocessingConfig
from src.constants.constants import*


logger = get_logger(__name__)

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig,
                 data_preprocessing_config: DataPreprocessingConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_preprocessing_config = data_preprocessing_config
            self.data_validation_config = data_validation_config
            
            # Create preprocessing directory
            os.makedirs(os.path.join(self.data_preprocessing_config.artifact_dir,PREPROCESSING_DATA_DIR,TIMESTAMP), exist_ok=True)
        
            logger.info("initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)
        
             
def clean_data(self, df: pd.DataFrame,
               drop_duplicates: bool = True,
               fill_missing_num: str = "median",   # options: "mean", "median", "mode", "zero"
               fill_missing_cat: str = "mode",     # options: "mode", "constant"
               outlier_method: str = "iqr",        # options: "iqr", "zscore", None
               constant_fill_value: str = "missing") -> pd.DataFrame:
    """
    Simple Data Cleaning Function

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataset.
    drop_duplicates : bool
        Remove duplicate rows.
    fill_missing_num : str
        Strategy for numerical missing values.
    fill_missing_cat : str
        Strategy for categorical missing values.
    outlier_method : str
        Outlier removal method: "iqr", "zscore", or None.
    constant_fill_value : str
        Value to fill categorical missing when fill_missing_cat="constant".

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            if fill_missing_num == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_missing_num == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_missing_num == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_missing_num == "zero":
                df[col] = df[col].fillna(0)
        else:  # categorical
            if fill_missing_cat == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_missing_cat == "constant":
                df[col] = df[col].fillna(constant_fill_value)

    # Handle outliers
    if outlier_method == "iqr":
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif outlier_method == "zscore":
        from scipy.stats import zscore
        for col in df.select_dtypes(include=[np.number]).columns:
            df = df[(np.abs(zscore(df[col])) < 3)]

    return df

def initiate_data_preprocessing(self, DataFrame) -> DataPreprocessingArtifact:
    
    train_df = pd.read_csv(self.data_ingestion_artifact.train_data_path)
    test_df = pd.read_csv(self.data_ingestion_artifact.test_data_path)









        

            
            
            
            
            
            
            
            
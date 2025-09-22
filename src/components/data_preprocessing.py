import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

from constants.constants import *
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPreprocessingConfig:
    """Configuration class for data preprocessing"""
    artifacts_dir: str = ARTIFACTS_DIR
    preprocessing_dir: str = "data_preprocessing"
    preprocessor_file_name: str = "preprocessor.pkl"
    transformed_train_file_name: str = "transformed_train.csv"
    transformed_test_file_name: str = "transformed_test.csv"
    preprocessing_report_file_name: str = "preprocessing_report.json"
    feature_engineering_report_file_name: str = "feature_engineering_report.json"
    
    # Preprocessing parameters
    numerical_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    scaling_method: str = "standard"  # Options: 'standard', 'minmax', 'robust'
    encoding_method: str = "onehot"   # Options: 'onehot', 'label', 'target'
    handle_outliers: bool = True
    outlier_method: str = "iqr"       # Options: 'iqr', 'zscore'
    feature_selection: bool = False
    feature_selection_k: int = 10
    
    def __post_init__(self):
        self.preprocessing_dir_path = os.path.join(self.artifacts_dir, self.preprocessing_dir)
        self.preprocessor_file_path = os.path.join(self.preprocessing_dir_path, self.preprocessor_file_name)
        self.transformed_train_file_path = os.path.join(self.preprocessing_dir_path, self.transformed_train_file_name)
        self.transformed_test_file_path = os.path.join(self.preprocessing_dir_path, self.transformed_test_file_name)
        self.preprocessing_report_path = os.path.join(self.preprocessing_dir_path, self.preprocessing_report_file_name)
        self.feature_engineering_report_path = os.path.join(self.preprocessing_dir_path, self.feature_engineering_report_file_name)

@dataclass
class DataPreprocessingArtifact:
    """Artifact class for data preprocessing outputs"""
    preprocessor_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessing_report_path: str
    feature_engineering_report_path: str
    is_preprocessing_successful: bool

class DataPreprocessing:
    """Data preprocessing component for ML pipeline"""
    
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_preprocessing_config: DataPreprocessingConfig):
        
        try:
            logger.info("Initializing DataPreprocessing component")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_preprocessing_config = data_preprocessing_config
            
            # Create preprocessing directory
            os.makedirs(self.data_preprocessing_config.preprocessing_dir_path, exist_ok=True)
            
            # Initialize preprocessing components
            self.numerical_columns = []
            self.categorical_columns = []
            self.target_column = None
            self.preprocessor = None
            self.feature_names = []
            
        except Exception as e:
            logger.error(f"Error in DataPreprocessing initialization: {str(e)}")
            raise e
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test datasets"""
        try:
            logger.info("Loading train and test datasets")
            
            # Load train data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_path)
            logger.info(f"Train data shape: {train_df.shape}")
            
            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_path)
            logger.info(f"Test data shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e
    
    def identify_columns(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, List[str]]:
        """Identify numerical and categorical columns"""
        try:
            logger.info("Identifying column types")
            
            # Get all columns except target
            feature_columns = [col for col in df.columns if col != target_column] if target_column else df.columns
            
            # Identify numerical columns
            numerical_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            
            # Identify categorical columns
            categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove columns with too many unique values (likely IDs)
            categorical_cols = [col for col in categorical_cols 
                             if df[col].nunique() <= len(df) * 0.5]
            
            column_info = {
                'numerical': numerical_cols,
                'categorical': categorical_cols,
                'target': target_column
            }
            
            logger.info(f"Numerical columns: {len(numerical_cols)}")
            logger.info(f"Categorical columns: {len(categorical_cols)}")
            
            return column_info
            
        except Exception as e:
            logger.error(f"Error identifying columns: {str(e)}")
            raise e
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            logger.info("Handling missing values")
            
            missing_info = {}
            df_processed = df.copy()
            
            # Handle numerical columns
            if self.numerical_columns:
                for col in self.numerical_columns:
                    missing_count = df_processed[col].isnull().sum()
                    missing_percentage = (missing_count / len(df_processed)) * 100
                    
                    if missing_count > 0:
                        if missing_percentage > 50:
                            # Drop column if more than 50% missing
                            df_processed = df_processed.drop(col, axis=1)
                            self.numerical_columns.remove(col)
                            missing_info[col] = f"Dropped - {missing_percentage:.2f}% missing"
                        else:
                            # Impute missing values
                            if self.data_preprocessing_config.numerical_imputation_strategy == "median":
                                df_processed[col].fillna(df_processed[col].median(), inplace=True)
                            elif self.data_preprocessing_config.numerical_imputation_strategy == "mean":
                                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                            else:
                                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                            
                            missing_info[col] = f"Imputed {missing_count} values ({missing_percentage:.2f}%)"
            
            # Handle categorical columns
            if self.categorical_columns:
                for col in self.categorical_columns:
                    missing_count = df_processed[col].isnull().sum()
                    missing_percentage = (missing_count / len(df_processed)) * 100
                    
                    if missing_count > 0:
                        if missing_percentage > 50:
                            # Drop column if more than 50% missing
                            df_processed = df_processed.drop(col, axis=1)
                            self.categorical_columns.remove(col)
                            missing_info[col] = f"Dropped - {missing_percentage:.2f}% missing"
                        else:
                            # Impute with mode
                            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                            missing_info[col] = f"Imputed {missing_count} values ({missing_percentage:.2f}%)"
            
            logger.info("Missing value handling completed")
            return df_processed, missing_info
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise e
    
    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers in numerical columns"""
        try:
            if not self.data_preprocessing_config.handle_outliers:
                return df, {}
            
            logger.info("Handling outliers")
            
            outlier_info = {}
            df_processed = df.copy()
            
            for col in self.numerical_columns:
                if col in df_processed.columns:
                    if self.data_preprocessing_config.outlier_method == "iqr":
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers_count = len(df_processed[(df_processed[col] < lower_bound) | 
                                                        (df_processed[col] > upper_bound)])
                        
                        # Cap outliers
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                        
                    elif self.data_preprocessing_config.outlier_method == "zscore":
                        z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                        outliers_count = len(df_processed[z_scores > 3])
                        
                        # Cap outliers at 3 standard deviations
                        mean_val = df_processed[col].mean()
                        std_val = df_processed[col].std()
                        lower_bound = mean_val - 3 * std_val
                        upper_bound = mean_val + 3 * std_val
                        
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    outlier_info[col] = f"Handled {outliers_count} outliers"
            
            logger.info("Outlier handling completed")
            return df_processed, outlier_info
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise e
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline"""
        try:
            logger.info("Creating preprocessing pipeline")
            
            # Numerical pipeline
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.data_preprocessing_config.numerical_imputation_strategy))
            ])
            
            # Add scaler
            if self.data_preprocessing_config.scaling_method == "standard":
                numerical_pipeline.steps.append(('scaler', StandardScaler()))
            elif self.data_preprocessing_config.scaling_method == "minmax":
                numerical_pipeline.steps.append(('scaler', MinMaxScaler()))
            
            # Categorical pipeline
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.data_preprocessing_config.categorical_imputation_strategy, 
                                        fill_value='missing'))
            ])
            
            # Add encoder
            if self.data_preprocessing_config.encoding_method == "onehot":
                categorical_pipeline.steps.append(('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')))
            elif self.data_preprocessing_config.encoding_method == "label":
                categorical_pipeline.steps.append(('encoder', LabelEncoder()))
            
            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, self.numerical_columns),
                ('categorical', categorical_pipeline, self.categorical_columns)
            ])
            
            self.preprocessor = preprocessor
            logger.info("Preprocessing pipeline created successfully")
            
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise e
    
    def generate_feature_names(self, X_train: pd.DataFrame) -> List[str]:
        """Generate feature names after preprocessing"""
        try:
            feature_names = []
            
            # Add numerical feature names
            feature_names.extend(self.numerical_columns)
            
            # Add categorical feature names (for one-hot encoding)
            if self.data_preprocessing_config.encoding_method == "onehot":
                categorical_transformer = self.preprocessor.named_transformers_['categorical']
                if 'encoder' in [step[0] for step in categorical_transformer.steps]:
                    encoder = categorical_transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_feature_names = encoder.get_feature_names_out(self.categorical_columns)
                        feature_names.extend(cat_feature_names)
                    else:
                        # Fallback for older sklearn versions
                        for col in self.categorical_columns:
                            unique_vals = X_train[col].unique()
                            for val in unique_vals[1:]:  # Skip first category (dropped)
                                feature_names.append(f"{col}_{val}")
            else:
                feature_names.extend(self.categorical_columns)
            
            self.feature_names = feature_names
            return feature_names
            
        except Exception as e:
            logger.error(f"Error generating feature names: {str(e)}")
            return []
    
    def save_preprocessing_report(self, preprocessing_info: Dict) -> str:
        """Save preprocessing report"""
        try:
            logger.info("Saving preprocessing report")
            
            report = {
                "timestamp": TIMESTAMP,
                "preprocessing_config": {
                    "numerical_imputation_strategy": self.data_preprocessing_config.numerical_imputation_strategy,
                    "categorical_imputation_strategy": self.data_preprocessing_config.categorical_imputation_strategy,
                    "scaling_method": self.data_preprocessing_config.scaling_method,
                    "encoding_method": self.data_preprocessing_config.encoding_method,
                    "handle_outliers": self.data_preprocessing_config.handle_outliers,
                    "outlier_method": self.data_preprocessing_config.outlier_method
                },
                "column_info": {
                    "numerical_columns": self.numerical_columns,
                    "categorical_columns": self.categorical_columns,
                    "total_features": len(self.numerical_columns) + len(self.categorical_columns)
                },
                "preprocessing_steps": preprocessing_info,
                "final_feature_names": self.feature_names,
                "total_final_features": len(self.feature_names)
            }
            
            # Save report
            with open(self.data_preprocessing_config.preprocessing_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Preprocessing report saved successfully")
            return self.data_preprocessing_config.preprocessing_report_path
            
        except Exception as e:
            logger.error(f"Error saving preprocessing report: {str(e)}")
            raise e
    
    def initiate_data_preprocessing(self, target_column: str = None) -> DataPreprocessingArtifact:
        """Main method to initiate data preprocessing"""
        try:
            logger.info("Starting data preprocessing")
            
            # Check if data validation was successful
            if not self.data_validation_artifact.validation_status:
                logger.error("Data validation failed. Cannot proceed with preprocessing.")
                raise Exception("Data validation failed")
            
            # Load data
            train_df, test_df = self.load_data()
            
            # Identify columns
            column_info = self.identify_columns(train_df, target_column)
            self.numerical_columns = column_info['numerical']
            self.categorical_columns = column_info['categorical']
            self.target_column = target_column
            
            preprocessing_info = {
                "missing_values_info": {},
                "outlier_info": {},
                "feature_engineering_info": {}
            }
            
            # Handle missing values
            train_df_processed, missing_info = self.handle_missing_values(train_df)
            test_df_processed, _ = self.handle_missing_values(test_df)
            preprocessing_info["missing_values_info"] = missing_info
            
            # Handle outliers (only on training data)
            train_df_processed, outlier_info = self.handle_outliers(train_df_processed)
            preprocessing_info["outlier_info"] = outlier_info
            
            # Prepare features and target
            if target_column and target_column in train_df_processed.columns:
                X_train = train_df_processed.drop(target_column, axis=1)
                y_train = train_df_processed[target_column]
                X_test = test_df_processed.drop(target_column, axis=1) if target_column in test_df_processed.columns else test_df_processed
                y_test = test_df_processed[target_column] if target_column in test_df_processed.columns else None
            else:
                X_train = train_df_processed
                y_train = None
                X_test = test_df_processed
                y_test = None
            
            # Create and fit preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Generate feature names
            feature_names = self.generate_feature_names(X_train)
            
            # Convert to DataFrame with proper column names
            X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
            
            # Add target column back if available
            if y_train is not None:
                X_train_transformed_df[target_column] = y_train.reset_index(drop=True)
            if y_test is not None:
                X_test_transformed_df[target_column] = y_test.reset_index(drop=True)
            
            # Save transformed data
            X_train_transformed_df.to_csv(self.data_preprocessing_config.transformed_train_file_path, index=False)
            X_test_transformed_df.to_csv(self.data_preprocessing_config.transformed_test_file_path, index=False)
            
            # Save preprocessor
            joblib.dump(preprocessor, self.data_preprocessing_config.preprocessor_file_path)
            
            # Save preprocessing report
            self.save_preprocessing_report(preprocessing_info)
            
            # Create artifact
            data_preprocessing_artifact = DataPreprocessingArtifact(
                preprocessor_file_path=self.data_preprocessing_config.preprocessor_file_path,
                transformed_train_file_path=self.data_preprocessing_config.transformed_train_file_path,
                transformed_test_file_path=self.data_preprocessing_config.transformed_test_file_path,
                preprocessing_report_path=self.data_preprocessing_config.preprocessing_report_path,
                feature_engineering_report_path=self.data_preprocessing_config.feature_engineering_report_path,
                is_preprocessing_successful=True
            )
            
            logger.info("Data preprocessing completed successfully")
            return data_preprocessing_artifact
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise e

# Example usage
if __name__ == "__main__":
    try:
        # This is just for testing purposes
        # In actual implementation, these artifacts would come from previous pipeline stages
        
        # Create dummy artifacts for testing
        data_ingestion_artifact = DataIngestionArtifact(
            raw_data_path="artifacts/raw/raw_data.csv",
            processed_data_path="artifacts/processed/processed_data.csv", 
            train_data_path="artifacts/split/train.csv",
            test_data_path="artifacts/split/test.csv",
            metadata_path="artifacts/metadata.json",
            schema_path="artifacts/schema.json"
        )
        
        data_validation_artifact = DataValidationArtifact(
            validation_status=True,
            validation_report_path="artifacts/data_validation/validation_report.json",
            data_drift_report_path="artifacts/data_validation/data_drift_report.json",
            missing_columns_report_path="artifacts/data_validation/missing_columns.json",
            data_type_report_path="artifacts/data_validation/data_types.json",
            validation_status_path="artifacts/data_validation/validation_status.json"
        )
        
        # Initialize preprocessing config
        preprocessing_config = DataPreprocessingConfig()
        
        # Initialize preprocessing component
        data_preprocessing = DataPreprocessing(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
            data_preprocessing_config=preprocessing_config
        )
        
        # Run preprocessing (specify your target column name here)
        preprocessing_artifact = data_preprocessing.initiate_data_preprocessing(target_column="target")
        
        print("Data preprocessing completed successfully!")
        print(f"Preprocessor saved at: {preprocessing_artifact.preprocessor_file_path}")
        print(f"Transformed train data saved at: {preprocessing_artifact.transformed_train_file_path}")
        print(f"Transformed test data saved at: {preprocessing_artifact.transformed_test_file_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
import os
import json
import sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants.constants import (VALIDATION_DATA_DIR, DATA_DRIFT_REPORT_FILE_NAME,
                                        DATA_VALIDATION_STATUS_FILE, 
                                        MISSING_COLUMNS_FILE_NAME,
                                        MISSING_NUMERICAL_COLUMNS_FILE_NAME, 
                                        MISSING_CATEGORICAL_COLUMNS_FILE_NAME, 
                                        DATA_TYPE_FILE_NAME, 
                                        VALIDATION_REPORT_FILE_NAME,TIMESTAMP)


logger = get_logger(__name__)


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
            # Create validation directory
            os.makedirs(os.path.join(self.data_validation_config.artifact_dir, 
                                   VALIDATION_DATA_DIR,TIMESTAMP), exist_ok=True)
            
            logger.info("DataValidation initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame, schema: Dict) -> bool:
        """
        Validate if the dataframe has the expected number of columns based on schema
        """
        try:
            expected_columns = len(schema.get('properties', {}))
            actual_columns = len(df.columns)
            
            logger.info(f"Expected columns: {expected_columns}, Actual columns: {actual_columns}")
            
            if expected_columns == actual_columns:
                logger.info("Column count validation passed.")
                return True
            else:
                logger.warning(f"Column count validation failed. Expected: {expected_columns}, Got: {actual_columns}")
                return False
                
        except Exception as e:
            raise CustomException(e, sys)

    def validate_column_names(self, df: pd.DataFrame, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate if all expected columns are present in the dataframe
        """
        try:
            expected_columns = set(schema.get('properties', {}).keys())
            actual_columns = set(df.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            logger.info(f"Expected columns: {expected_columns}")
            logger.info(f"Actual columns: {actual_columns}")
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                
            if extra_columns:
                logger.info(f"Extra columns found: {extra_columns}")
            
            validation_status = len(missing_columns) == 0
            return validation_status, list(missing_columns)
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_types(self, df: pd.DataFrame, schema: Dict) -> Tuple[bool, Dict]:
        """
        Validate data types of columns against schema
        """
        try:
            type_validation_report = {}
            type_mapping = {
                'number': ['int64', 'float64', 'int32', 'float32'],
                'string': ['object', 'string'],
                'integer': ['int64', 'int32'],
                'boolean': ['bool']
            }
            
            properties = schema.get('properties', {})
            validation_passed = True
            
            for column, expected_type_info in properties.items():
                if column in df.columns:
                    expected_type = expected_type_info.get('type', 'string')
                    actual_type = str(df[column].dtype)
                    
                    expected_dtypes = type_mapping.get(expected_type, ['object'])
                    type_match = actual_type in expected_dtypes
                    
                    type_validation_report[column] = {
                        'expected_type': expected_type,
                        'actual_type': actual_type,
                        'validation_passed': type_match
                    }
                    
                    if not type_match:
                        validation_passed = False
                        logger.warning(f"Data type mismatch for {column}: expected {expected_type}, got {actual_type}")
                    else:
                        logger.info(f"Data type validation passed for {column}")
                        
            return validation_passed, type_validation_report
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate missing values in the dataset
        """
        try:
            missing_report = {}
            validation_passed = True
            
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                missing_percentage = missing_count / len(df)
                
                missing_report[column] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_percentage),
                    'threshold_exceeded': missing_percentage > self.data_validation_config.max_missing_threshold
                }
                
                if missing_percentage > self.data_validation_config.max_missing_threshold:
                    validation_passed = False
                    logger.warning(f"Column {column} has {missing_percentage:.2%} missing values, exceeding threshold")
                else:
                    logger.info(f"Missing value validation passed for {column}: {missing_percentage:.2%}")
                    
            return validation_passed, missing_report
            
        except Exception as e:
            raise CustomException(e, sys)

    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Detect data drift between reference and current datasets using KS test
        """
        try:
            drift_report = {}
            drift_detected = False
            
            # Get numerical columns
            numerical_columns = reference_df.select_dtypes(include=[np.number]).columns
            
            for column in numerical_columns:
                if column in current_df.columns:
                    # Remove null values for comparison
                    ref_data = reference_df[column].dropna()
                    curr_data = current_df[column].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        # Perform KS test
                        statistic, p_value = ks_2samp(ref_data, curr_data)
                        
                        is_drift = p_value < self.data_validation_config.drift_threshold
                        
                        drift_report[column] = {
                            'ks_statistic': float(statistic),
                            'p_value': float(p_value),
                            'drift_detected': is_drift,
                            'threshold': self.data_validation_config.drift_threshold
                        }
                        
                        if is_drift:
                            drift_detected = True
                            logger.warning(f"Data drift detected in column {column}: p_value={p_value:.6f}")
                        else:
                            logger.info(f"No data drift detected in column {column}: p_value={p_value:.6f}")
                    else:
                        drift_report[column] = {
                            'ks_statistic': None,
                            'p_value': None,
                            'drift_detected': None,
                            'note': 'Insufficient data for comparison'
                        }
                        
            return not drift_detected, drift_report
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_numerical_columns(self, df: pd.DataFrame, schema: Dict) -> Tuple[bool, Dict]:
        """
        Validate numerical columns distribution and statistics
        """
        try:
            numerical_report = {}
            validation_passed = True
            
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numerical_columns:
                column_data = df[column].dropna()
                
                if len(column_data) > 0:
                    stats = {
                        'count': int(len(column_data)),
                        'mean': float(column_data.mean()),
                        'std': float(column_data.std()),
                        'min': float(column_data.min()),
                        'max': float(column_data.max()),
                        'q25': float(column_data.quantile(0.25)),
                        'q50': float(column_data.quantile(0.50)),
                        'q75': float(column_data.quantile(0.75)),
                        'has_negative': bool((column_data < 0).any()),
                        'has_zero': bool((column_data == 0).any()),
                        'has_outliers': self._detect_outliers(column_data)
                    }
                    
                    numerical_report[column] = stats
                    logger.info(f"Numerical validation completed for {column}")
                else:
                    numerical_report[column] = {'note': 'Column has no valid numerical data'}
                    validation_passed = False
                    
            return validation_passed, numerical_report
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_categorical_columns(self, df: pd.DataFrame, schema: Dict) -> Tuple[bool, Dict]:
        """
        Validate categorical columns
        """
        try:
            categorical_report = {}
            validation_passed = True
            
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            
            for column in categorical_columns:
                column_data = df[column].dropna()
                
                if len(column_data) > 0:
                    unique_values = column_data.unique()
                    value_counts = column_data.value_counts()
                    
                    stats = {
                        'unique_count': int(len(unique_values)),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'unique_values': [str(val) for val in unique_values[:50]],  # Limit to first 50
                        'value_distribution': {str(k): int(v) for k, v in value_counts.head(20).items()}
                    }
                    
                    categorical_report[column] = stats
                    logger.info(f"Categorical validation completed for {column}")
                else:
                    categorical_report[column] = {'note': 'Column has no valid categorical data'}
                    
            return validation_passed, categorical_report
            
        except Exception as e:
            raise CustomException(e, sys)

    def _detect_outliers(self, data: pd.Series) -> bool:
        """
        Detect outliers using IQR method
        """
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            return len(outliers) > 0
            
        except Exception as e:
            return False

    def save_validation_reports(self, validation_results: Dict) -> Dict[str, str]:
        """
        Save all validation reports to files
        """
        try:
            report_paths = {}
            validation_dir = os.path.join(self.data_validation_config.artifact_dir, VALIDATION_DATA_DIR)
            
            # Save main validation report
            validation_report_path = os.path.join(validation_dir, VALIDATION_REPORT_FILE_NAME)
            with open(validation_report_path, 'w') as f:
                json.dump(validation_results, f, indent=4, default=str)
            report_paths['validation_report'] = validation_report_path
            
            # Save data drift report
            drift_report_path = os.path.join(validation_dir, DATA_DRIFT_REPORT_FILE_NAME)
            with open(drift_report_path, 'w') as f:
                json.dump(validation_results.get('data_drift_report', {}), f, indent=4, default=str)
            report_paths['data_drift_report'] = drift_report_path
            
            # Save missing columns report
            missing_columns_path = os.path.join(validation_dir, MISSING_COLUMNS_FILE_NAME)
            with open(missing_columns_path, 'w') as f:
                json.dump(validation_results.get('missing_columns', []), f, indent=4)
            report_paths['missing_columns'] = missing_columns_path
            
            # Save data type report
            data_type_path = os.path.join(validation_dir, DATA_TYPE_FILE_NAME)
            with open(data_type_path, 'w') as f:
                json.dump(validation_results.get('data_type_report', {}), f, indent=4, default=str)
            report_paths['data_type'] = data_type_path
            
            # Save validation status
            status_path = os.path.join(validation_dir, DATA_VALIDATION_STATUS_FILE)
            status_data = {
                'validation_status': validation_results.get('overall_validation_status', False),
                'timestamp': datetime.now().isoformat(),
                'summary': validation_results.get('validation_summary', {})
            }
            with open(status_path, 'w') as f:
                json.dump(status_data, f, indent=4)
            report_paths['validation_status'] = status_path
            
            logger.info("All validation reports saved successfully.")
            return report_paths
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiate the complete data validation process
        """
        try:
            logger.info("Starting data validation process...")
            
            # Load data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_path)
            
            # Load schema
            with open(self.data_ingestion_artifact.schema_path, 'r') as f:
                schema = json.load(f)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'train_data_shape': train_df.shape,
                'test_data_shape': test_df.shape
            }
            
            # 1. Validate column count
            column_count_valid = self.validate_number_of_columns(train_df, schema)
            validation_results['column_count_validation'] = column_count_valid
            
            # 2. Validate column names
            column_names_valid, missing_columns = self.validate_column_names(train_df, schema)
            validation_results['column_names_validation'] = column_names_valid
            validation_results['missing_columns'] = missing_columns
            
            # 3. Validate data types
            data_types_valid, data_type_report = self.validate_data_types(train_df, schema)
            validation_results['data_types_validation'] = data_types_valid
            validation_results['data_type_report'] = data_type_report
            
            # 4. Validate missing values
            missing_values_valid, missing_report = self.validate_missing_values(train_df)
            validation_results['missing_values_validation'] = missing_values_valid
            validation_results['missing_values_report'] = missing_report
            
            # 5. Detect data drift between train and test
            drift_valid, drift_report = self.detect_data_drift(train_df, test_df)
            validation_results['data_drift_validation'] = drift_valid
            validation_results['data_drift_report'] = drift_report
            
            # 6. Validate numerical columns
            numerical_valid, numerical_report = self.validate_numerical_columns(train_df, schema)
            validation_results['numerical_validation'] = numerical_valid
            validation_results['numerical_report'] = numerical_report
            
            # 7. Validate categorical columns
            categorical_valid, categorical_report = self.validate_categorical_columns(train_df, schema)
            validation_results['categorical_validation'] = categorical_valid
            validation_results['categorical_report'] = categorical_report
            
            # Overall validation status
            overall_status = all([
                column_count_valid,
                column_names_valid,
                data_types_valid,
                missing_values_valid,
                drift_valid,
                numerical_valid,
                categorical_valid
            ])
            
            validation_results['overall_validation_status'] = overall_status
            validation_results['validation_summary'] = {
                'total_validations': 7,
                'passed_validations': sum([
                    column_count_valid,
                    column_names_valid, 
                    data_types_valid,
                    missing_values_valid,
                    drift_valid,
                    numerical_valid,
                    categorical_valid
                ]),
                'validation_score': sum([
                    column_count_valid,
                    column_names_valid,
                    data_types_valid, 
                    missing_values_valid,
                    drift_valid,
                    numerical_valid,
                    categorical_valid
                ]) / 7
            }
            
            # Save all reports
            report_paths = self.save_validation_reports(validation_results)
            
            logger.info(f"Data validation completed. Overall status: {overall_status}")
            logger.info(f"Validation score: {validation_results['validation_summary']['validation_score']:.2%}")
            
            return DataValidationArtifact(
                validation_status=overall_status,
                validation_report_path=report_paths['validation_report'],
                data_drift_report_path=report_paths['data_drift_report'],
                missing_columns_report_path=report_paths['missing_columns'],
                data_type_report_path=report_paths['data_type'],
                validation_status_path=report_paths['validation_status']
            )
            
        except Exception as e:
            logger.error("Error occurred during data validation.", exc_info=True)
            raise CustomException(e, sys)
        


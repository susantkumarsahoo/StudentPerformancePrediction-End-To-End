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
from src.constants.constants import*


logger = get_logger(__name__)


def validate_number_of_columns(df: pd.DataFrame, schema: Dict) -> bool:
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


def validate_column_names(df: pd.DataFrame, schema: Dict) -> Tuple[bool, List[str]]:
    """
    Validate if all expected columns are present in the dataframe
    """
    try:
        expected_columns = set(schema.get('properties', {}).keys())
        actual_columns = set(df.columns)
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        if extra_columns:
            logger.info(f"Extra columns found: {extra_columns}")

        validation_status = len(missing_columns) == 0
        return validation_status, list(missing_columns)
    except Exception as e:
        raise CustomException(e, sys)


def validate_data_types(df: pd.DataFrame, schema: Dict) -> Tuple[bool, Dict]:
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


def validate_missing_values(df: pd.DataFrame, max_missing_threshold: float) -> Tuple[bool, Dict]:
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
                'threshold_exceeded': missing_percentage > max_missing_threshold
            }

            if missing_percentage > max_missing_threshold:
                validation_passed = False
                logger.warning(f"Column {column} has {missing_percentage:.2%} missing values, exceeding threshold")
            else:
                logger.info(f"Missing value validation passed for {column}: {missing_percentage:.2%}")

        return validation_passed, missing_report
    except Exception as e:
        raise CustomException(e, sys)


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, drift_threshold: float) -> Tuple[bool, Dict]:
    """
    Detect data drift between reference and current datasets using KS test
    """
    try:
        drift_report = {}
        drift_detected = False
        numerical_columns = reference_df.select_dtypes(include=[np.number]).columns

        for column in numerical_columns:
            if column in current_df.columns:
                ref_data = reference_df[column].dropna()
                curr_data = current_df[column].dropna()

                if len(ref_data) > 0 and len(curr_data) > 0:
                    statistic, p_value = ks_2samp(ref_data, curr_data)
                    is_drift = p_value < drift_threshold
                    drift_report[column] = {
                        'ks_statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift_detected': is_drift,
                        'threshold': drift_threshold
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


def validate_numerical_columns(df: pd.DataFrame) -> Tuple[bool, Dict]:
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
                    'has_outliers': detect_outliers(column_data)
                }
                numerical_report[column] = stats
                logger.info(f"Numerical validation completed for {column}")
            else:
                numerical_report[column] = {'note': 'Column has no valid numerical data'}
                validation_passed = False

        return validation_passed, numerical_report
    except Exception as e:
        raise CustomException(e, sys)


def validate_categorical_columns(df: pd.DataFrame) -> Tuple[bool, Dict]:
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
                    'unique_values': [str(val) for val in unique_values[:50]],  # Limit first 50
                    'value_distribution': {str(k): int(v) for k, v in value_counts.head(20).items()}
                }
                categorical_report[column] = stats
                logger.info(f"Categorical validation completed for {column}")
            else:
                categorical_report[column] = {'note': 'Column has no valid categorical data'}

        return validation_passed, categorical_report
    except Exception as e:
        raise CustomException(e, sys)


def detect_outliers(data: pd.Series) -> bool:
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
        












# StudentPerformancePrediction-End-To-End
Student performance prediction uses machine learning and analytics to forecast a student's academic success by analyzing historical and behavioral data like grades, attendance, and engagement.

1 constant.py
2 artifact_entity.py
3 config_entity.py
4 data_ingesation.py
5 data_validation.py
6 data_preprocessing.py  
6 taining_pipeline.py
7 check demo.py
































data_validation.py

Key Features:
1. Updated Constants

Added validation-specific constants for directories, file names, and thresholds
Maintained consistency with your existing naming conventions

2. Enhanced Entity Classes

Added DataValidationArtifact to track validation outputs
Added DataValidationConfig for configuration management

3. Comprehensive Data Validation Class
The DataValidation class includes:
Core Validations:

Column Count Validation: Ensures expected number of columns
Column Name Validation: Checks for missing/extra columns
Data Type Validation: Validates against schema definitions
Missing Value Validation: Checks missing data thresholds
Data Drift Detection: Uses KS test to detect distribution changes
Numerical Column Validation: Statistical analysis and outlier detection
Categorical Column Validation: Unique values and distribution analysis

Key Methods:

validate_number_of_columns(): Column count verification
validate_column_names(): Column presence validation
validate_data_types(): Type consistency checking
validate_missing_values(): Missing data analysis
detect_data_drift(): Statistical drift detection
validate_numerical_columns(): Numerical data validation
validate_categorical_columns(): Categorical data validation
save_validation_reports(): Comprehensive report generation

Features:

Modular Design: Each validation is separate and reusable
Comprehensive Reporting: Detailed JSON reports for all validations
Statistical Analysis: KS test for drift detection, IQR for outliers
Error Handling: Consistent exception handling using your custom exception class
Logging: Detailed logging throughout the process
Configurable Thresholds: Customizable validation thresholds







# StudentPerformancePrediction-End-To-End
Student performance prediction uses machine learning and analytics to forecast a student's academic success by analyzing historical and behavioral data like grades, attendance, and engagement.

constant.py
artifect_entity.py
config_entity.py
data_ingesation.py
data_validation.py
training_pipeline.py
check demo.py



# Data Ingestion Module

This module handles loading, processing, and splitting structured datasets for machine learning pipelines.

---

## **Key Steps**

1. **Directory Management** – Automatically creates required directories for raw, processed, and split data.  
2. **Raw Data Loading** – Reads the dataset from the source path and logs its shape.  
3. **Data Cleaning** – Performs basic preprocessing such as dropping or imputing missing values.  
4. **Data Saving** – Saves raw and processed datasets as CSV artifacts for traceability.  
5. **Train/Test Split** – Splits data into train and test sets with configurable size and random state.  
6. **Metadata Logging** – Stores dataset shapes, timestamps, and other metadata for reproducibility.  
7. **Schema Generation** – Generates JSON schema with column types, unique counts, and example values.  
8. **Logging & Error Handling** – Detailed logs and robust exception handling for reliable pipelines.  

---

## **Benefits**
- Ensures **consistent, reproducible datasets** for ML workflows.  
- Provides **traceable raw and processed data artifacts**.  
- Supports **robust, modular, and production-ready pipeline integration**.


# Data Validation Module

This module validates structured regression datasets to ensure data quality and reliability before modeling.  

---

## **Key Validation Steps**

1. **Schema Validation** – Check column names, types, and order.  
2. **Missing Value Checks** – Verify missing data and apply thresholds.  
3. **Outlier Detection** – Identify extreme numerical values using IQR or Z-score.  
4. **Categorical Validation** – Ensure allowed values and uniqueness.  
5. **Data Drift Detection** – Detect distribution changes using KS test.  
6. **Column Count & Type Validation** – Confirm column number and data types.  
7. **Statistical Summaries** – Compute mean, median, min, max, std, quantiles.  
8. **Comprehensive Reporting** – Generate JSON/HTML validation reports for traceability.  

---

## **Benefits**
- Ensures **clean, consistent data** for regression modeling.  
- Provides **transparent, auditable validation reports**.  
- Detects **data issues early** to improve model performance.




# Data Transformation Module

This module prepares structured datasets for machine learning models by applying feature engineering, scaling, and encoding.

---

## **Key Steps**

1. **Feature Engineering** – Create new features or transform existing ones for better model performance.  
2. **Numerical Transformations** – Apply scaling or normalization to numerical columns.  
3. **Categorical Encoding** – Convert categorical variables using one-hot, label encoding, or ordinal encoding.  
4. **Target Transformation** – Optionally transform target variables (e.g., log transformation) for regression tasks.  
5. **Train/Test Transformation** – Apply transformations consistently to both train and test sets.  
6. **Pipeline Integration** – Use **modular and reusable transformers** for seamless ML pipeline integration.  
7. **Artifact Saving** – Save transformed datasets and transformation objects for reproducibility.  
8. **Logging & Error Handling** – Detailed logging and exception handling throughout the transformation process.  

---

## **Benefits**
- Ensures **model-ready datasets** with consistent transformations.  
- Provides **reproducible transformations** for both training and inference.  
- Supports **modular, production-ready ML pipelines**.








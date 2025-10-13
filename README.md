# 🎓 Student Performance Prediction - End-to-End ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive end-to-end machine learning project that predicts student math scores based on various demographic and academic factors.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Dataset Description](#-dataset-description)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline Stages](#-pipeline-stages)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **production-ready machine learning pipeline** that predicts student mathematics performance scores. Built with industry-standard ML practices, it demonstrates the complete lifecycle from data ingestion to model deployment through a RESTful API.

### Key Highlights

- ✅ **87.1% R² Score** on test data
- ✅ Comprehensive data validation and drift detection
- ✅ Advanced feature engineering with polynomial features
- ✅ Automated pipeline execution
- ✅ REST API for real-time predictions
- ✅ Production-grade logging and monitoring

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Automated Pipeline** | End-to-end workflow from data ingestion to deployment |
| **Data Validation** | Schema validation, drift detection, quality checks |
| **Feature Engineering** | Polynomial features and intelligent selection |
| **Model Training** | Lasso regression with hyperparameter tuning |
| **Performance Monitoring** | Real-time metrics (R², MSE, MAE, RMSE) |
| **REST API** | Flask-based API for production predictions |
| **Artifact Management** | Structured storage of all pipeline outputs |
| **Comprehensive Logging** | Detailed logs for debugging and monitoring |

---

## 🏗️ Project Architecture

```mermaid
graph LR
    A[Data Ingestion] --> B[Data Validation]
    B --> C[Data Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Data Transformation]
    E --> F[Model Training]
    F --> G[Model Evaluation]
    G --> H[Model Deployment]
    H --> I[Prediction API]
    
    style A fill:#e1f5ff
    style B fill:#ffe1e1
    style C fill:#fff4e1
    style D fill:#e1ffe1
    style E fill:#f0e1ff
    style F fill:#ffe1f5
    style G fill:#e1ffff
    style H fill:#ffe1e1
    style I fill:#f5e1ff
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Data Ingestion      → Load & Split Dataset (80/20)           │
│ 2. Data Validation     → Schema, Drift & Quality Checks         │
│ 3. Data Preprocessing  → Clean & Handle Missing Values          │
│ 4. Feature Engineering → Create Polynomial Features             │
│ 5. Data Transformation → Scale & Encode Features                │
│ 6. Model Training      → Train Lasso Regression                 │
│ 7. Model Evaluation    → Calculate Performance Metrics          │
│ 8. Model Deployment    → Save to Production Artifacts           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│ Input → Preprocess → Transform → Predict → Output               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Description

### Overview

- **Total Records:** 1,000 students
- **Features:** 8 columns (7 input + 1 target)
- **Training Set:** 800 samples (80%)
- **Test Set:** 200 samples (20%)

### Features

#### Input Features

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `gender` | Categorical | Student's gender | male, female |
| `race_ethnicity` | Categorical | Ethnic group | Group A, B, C, D, E |
| `parental_level_of_education` | Categorical | Parent's highest education | some high school, high school, some college, associate's degree, bachelor's degree, master's degree |
| `lunch` | Categorical | Lunch type | standard, free/reduced |
| `test_preparation_course` | Categorical | Test prep status | none, completed |
| `reading_score` | Numerical | Reading test score | 0-100 |
| `writing_score` | Numerical | Writing test score | 0-100 |

#### Target Variable

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `math_score` | Numerical | Mathematics test score | 0-100 |

### Data Quality

- ✅ **Zero Missing Values:** Complete dataset
- ✅ **No Data Drift Detected:** Stable feature distributions
- ✅ **100% Validation Score:** All quality checks passed

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/StudentPerformancePrediction-End-To-End.git
cd StudentPerformancePrediction-End-To-End
```

#### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv myenv
myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Dataset

Ensure the dataset exists at:
```
data/raw/student.csv
```

---

## 💻 Quick Start

### Running the Complete Pipeline

```bash
python main.py
```

**What happens:**
1. ✅ Dataset validation
2. ✅ Training pipeline execution
3. ✅ Model training and evaluation
4. ✅ Model deployment
5. ✅ Flask API server starts

### Expected Output

```
[2025-10-14 01:35:48,996] INFO - ✅ Dataset found
[2025-10-14 01:35:49,099] INFO - ✅ Data Validation completed
[2025-10-14 01:35:49,127] INFO - ✅ Data Preprocessing completed
[2025-10-14 01:35:49,155] INFO - ✅ Feature Engineering completed
[2025-10-14 01:35:49,196] INFO - ✅ Data Transformation completed
[2025-10-14 01:35:49,212] INFO - ✅ Model Training completed
[2025-10-14 01:35:49,222] INFO - ✅ Model Evaluation completed
[2025-10-14 01:35:49,257] INFO - ✅ Model Deployment completed
[2025-10-14 01:35:49,258] INFO - 🎉 Deployment Pipeline executed successfully
 * Serving Flask app 'main'
 * Debug mode: on
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion

**Purpose:** Load and prepare raw data

**Actions:**
- Load dataset from CSV
- Perform 80-20 train-test split
- Save processed data to artifacts
- Generate metadata and schema files

**Output:**
```
artifacts/data_ingestion/
├── raw/raw_data.csv
├── processed/processed_data.csv
├── split/train.csv
├── split/test.csv
├── metadata.json
└── schema.json
```

---

### Stage 2: Data Validation

**Purpose:** Ensure data quality and consistency

**Validations Performed:**

| Check | Description | Status |
|-------|-------------|--------|
| Column Count | Verify expected columns (8) | ✅ Passed |
| Column Names | Check all required columns exist | ✅ Passed |
| Data Types | Validate categorical and numerical types | ✅ Passed |
| Missing Values | Identify null/missing data | ✅ 0% missing |
| Data Drift | Detect distribution changes | ✅ No drift |

**Statistical Tests:**
- Math Score: p-value = 0.706065 (No drift)
- Reading Score: p-value = 0.652868 (No drift)
- Writing Score: p-value = 0.547307 (No drift)

**Output:**
- Validation Score: **100%**
- Validation Reports: JSON format

---

### Stage 3: Data Preprocessing

**Purpose:** Clean and prepare data for modeling

**Operations:**
- Handle missing values (mean imputation)
  - Math Score: 66.50
  - Reading Score: 69.56
  - Writing Score: 68.46
- Data standardization
- Quality checks

**Output:**
```
artifacts/data_preprocessing/
├── train_preprocessed.csv
├── test_preprocessed.csv
└── preprocessing_report.json
```

---

### Stage 4: Feature Engineering

**Purpose:** Create enhanced features for better predictions

**Techniques:**
- Polynomial feature generation (degree 2)
- Feature interaction terms
- Feature selection based on importance

**Output:**
```
artifacts/feature_engineering/
├── train_engineered.csv
├── test_engineered.csv
└── feature_report.json
```

---

### Stage 5: Data Transformation

**Purpose:** Scale and encode features

**Transformations:**
- Numerical feature scaling (StandardScaler)
- Categorical encoding (OneHotEncoder)
- Fitted on training data only (prevents leakage)

**Output:**
```
artifacts/data_transformation/
├── train_transformed.npy
├── test_transformed.npy
└── transformer.pkl
```

---

### Stage 6: Model Training

**Purpose:** Train predictive model

**Algorithm:** Lasso Regression
- **Regularization:** L1 penalty
- **Hyperparameter Tuning:** Cross-validation
- **Training Strategy:** Fit on training set only

**Training Metrics:**
- MSE: 29.15
- R² Score: 0.877

**Output:**
```
artifacts/model_training/
├── best_model.pkl
└── model_report.json
```

---

### Stage 7: Model Evaluation

**Purpose:** Assess model performance on unseen data

**Test Metrics:**
- **R² Score:** 0.871
- **MAE:** 4.05 points
- **MSE:** 25.60
- **RMSE:** 5.06 points

**Output:**
```
artifacts/model_evaluation/
└── model_evaluation_report.json
```

---

### Stage 8: Model Deployment

**Purpose:** Deploy model to production

**Actions:**
- Save model to deployment artifacts
- Save preprocessor pipeline
- Generate deployment report
- Initialize prediction service

**Output:**
```
deployment_artifacts/
├── best_model.pkl
├── preprocessor.pkl
└── model_deployment_report.json
```

---

## 📈 Model Performance

### Performance Summary

<div align="center">

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **R² Score** | 0.877 | **0.871** |
| **MSE** | 29.15 | 25.60 |
| **MAE** | - | 4.05 |
| **RMSE** | - | 5.06 |

</div>

### Model Analysis

✅ **Strong Generalization:** Minimal gap between training (0.877) and test (0.871) R² scores

✅ **Low Overfitting:** Model generalizes well to unseen data

✅ **Practical Accuracy:** Average prediction error of ~4 points (MAE)

✅ **Reliable Predictions:** RMSE of 5.06 indicates consistent performance

### Performance Interpretation

- **87.1%** of variance in math scores is explained by the model
- On average, predictions are off by **±4-5 points**
- Model performs equally well on training and test data
- No signs of overfitting or underfitting

---

## 🌐 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### `POST /predict`

Predict student math score based on input features.

**Request**

```http
POST /predict
Content-Type: application/json
```

**Request Body:**

```json
{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "none",
  "reading_score": 72,
  "writing_score": 74
}
```

**Response:**

```json
{
  "prediction": 74.87,
  "status": "success",
  "timestamp": "2025-10-14T01:36:06.699"
}
```

### Example Usage

#### Python (requests)

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Input data
data = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 72,
    "writing_score": 74
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Math Score: {result['prediction']:.2f}")
```

#### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor'\''s degree",
    "lunch": "standard",
    "test_preparation_course": "none",
    "reading_score": 72,
    "writing_score": 74
  }'
```

#### JavaScript (fetch)

```javascript
const url = 'http://localhost:5000/predict';

const data = {
  gender: 'female',
  race_ethnicity: 'group B',
  parental_level_of_education: "bachelor's degree",
  lunch: 'standard',
  test_preparation_course: 'completed',
  reading_score: 72,
  writing_score: 74
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
})
.then(response => response.json())
.then(result => console.log('Predicted Score:', result.prediction))
.catch(error => console.error('Error:', error));
```

### Response Codes

| Code | Description |
|------|-------------|
| 200 | Successful prediction |
| 400 | Bad request (invalid input) |
| 500 | Internal server error |

---

## 📁 Project Structure

```
StudentPerformancePrediction-End-To-End/
│
├── 📂 data/
│   └── 📂 raw/
│       └── 📄 student.csv                    # Raw dataset
│
├── 📂 artifacts/                              # Pipeline artifacts
│   ├── 📂 data_ingestion/
│   │   ├── 📂 raw/
│   │   ├── 📂 processed/
│   │   ├── 📂 split/
│   │   ├── 📄 metadata.json
│   │   └── 📄 schema.json
│   │
│   ├── 📂 data_preprocessing/
│   │   ├── 📄 train_preprocessed.csv
│   │   ├── 📄 test_preprocessed.csv
│   │   └── 📄 preprocessing_report.json
│   │
│   ├── 📂 feature_engineering/
│   │   ├── 📄 train_engineered.csv
│   │   └── 📄 test_engineered.csv
│   │
│   ├── 📂 data_transformation/
│   │   ├── 📄 train_transformed.npy
│   │   ├── 📄 test_transformed.npy
│   │   └── 📄 transformer.pkl
│   │
│   ├── 📂 model_training/
│   │   ├── 📄 best_model.pkl
│   │   └── 📄 model_report.json
│   │
│   └── 📂 model_evaluation/
│       └── 📄 model_evaluation_report.json
│
├── 📂 deployment_artifacts/                   # Production artifacts
│   ├── 📄 best_model.pkl
│   ├── 📄 preprocessor.pkl
│   └── 📄 model_deployment_report.json
│
├── 📂 src/                                    # Source code
│   ├── 📂 components/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_ingestion.py              # Data loading & splitting
│   │   ├── 📄 data_validation.py             # Data quality checks
│   │   ├── 📄 data_preprocessing.py          # Data cleaning
│   │   ├── 📄 feature_engineering.py         # Feature creation
│   │   └── 📄 feature_transformer.py         # Feature transformation
│   │
│   ├── 📂 models/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 trainer.py                     # Model training
│   │   ├── 📄 evaluator.py                   # Model evaluation
│   │   ├── 📄 predictor.py                   # Prediction logic
│   │   └── 📄 model_registry.py              # Model deployment
│   │
│   ├── 📂 pipelines/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 training_pipeline.py           # Training workflow
│   │   ├── 📄 prediction_pipeline.py         # Prediction workflow
│   │   └── 📄 deployment_pipeline.py         # Deployment workflow
│   │
│   └── 📂 utils/
│       ├── 📄 __init__.py
│       └── 📄 logger.py                       # Logging configuration
│
├── 📄 main.py                                 # Application entry point
├── 📄 requirements.txt                        # Python dependencies
├── 📄 README.md                               # Project documentation
└── 📄 .gitignore                              # Git ignore rules
```

---

## 🛠️ Technologies Used

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **scikit-learn** | 1.0+ | Machine learning |
| **pandas** | 1.3+ | Data manipulation |
| **numpy** | 1.21+ | Numerical computing |
| **Flask** | 2.0+ | Web framework |
| **scipy** | 1.7+ | Statistical testing |

### Libraries & Frameworks

**Machine Learning:**
- scikit-learn (model training, evaluation)
- scipy (statistical tests)

**Data Processing:**
- pandas (data manipulation)
- numpy (numerical operations)

**Web & API:**
- Flask (REST API)
- Flask-CORS (cross-origin support)

**Utilities:**
- pickle/joblib (model serialization)
- json (data interchange)
- logging (application logs)

---

## 📊 Results

### Model Performance Visualization

```
R² Score Comparison:
Training:  ████████████████████████████████████████ 87.7%
Test:      ███████████████████████████████████████  87.1%

Error Metrics:
MAE:       ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.05
RMSE:      █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.06
```

### Sample Predictions

| Input Features | Predicted Score | Actual Score | Error |
|---------------|-----------------|--------------|-------|
| Female, Group B, Bachelor's, Standard lunch | 74.87 | 76 | -1.13 |
| Male, Group A, High school, Free lunch | 58.32 | 59 | -0.68 |
| Female, Group E, Master's, Standard lunch | 82.15 | 81 | +1.15 |

### Key Insights

1. **Strong Predictive Power:** Model explains 87.1% of variance in math scores
2. **Low Prediction Error:** Average error of only 4 points on 0-100 scale
3. **Consistent Performance:** Similar results on training and test sets
4. **No Data Drift:** Stable feature distributions ensure reliable predictions

---

## 🔮 Future Enhancements

### Short-term Goals

- [ ] Add more ML algorithms (Random Forest, XGBoost, Neural Networks)
- [ ] Implement model comparison dashboard
- [ ] Add cross-validation reporting
- [ ] Create interactive visualization dashboard
- [ ] Add model explainability (SHAP values, feature importance)

### Medium-term Goals

- [ ] Implement automated retraining pipeline
- [ ] Add A/B testing framework
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] Add authentication and rate limiting
- [ ] Implement comprehensive testing suite
- [ ] Add CI/CD pipeline

### Long-term Goals

- [ ] Real-time model monitoring and alerting
- [ ] Multi-model ensemble predictions
- [ ] Automated hyperparameter tuning
- [ ] Mobile application integration
- [ ] Advanced feature store implementation
- [ ] Model versioning and rollback system

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/StudentPerformancePrediction-End-To-End.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass
- Write clear commit messages

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset source: [Student Performance Dataset]
- Inspired by real-world ML deployment practices
- Thanks to the open-source community for amazing tools
- Special thanks to scikit-learn and Flask communities

---

## 📞 Contact

### Get in Touch

- **Issues:** [GitHub Issues](https://github.com/yourusername/StudentPerformancePrediction-End-To-End/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/StudentPerformancePrediction-End-To-End/discussions)
- **Email:** your.email@example.com

### Support

If you encounter any issues or have questions:

1. Check the [documentation](#-quick-start)
2. Search [existing issues](https://github.com/yourusername/StudentPerformancePrediction-End-To-End/issues)
3. Open a [new issue](https://github.com/yourusername/StudentPerformancePrediction-End-To-End/issues/new)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for the ML Community**

[⬆ Back to Top](#-student-performance-prediction---end-to-end-ml-pipeline)

---

**© 2025 Student Performance Prediction Project. All rights reserved.**

</div>








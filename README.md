# Real-Time Fraud Detection Pipeline

An anomaly-detection-based fraud detection project using machine learning to identify potentially fraudulent credit card transactions. This project explores transaction data, performs preprocessing and exploratory data analysis, develops an Isolation Forest model, and evaluates its performance using appropriate metrics for highly imbalanced datasets.

## Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small proportion of all transactions.

This project aims to build a fraud detection pipeline that identifies suspicious transactions while studying the trade-off between detecting fraud and generating false alarms.

**Current stage:** Core Data Science Development and Exploratory Analysis completed.

## Objectives

- Analyze and understand credit card transaction data.
- Identify class imbalance and duplicate records.
- Perform data preprocessing and feature scaling.
- Conduct exploratory data analysis (EDA).
- Study correlations between transaction features and fraud labels.
- Develop an anomaly detection model using Isolation Forest.
- Tune the anomaly detection threshold.
- Compare baseline and tuned model performance.

## Dataset

**Dataset:** Credit Card Fraud Detection Dataset (Kaggle)

| Attribute | Details |
|---|---|
| Total transactions | 284,807 |
| Total features | 30 + Class |
| Genuine transactions | 284,315 |
| Fraudulent transactions | 492 |
| Missing values | 0 |
| Duplicate rows identified | 1,081 |

The dataset contains anonymized transaction features (V1–V28), Time, Amount, and Class.

- `Class = 0`: Genuine transaction
- `Class = 1`: Fraudulent transaction

**Note:** The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.173% of all transactions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Git and GitHub

## Project Workflow

### 1. Data Exploration

- Inspected dataset dimensions and data types.
- Analyzed fraud and genuine transaction distribution.
- Checked missing values.
- Examined transaction amount statistics.
- Identified duplicate records.

### 2. Exploratory Data Analysis

Generated visualizations to understand the dataset:

- Class distribution.
- Transaction amount distribution.
- Transaction amount comparison by class.
- Transaction time distribution.

### 3. Correlation Analysis

Performed Pearson correlation analysis to identify features associated with the target variable.

Top features by absolute correlation with the fraud label:

| Feature | Correlation |
|---|---:|
| V17 | -0.3265 |
| V14 | -0.3025 |
| V12 | -0.2606 |
| V10 | -0.2169 |
| V16 | -0.1965 |

A correlation heatmap was generated to visualize relationships between features.

**Note:** Correlation indicates statistical association, not causation.

### 4. Data Preprocessing

- Performed stratified train-test splitting.
- Used an 80:20 training and testing ratio.
- Applied StandardScaler for feature scaling.
- Preserved the original fraud distribution across training and testing sets.

| Dataset | Records |
|---|---:|
| Training | 227,845 |
| Testing | 56,962 |

### 5. Baseline Model – Isolation Forest

Implemented Isolation Forest for anomaly detection.

The model identifies unusual transaction patterns and classifies potential anomalies as fraudulent transactions.

**Configuration:**
- Estimators: 200
- Contamination: Training fraud proportion
- Random state: 42

### 6. Threshold Tuning

A second Isolation Forest model was developed using a validation-based threshold selection approach.

- Used a stratified training-validation split.
- Generated anomaly scores.
- Applied Precision-Recall analysis.
- Selected a threshold based on validation F1-score.
- Evaluated the selected threshold on the test dataset.

Selected threshold: `0.132135`

## Model Evaluation

The models were evaluated using:

- Precision
- Recall
- F1-score
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

### Model Comparison

| Metric | Baseline | Tuned |
|---|---:|---:|
| Precision | 0.2804 | 0.2162 |
| Recall | 0.3061 | 0.3265 |
| F1-score | 0.2927 | 0.2602 |
| PR-AUC | 0.1717 | 0.1553 |
| True Positives | 30 | 32 |
| False Positives | 77 | 116 |
| False Negatives | 68 | 66 |

### Key Observations

- The tuned model detected 2 additional fraudulent transactions.
- It also generated 39 additional false positives.
- The baseline model achieved better precision, F1-score, and PR-AUC.
- Threshold selection involves a trade-off between fraud detection and false alarms.

The results demonstrate why model evaluation must consider multiple metrics rather than accuracy alone.

## Repository Structure

```text
Real-Time-Fraud-Detection/
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── eda/
│   │   ├── amount_comparison.png
│   │   ├── amount_distribution.png
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_correlations.csv
│   │   └── transaction_time.png
│   │
│   └── model_comparison.csv
│
├── src/
│   ├── data_analysis.py
│   ├── duplicate_analysis.py
│   ├── preprocessing.py
│   ├── eda_visualization.py
│   ├── correlation_analysis.py
│   ├── train_model.py
│   ├── tune_threshold.py
│   └── model_comparison.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

*Note: Large raw and processed datasets are not included in the public repository.*

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Vallabh-2946/Real-Time-Fraud-Detection.git
```

### 2. Navigate to the project directory

```bash
cd Real-Time-Fraud-Detection
```

### 3. Create a virtual environment

```bash
py -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Dataset Setup

Download the Credit Card Fraud Detection dataset from Kaggle and place it in the expected data directory.

Refer to `data/README.md` for dataset instructions.

### 7. Execute the analysis scripts

Run the scripts in the appropriate workflow order:

```bash
py src/data_analysis.py
py src/duplicate_analysis.py
py src/preprocessing.py
py src/eda_visualization.py
py src/correlation_analysis.py
py src/train_model.py
py src/tune_threshold.py
py src/model_comparison.py
```

## Current Development Status

- [x] Dataset exploration
- [x] Duplicate analysis
- [x] Data preprocessing
- [x] Exploratory data analysis
- [x] Feature correlation analysis
- [x] Baseline Isolation Forest model
- [x] Threshold tuning
- [x] Model comparison
- [ ] Real-time transaction simulation
- [ ] Streamlit dashboard
- [ ] End-to-end pipeline integration

## Future Enhancements

- Develop a real-time transaction simulation pipeline.
- Build an interactive Streamlit dashboard.
- Improve fraud detection using additional machine learning algorithms.
- Explore advanced anomaly detection techniques.
- Optimize the balance between precision and recall.
- Integrate model predictions into an end-to-end fraud monitoring system.

## Contributors

Developed as an academic project.

## Disclaimer

This project is developed for educational and research purposes. The current model is an experimental anomaly detection prototype and is not intended for deployment in production banking systems.

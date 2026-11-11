
# Real-Time Fraud Detection Pipeline

## Project Overview

The **Real-Time Fraud Detection Pipeline** is a machine learning-based project designed to identify potentially fraudulent credit card transactions.

The project aims to analyze transaction data, detect unusual patterns, generate fraud alerts, and display transaction insights through an interactive dashboard.

## Problem Statement

Credit card fraud is a major challenge in digital banking. Fraudulent transactions represent a very small proportion of all transactions, making detection difficult due to highly imbalanced datasets.

This project explores machine learning techniques to identify suspicious transactions and build a simulated real-time fraud detection system.

## Objectives

- Analyze and understand credit card transaction data.
- Perform data cleaning and preprocessing.
- Investigate duplicate records and class imbalance.
- Develop an anomaly detection model using Isolation Forest.
- Evaluate model performance using suitable classification metrics.
- Build a simulated transaction processing pipeline.
- Generate alerts for suspicious transactions.
- Develop an interactive Streamlit dashboard.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Development Environment | VS Code |

## Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders in September 2013.

| Attribute | Value |
|---|---:|
| Total transactions | 284,807 |
| Genuine transactions | 284,315 |
| Fraudulent transactions | 492 |
| Total features | 30 |
| Total columns | 31 |
| Missing values | 0 |

The `Class` column represents the transaction label:

- `0` – Genuine transaction
- `1` – Fraudulent transaction

### Dataset Setup

The original dataset is not included in this repository because the CSV file is approximately 150 MB.

To use the project:

1. Download the dataset from Kaggle.
2. Extract `creditcard.csv`.
3. Create a `data` folder in the project root.
4. Place `creditcard.csv` inside the `data` folder.

Expected location:

`data/creditcard.csv`

## Project Structure

```text
Real-Time-Fraud-Detection/
│
├── data/
│   └── creditcard.csv          # Download separately
│
├── src/
│   ├── data_analysis.py
│   └── duplicate_analysis.py
│
├── models/                     # Saved ML models
│
├── dashboard/                  # Streamlit dashboard
│
├── reports/                    # Analysis reports and outputs
│
├── .gitignore
├── requirements.txt
└── README.md
```

Additional files and modules will be added as development progresses.

## Work Completed

### 1. Dataset Exploration

- Loaded the credit card transaction dataset using Pandas.
- Examined dataset dimensions and column names.
- Inspected data types and statistical summaries.
- Checked for missing values.
- Analyzed transaction class distribution.
- Compared transaction amounts for genuine and fraudulent transactions.

### 2. Duplicate Analysis

- Identified 1,081 duplicate occurrences.
- Identified 1,854 rows belonging to duplicate groups.
- Examined genuine and fraudulent labels within duplicate groups.
- Inspected sample duplicate records.

Duplicate records have been investigated but have not yet been removed.

## Machine Learning Approach

The project plans to use **Isolation Forest**, an unsupervised anomaly detection algorithm, to identify unusual transaction patterns.

Model evaluation will consider:

- Precision
- Recall
- F1-score
- Precision-Recall AUC
- Confusion Matrix

Accuracy alone will not be used as the primary evaluation metric because the dataset is highly imbalanced.

## Planned Development

1. Data preprocessing and feature preparation.
2. Machine learning model development.
3. Model evaluation and optimization.
4. Simulated real-time transaction processing.
5. Fraud alert generation.
6. Transaction velocity monitoring.
7. Interactive Streamlit dashboard.
8. Visualization and report generation.
9. Testing and documentation.

## Project Scope and Limitations

- This project uses a publicly available historical dataset.
- Real-time processing will be simulated.
- The dataset does not contain actual customer account identifiers.
- Any generated account identifiers will be simulated.
- An anomaly detected by the model is not necessarily confirmed fraud.
- This project does not connect to real banking systems.

## Team Contributions

The project is developed collaboratively, covering:

- Data analysis and machine learning.
- Pipeline development and dashboard implementation.
- Integration, testing, documentation, and presentation.

## Project Status

**Status:** In Progress

This repository will be updated throughout the development process as new modules, models, visualizations, and documentation are completed.


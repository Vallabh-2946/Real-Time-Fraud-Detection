
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# File paths
DATA_DIR = "data/processed"
MODEL_DIR = "models"

# Load preprocessed training and testing data
print("Loading preprocessed datasets...")

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Calculate approximate fraud proportion in training data
fraud_rate = y_train.mean()

print("\nExpected fraud proportion:", fraud_rate)

# Initialize Isolation Forest
print("\nInitializing Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=fraud_rate,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("Training model...")

model.fit(X_train)

print("Model training completed!")

# Predict test transactions
# Isolation Forest: 1 = normal, -1 = anomaly
raw_predictions = model.predict(X_test)

# Convert predictions into project labels:
# 0 = genuine, 1 = potentially fraudulent
y_pred = np.where(raw_predictions == -1, 1, 0)

# Calculate anomaly scores
anomaly_scores = -model.decision_function(X_test)

# Evaluate model
print("\n========== MODEL EVALUATION ==========")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraud"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

pr_auc = average_precision_score(y_test, anomaly_scores)

print("\n========== PERFORMANCE METRICS ==========")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))

# Save trained model
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, f"{MODEL_DIR}/isolation_forest.pkl")

print("\nModel saved successfully!")
print("Location: models/isolation_forest.pkl")

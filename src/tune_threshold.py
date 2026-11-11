
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load preprocessed training and testing data
DATA_DIR = "data/processed"
MODEL_DIR = "models"

print("Loading datasets...")

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

# Create separate training and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Sub-training samples:", len(X_subtrain))
print("Validation samples:", len(X_val))

# Train Isolation Forest
print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_subtrain)

print("Model training completed!")

# Calculate validation anomaly scores
val_scores = -model.decision_function(X_val)

# Find the best threshold using validation data
precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_scores
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\n========== VALIDATION RESULTS ==========")
print("Selected threshold:", round(best_threshold, 6))
print("Validation precision:", round(precision[best_index], 4))
print("Validation recall:", round(recall[best_index], 4))
print("Validation F1-score:", round(f1_scores[best_index], 4))

# Evaluate on untouched test data
test_scores = -model.decision_function(X_test)
y_pred = (test_scores >= best_threshold).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraud"],
        zero_division=0
    )
)

print("\n========== PERFORMANCE METRICS ==========")

print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))

print("F1-score:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("PR-AUC:", round(
    average_precision_score(y_test, test_scores), 4
))

# Save tuned model and selected threshold
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    f"{MODEL_DIR}/isolation_forest_tuned.pkl"
)

print("\nTuned model and threshold saved successfully!")

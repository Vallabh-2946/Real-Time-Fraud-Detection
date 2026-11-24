
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading test dataset...")

X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print("Test samples:", len(X_test))

# Load baseline model
print("\nLoading baseline model...")
baseline_model = joblib.load(
    f"{MODEL_DIR}/isolation_forest.pkl"
)

# Load tuned model and threshold
print("Loading tuned model...")
tuned_data = joblib.load(
    f"{MODEL_DIR}/isolation_forest_tuned.pkl"
)

tuned_model = tuned_data["model"]
best_threshold = tuned_data["threshold"]

# Baseline predictions
baseline_scores = -baseline_model.decision_function(X_test)
baseline_predictions = np.where(
    baseline_model.predict(X_test) == -1, 1, 0
)

# Tuned model predictions
tuned_scores = -tuned_model.decision_function(X_test)
tuned_predictions = (
    tuned_scores >= best_threshold
).astype(int)

# Function to calculate evaluation metrics
def evaluate_model(name, y_true, predictions, scores):

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "PR_AUC": average_precision_score(
            y_true, scores
        ),
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn
    }

# Compare both models
results = []

results.append(
    evaluate_model(
        "Baseline Isolation Forest",
        y_test,
        baseline_predictions,
        baseline_scores
    )
)

results.append(
    evaluate_model(
        "Tuned Isolation Forest",
        y_test,
        tuned_predictions,
        tuned_scores
    )
)

comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# Save comparison results
comparison.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)

print("\nModel comparison completed successfully!")
print("Results saved in reports/model_comparison.csv")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# File paths
DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)

# Save processed datasets
X_train_scaled.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test_scaled.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

# Save scaler for future predictions
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

print("\nPreprocessing completed successfully!")
print("Processed datasets saved in:", OUTPUT_DIR)
print("Scaler saved in:", MODEL_DIR)
print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

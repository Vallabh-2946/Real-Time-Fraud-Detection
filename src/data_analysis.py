
import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("\n========== DATASET INFORMATION ==========")

# Dataset dimensions
print("\nDataset Shape:", df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types and null values
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Fraud vs genuine transactions
print("\nTransaction Distribution:")
print(df["Class"].value_counts())

# Percentage distribution
print("\nTransaction Percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())


# Compare transaction amounts by class
print("\n========== FRAUD vs GENUINE ANALYSIS ==========")

print("\nTransaction Amount Statistics:")
print(df.groupby("Class")["Amount"].describe())

# Count duplicate records
print("\nDuplicate Rows:", df.duplicated().sum())

# Fraud and genuine counts
fraud_count = (df["Class"] == 1).sum()
genuine_count = (df["Class"] == 0).sum()

print("\nFraud-to-Genuine Ratio:")
print("Genuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)
print("Genuine transactions per fraud:", round(genuine_count / fraud_count, 2))


import pandas as pd

df = pd.read_csv("data/creditcard.csv")

# Identify exact duplicate records
duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df[duplicate_mask]

print("========== DUPLICATE ANALYSIS ==========")

# Number of repeated rows beyond their first occurrence
print("\nDuplicate count:", df.duplicated().sum())

# Total rows involved in duplicate groups
print("Rows involved in duplicate groups:", len(duplicate_rows))

# Check fraud labels among duplicate groups
print("\nClass distribution in duplicate groups:")
print(duplicate_rows["Class"].value_counts())

# Display sample duplicate records
print("\nSample duplicate records:")
print(duplicate_rows.head(10))

# Check duplicate records with fraud labels
print("\nFraud-labelled rows in duplicate groups:")
print(duplicate_rows[duplicate_rows["Class"] == 1].shape[0])

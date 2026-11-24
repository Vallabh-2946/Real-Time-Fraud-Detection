
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "reports/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Calculate correlation with the fraud label
correlation = df.corr(numeric_only=True)["Class"].drop("Class")

# Display the strongest positive and negative correlations
correlation = correlation.sort_values(key=abs, ascending=False)

print("\nTop 10 features correlated with the fraud label:")
print(correlation.head(10))

# Generate a feature correlation heatmap
plt.figure(figsize=(16, 12))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.show()

# Save correlation values
correlation.to_csv(f"{OUTPUT_DIR}/feature_correlations.csv")

print("\nCorrelation analysis completed successfully!")
print("Heatmap and correlation values saved in:", OUTPUT_DIR)

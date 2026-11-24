
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

sns.set_theme(style="whitegrid")

# 1. Fraud vs Genuine Transactions
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Class")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.show()

# 2. Transaction Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Amount", bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png")
plt.show()

# 3. Amount Comparison: Fraud vs Genuine
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Class", y="Amount")
plt.title("Transaction Amount: Fraud vs Genuine")
plt.xlabel("Transaction Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.savefig(f"{OUTPUT_DIR}/amount_comparison.png")
plt.show()

# 4. Transaction Activity Over Time
df["Time_Hours"] = df["Time"] / 3600

plt.figure(figsize=(10, 5))
sns.histplot(
    data=df,
    x="Time_Hours",
    hue="Class",
    bins=48,
    element="step",
    stat="count",
    common_norm=False
)
plt.title("Transaction Activity Over Time")
plt.xlabel("Time (Hours from Dataset Start)")
plt.ylabel("Number of Transactions")
plt.savefig(f"{OUTPUT_DIR}/transaction_time.png")
plt.show()

print("\nEDA visualizations generated successfully!")
print("Saved in:", OUTPUT_DIR)

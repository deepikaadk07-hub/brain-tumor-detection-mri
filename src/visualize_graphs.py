import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("tumor_features.csv")

# ----------------------------
# 1. Volume Distribution
# ----------------------------
plt.figure()
plt.hist(df["volume"], bins=10)
plt.title("Tumor Volume Distribution")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.show()

# ----------------------------
# 2. Slices vs Volume
# ----------------------------
plt.figure()
plt.scatter(df["slices"], df["volume"])
plt.title("Slices vs Tumor Volume")
plt.xlabel("Number of Slices")
plt.ylabel("Tumor Volume")
plt.show()

# ----------------------------
# 3. Label Count
# ----------------------------
plt.figure()
df["label"].value_counts().plot(kind='bar')
plt.title("Tumor Severity Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
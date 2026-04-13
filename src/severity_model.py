import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
# ----------------------------
# Load dataset
# ----------------------------

df = pd.read_csv("tumor_features.csv")
X = df[["volume", "slices", "max_area"]]
y = df["label"]
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "severity_model.pkl")
print(" Trained on FULL dataset!")

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier()
model.fit(X, y)

# ----------------------------
# Save model
# ----------------------------
joblib.dump(model, "severity_model.pkl")
print(" Severity model trained and saved!")
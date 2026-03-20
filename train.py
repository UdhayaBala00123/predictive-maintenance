import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data/ai4i2020.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------

# Rolling features
df["temp_roll_mean"] = df["temperature"].rolling(3).mean()
df["vib_roll_std"] = df["vibration"].rolling(3).std()

# Lag features
df["temp_lag1"] = df["temperature"].shift(1)
df["vib_lag1"] = df["vibration"].shift(1)

# Remove NaN values
df = df.dropna()

# -------------------------------
# 3. FEATURES & TARGET
# -------------------------------
X = df.drop(["time", "failure"], axis=1)
y = df["failure"]

# -------------------------------
# 4. CHECK CLASS DISTRIBUTION
# -------------------------------
print("Class Distribution:\n", y.value_counts())

# -------------------------------
# 5. HANDLE IMBALANCE
# -------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = {
    0: weights[0],
    1: weights[1] * 2   # give extra importance to failure class
}

# -------------------------------
# 6. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight=class_weights,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 8. PREDICTION (WITH THRESHOLD)
# -------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# Adjust threshold (important)
threshold = 0.3
y_pred = (y_prob > threshold).astype(int)

# -------------------------------
# 9. EVALUATION
# -------------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 10. SAVE MODEL
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\n✅ Model and columns saved successfully!")

import os
import glob
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

START = time.time()

DATA_PATH = "datasets/raw/CIC-IDS2017/MachineLearningCVE"
os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

print("Loading CIC-IDS2017 dataset...")

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

frames = []

for file in files:
    df = pd.read_csv(file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("Original records:", len(data))

# Clean column names and labels
data["Label"] = data["Label"].astype(str).str.strip()
data = data.dropna(subset=["Label"])

# Binary classification: BENIGN vs ATTACK
y = (data["Label"] != "BENIGN").astype(int)

X = data.drop(columns=["Label"])

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove constant columns
X = X.loc[:, X.nunique(dropna=False) > 1]

# Remove exact duplicate feature vectors
# Conflicting labels are excluded rather than arbitrarily assigned.
label_counts = pd.concat([X, y.rename("target")], axis=1)

conflict_mask = label_counts.groupby(
    list(X.columns),
    dropna=False
)["target"].transform("nunique") > 1

X = X.loc[~conflict_mask]
y = y.loc[~conflict_mask]

combined = X.copy()
combined["target"] = y

combined = combined.drop_duplicates()

y = combined.pop("target")
X = combined

print("Records after duplicate handling:", len(X))
print("Feature count:", X.shape[1])

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Imputation fitted on training data only
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

print("\n========== RANDOM FOREST RESULTS ==========")
print(classification_report(
    y_test,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and preprocessing
joblib.dump(rf, "models/random_forest_mvp.joblib")
joblib.dump(imputer, "models/imputer_mvp.joblib")

# Unsupervised Isolation Forest
print("\nTraining Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)

iso_predictions = iso.predict(X_test)
iso_predictions = (iso_predictions == -1).astype(int)

print("\n========== ISOLATION FOREST RESULTS ==========")
print(classification_report(
    y_test,
    iso_predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

joblib.dump(iso, "models/isolation_forest_mvp.joblib")

# Save evaluation
results = {
    "Random Forest Macro F1": f1_score(
        y_test, predictions, average="macro"
    ),
    "Isolation Forest Macro F1": f1_score(
        y_test, iso_predictions, average="macro"
    ),
    "Training Records": len(X_train),
    "Testing Records": len(X_test),
    "Features": X.shape[1]
}

pd.DataFrame([results]).to_csv(
    "results/metrics/mvp_results.csv",
    index=False
)

print("\n========== COMPLETE ==========")
print(results)
print("Elapsed seconds:", round(time.time() - START, 2))

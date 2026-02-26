import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

MODELS = "models/churn_model.pkl"

df = pd.read_csv("output.csv", parse_dates=["InvoiceDate"])

# Time-based split
midpoint = df["InvoiceDate"].min() + (
    df["InvoiceDate"].max() - df["InvoiceDate"].min()
) / 2

first_half = df[df["InvoiceDate"] <= midpoint]
second_half = df[df["InvoiceDate"] > midpoint]

churned_ids = set(first_half["CustomerID"]) - set(second_half["CustomerID"])

# Feature Engineering (first-half behavior only)
features = first_half.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (midpoint - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    AvgOrderValue=("TotalPrice", "mean"),
    UniqueItems=("StockCode", "nunique"),
    TotalQuantity=("Quantity", "sum")
).reset_index()

features["Churned"] = features["CustomerID"].isin(churned_ids).astype(int)
features = features.dropna()

X = features.drop(columns=["CustomerID", "Churned"])
y = features["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print(f"Test ROC-AUC: {auc:.4f}")
print(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(classification_report(y_test, y_pred))


os.makedirs("models", exist_ok=True)
joblib.dump(model, MODELS)
print(f"Model saved to {MODELS}") 
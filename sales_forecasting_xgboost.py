import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import os

MODELS = "models/sales_forecast_xgb_tuned.pkl"

df = pd.read_csv("output.csv", parse_dates=["InvoiceDate"])

# Weekly revenue
ts = (
    df.set_index("InvoiceDate")
      .resample("W")["TotalPrice"]
      .sum()
      .reset_index()
)

ts.columns = ["Date", "Revenue"]
ts = ts[ts["Revenue"] > 0]

print(f"Total weeks in data: {len(ts)}\n")

# Simplified Feature Engineering - LESS IS MORE
ts["WeekIndex"] = np.arange(len(ts))
ts["Month"] = ts["Date"].dt.month
ts["WeekOfYear"] = ts["Date"].dt.isocalendar().week.astype(int)

# Only 1 lag and 1 rolling mean (minimal features)
ts["Lag1"] = ts["Revenue"].shift(1)
ts["RollingMean4"] = ts["Revenue"].shift(1).rolling(4).mean()

ts = ts.dropna().reset_index(drop=True)

FEATURES = ["WeekIndex", "Month", "WeekOfYear", "Lag1", "RollingMean4"]

print(f"Features: {FEATURES}")
print(f"Training samples: {len(ts)}\n")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
mape_scores = []
mae_scores = []

print("="*60)
print("CROSS-VALIDATION (HEAVILY REGULARIZED XGBOOST)")
print("="*60)

for fold, (train_index, test_index) in enumerate(tscv.split(ts)):
    X_train, X_test = ts.iloc[train_index][FEATURES], ts.iloc[test_index][FEATURES]
    y_train, y_test = ts.iloc[train_index]["Revenue"], ts.iloc[test_index]["Revenue"]
    
    print(f"\nFold {fold + 1}: Train={len(X_train)}, Test={len(X_test)}")
    
    # HEAVILY REGULARIZED XGBoost - prevents overfitting
    model = XGBRegressor(
        n_estimators=50,              # Few trees
        learning_rate=0.05,           # Very slow learning
        max_depth=2,                  # Shallow trees
        min_child_weight=5,           # Prevent small leaves
        subsample=0.6,                # Sample 60% of rows
        colsample_bytree=0.6,         # Sample 60% of features
        gamma=1,                      # Penalty for splits
        reg_alpha=1,                  # L1 regularization
        reg_lambda=2,                 # L2 regularization
        random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    mape_scores.append(mape)
    mae_scores.append(mae)
    
    # Show actual vs predicted
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  Actual test values:   {[f'${v:,.0f}' for v in y_test.values[:3]]}")
    print(f"  Predicted values:     {[f'${v:,.0f}' for v in y_pred[:3]]}")

print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)
print(f"Average MAE:  ${np.mean(mae_scores):,.0f}")
print(f"Average MAPE: {np.mean(mape_scores):.2f}%")

# Train final model on all data
print("\n" + "="*60)
print("TRAINING FINAL MODEL")
print("="*60)

X_all = ts[FEATURES]
y_all = ts["Revenue"]

final_model = XGBRegressor(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=2,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=1,
    reg_alpha=1,
    reg_lambda=2,
    random_state=42
)

final_model.fit(X_all, y_all, verbose=False)

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, MODELS)
print(f"Model saved to {MODELS}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Forecast next 4 weeks
print("\n" + "="*60)
print("4-WEEK FORECAST")
print("="*60)

history = ts.copy()
last_index = history["WeekIndex"].iloc[-1]
future_preds = []

for i in range(1, 5):
    lag1 = history["Revenue"].iloc[-1]
    rm4 = history["Revenue"].iloc[-4:].mean()
    
    future_date = history["Date"].iloc[-1] + pd.Timedelta(weeks=i)
    
    new_row = pd.DataFrame([{
        "WeekIndex": last_index + i,
        "Month": future_date.month,
        "WeekOfYear": future_date.isocalendar().week,
        "Lag1": lag1,
        "RollingMean4": rm4
    }])
    
    pred = final_model.predict(new_row[FEATURES])[0]
    pred = max(0, pred)
    future_preds.append(round(pred, 0))
    
    print(f"Week {i} ({future_date.date()}): ${pred:,.0f}")

print(f"\nForecast: {future_preds}")
print(f"Average forecast: ${np.mean(future_preds):,.0f}")
print(f"Recent 4-week average: ${ts['Revenue'].tail(4).mean():,.0f}")

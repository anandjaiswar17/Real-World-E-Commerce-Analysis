import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS= "models/sales_forecast_model.pkl"

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

# Feature Engineering
ts["WeekIndex"] = np.arange(len(ts))
ts["Lag1"] = ts["Revenue"].shift(1)
ts["Lag2"] = ts["Revenue"].shift(2)
ts["Lag3"] = ts["Revenue"].shift(3)
ts["RollingMean4"] = ts["Revenue"].shift(1).rolling(4).mean()

ts = ts.dropna().reset_index(drop=True)

split = int(len(ts) * 0.8)
train = ts.iloc[:split]
test = ts.iloc[split:]

FEATURES = ["WeekIndex", "Lag1", "Lag2", "Lag3", "RollingMean4"]

X_train, y_train = train[FEATURES], train["Revenue"]
X_test, y_test = test[FEATURES], test["Revenue"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²:   {r2:.4f}")

# Save model
joblib.dump(model, MODELS)
print(f"Model saved to {MODELS}")

# Forecasting 4 week forward
history = ts.copy()
last_week = history["WeekIndex"].iloc[-1]

future_preds = []

for i in range(1, 5):
    lag1 = history["Revenue"].iloc[-1]
    lag2 = history["Revenue"].iloc[-2]
    lag3 = history["Revenue"].iloc[-3]
    rm4 = history["Revenue"].tail(4).mean()

    new_row = pd.DataFrame([{
        "WeekIndex": last_week + i,
        "Lag1": lag1,
        "Lag2": lag2,
        "Lag3": lag3,
        "RollingMean4": rm4
    }])

    pred = model.predict(new_row)[0]
    future_preds.append(pred)

    history = pd.concat([
        history,
        pd.DataFrame([{
            "Date": None,
            "Revenue": pred,
            "WeekIndex": last_week + i,
            "Lag1": None,
            "Lag2": None,
            "Lag3": None,
            "RollingMean4": None
        }])
    ], ignore_index=True) 

print("Next 4-week forecast:", [round(x, 0) for x in future_preds])
# 🛍️ Retail Data Analysis & ML Project

End-to-end machine learning project on an e-commerce retail dataset.

Includes:
- Data preprocessing (Excel → Clean CSV)
- Customer Churn Prediction (Random Forest)
- Sales Forecasting (Regression with lag features)

---

## 📁 Project Structure

```
Retail-Data-Analysis/
│
├── data/
│   ├── online_retail_II.xlsx
│   ├── output.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── sales_forecast_model.pkl
│
├── data_processing.py
├── churn_prediction.py
├── sales_forecasting.py
└── README.md
```

---

## ⚙️ Setup

Install dependencies:

```
pip install pandas numpy scikit-learn openpyxl joblib
```

---

## 🔄 Data Preprocessing

```
python data_processing.py
```

- Cleans Excel data
- Removes invalid records
- Creates `TotalPrice`
- Saves `retail_clean.csv`

---

## 🤖 Churn Prediction

```
python churn_prediction.py
```

- Random Forest model
- Time-based validation
- ROC-AUC evaluation
- Saves model to `models/churn_model.pkl`

---

## 📈 Sales Forecasting

```
python sales_forecasting.py
```

- Weekly revenue prediction
- Lag features
- MAE / RMSE / R² metrics
- 4-week forward forecast
- Saves model to `models/sales_forecast_model.pkl`

---

## 🚀 Highlights

- Time-based validation (no data leakage)
- Cross-validation
- Model persistence 
- Clean production-style structure 
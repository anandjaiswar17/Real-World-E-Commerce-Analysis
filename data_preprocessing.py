import pandas as pd

df = pd.read_excel("online_retail_II.xlsx")

df.rename(columns={
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID"
}, inplace=True)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

df = df.dropna(subset=["CustomerID", "Description", "InvoiceDate"])

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df = df.drop_duplicates()

df["CustomerID"] = df["CustomerID"].astype(int)

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

df.to_csv("output.csv", index=False)

print(f"Excel converted and cleaned CSV saved to: {"output.csv"}")
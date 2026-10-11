# 1. Install and Import Libraries
!pip install -q openpyxl tensorflow scikit-learn

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 2. Download and Load Dataset
url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
!wget -q "$url" -O online_retail.zip
!unzip -o -q online_retail.zip

df = pd.read_excel("Online Retail.xlsx")
print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)
print("\nFirst 5 records:\n", df.head())

# 3. Select Required Columns
df = df[["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]]

# 4. Data Cleaning
df = df.dropna(subset=["CustomerID"])                       # Remove missing customers
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]   # Remove cancelled orders
df = df[df["Quantity"] > 0]                                 # Keep positive quantities
df = df[df["UnitPrice"] > 0]                                # Keep positive prices
df = df.dropna(subset=["Description"])                      # Remove missing descriptions

print("\nAfter cleaning:")
print("Dataset shape:", df.shape)

# 5. Create Purchase Amount
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
print("\nTotal Sales:", round(df["TotalAmount"].sum(), 2))

# 6. Create Customer-Product Interactions
interaction_df = df.groupby(["CustomerID", "StockCode"]).agg(
    Quantity=("Quantity", "sum"),
    TotalAmount=("TotalAmount", "sum")
).reset_index()

print("\nCustomer-Product interactions:\n", interaction_df.head())

# 7. Encode Customer IDs and Product IDs
customers = interaction_df["CustomerID"].unique()
products = interaction_df["StockCode"].unique()

customer_to_index = {customer: idx for idx, customer in enumerate(customers)}
product_to_index = {product: idx for idx, product in enumerate(products)}

index_to_customer = {idx: customer for customer, idx in customer_to_index.items()}
index_to_product = {idx: product for product, idx in product_to_index.items()}

interaction_df["CustomerIndex"] = interaction_df["CustomerID"].map(customer_to_index)
interaction_df["ProductIndex"] = interaction_df["StockCode"].map(product_to_index)

print("\nEncoded interactions:\n", interaction_df.head())

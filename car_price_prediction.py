"""
Car Price Prediction - Simple Linear Regression Demo
=====================================================
Dataset: Used cars from CarDekho (Kaggle-sourced, ~430 rows)
Steps:  Load -> Clean -> Encode -> Split -> Train -> Evaluate -> Plot -> Predict
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
URL = ("https://raw.githubusercontent.com/YBIFoundation/Dataset/"
       "main/Car%20Price.csv")

df = pd.read_csv(URL)

# Keep only the columns we need; rename for clarity
df = df[["Brand", "Year", "KM_Driven", "Fuel", "Transmission", "Selling_Price"]]
df.columns = ["brand", "year", "mileage_km", "fuel_type", "transmission", "price"]

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. CLEAN
# ------------------------------------------------------------
print("Null values per column:")
print(df.isnull().sum(), "\n")

# Drop extreme outliers (prices outside a reasonable band)
df = df[(df["price"] > 50000) & (df["price"] < 10000000)]

print(f"After cleaning: {df.shape[0]} rows\n")

# ------------------------------------------------------------
# 3. ENCODE CATEGORICALS
# ------------------------------------------------------------
df_encoded = pd.get_dummies(
    df,
    columns=["brand", "fuel_type", "transmission"],
    drop_first=True          # avoid dummy-variable trap
)

# Features (X) and target (y)
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

print(f"Features after one-hot encoding: {X.shape[1]}\n")

# ------------------------------------------------------------
# 4. SPLIT (80% train, 20% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

# ------------------------------------------------------------
# 5. TRAIN
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. EVALUATE
# ------------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared:     {r2:.3f}")
print(f"Mean Abs Error: GHS {mae:,.0f}\n")

# ------------------------------------------------------------
# 7. PLOT
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=1, label="Perfect prediction")
plt.xlabel("Actual Price (GHS)")
plt.ylabel("Predicted Price (GHS)")
plt.title("Predicted vs Actual Car Prices")
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150)
plt.close()
print("Plot saved as pred_vs_actual.png\n")

# ------------------------------------------------------------
# 8. PREDICT A NEW CAR  <-- EDIT THE VALUES BELOW
# ------------------------------------------------------------
new_car = {
    "brand": "Toyota",
    "year": 2020,
    "mileage_km": 30000,
    "fuel_type": "Petrol",
    "transmission": "Manual"
}

# Build a DataFrame row with the same dummy columns as X_train
new_row = pd.DataFrame([{col: 0 for col in X_train.columns}])

for col in X_train.columns:
    if col.startswith("brand_"):
        if new_car["brand"] == col.replace("brand_", ""):
            new_row[col] = 1
    elif col.startswith("fuel_type_"):
        if new_car["fuel_type"] == col.replace("fuel_type_", ""):
            new_row[col] = 1
    elif col.startswith("transmission_"):
        if new_car["transmission"] == col.replace("transmission_", ""):
            new_row[col] = 1
    elif col == "year":
        new_row[col] = new_car["year"]
    elif col == "mileage_km":
        new_row[col] = new_car["mileage_km"]

pred_price = model.predict(new_row)[0]
print(f"--- Single Prediction ---")
print(f"{new_car['brand']} | Year: {new_car['year']} | "
      f"{new_car['mileage_km']:,} km | {new_car['fuel_type']} | "
      f"{new_car['transmission']}")
print(f"Predicted Price: GHS {pred_price:,.0f}")

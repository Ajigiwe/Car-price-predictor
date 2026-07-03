import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Car Price Predictor", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .stApp { background: white; }
    .stButton>button { width: 100%; }
    .stSelectbox label, .stNumberInput label { font-weight: 600; }
    h1 { color: black !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    URL = ("https://raw.githubusercontent.com/YBIFoundation/Dataset/"
           "main/Car%20Price.csv")
    df = pd.read_csv(URL)
    df = df[["Brand", "Year", "KM_Driven", "Fuel", "Transmission", "Selling_Price"]]
    df.columns = ["brand", "year", "mileage_km", "fuel_type", "transmission", "price"]
    df = df[(df["price"] > 50000) & (df["price"] < 10000000)]
    df_encoded = pd.get_dummies(
        df, columns=["brand", "fuel_type", "transmission"], drop_first=True
    )
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist(), df

model, feature_cols, df = load_and_train()

st.title("Car Price Predictor")

brand = st.selectbox("Brand", sorted(df["brand"].unique()))
fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))
year = st.selectbox("Year", list(range(2005, 2025)))
mileage_km = st.selectbox("Mileage (km)", [10000, 30000, 50000, 70000, 100000, 150000, 200000])

if st.button("Predict Price"):
    row = {col: 0 for col in feature_cols}
    row["year"] = year
    row["mileage_km"] = mileage_km
    for prefix, val in [("brand_", brand), ("fuel_type_", fuel_type), ("transmission_", transmission)]:
        col = prefix + val
        if col in row:
            row[col] = 1
    price = model.predict(pd.DataFrame([row]))[0]
    st.success(f"### GHS {price:,.0f}")

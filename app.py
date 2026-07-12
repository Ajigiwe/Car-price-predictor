# Bring in the Streamlit library so we can build a web app with buttons, dropdowns, and text
import streamlit as st

# Bring in pandas, a library for working with tables of data (like Excel spreadsheets)
import pandas as pd

# Import a function that splits our data into two parts: one for teaching the model, one for testing it
from sklearn.model_selection import train_test_split

# Import the Linear Regression algorithm -- our "brain" that learns patterns from car data
from sklearn.linear_model import LinearRegression

# Set up the browser tab title to "Car Price Predictor"
# layout="centered" keeps the content in the middle instead of stretching across the screen
# initial_sidebar_state="collapsed" hides the left sidebar since we are not using it
st.set_page_config(page_title="Car Price Predictor", layout="centered", initial_sidebar_state="collapsed")

# Inject custom CSS (styling instructions) into the web page
# This changes how the app looks -- colors, font weights, button sizes
st.markdown("""
<style>
    .stApp { background: white; }                           /* Make the whole app background white */
    .stButton>button { width: 100%; }                       /* Make every button stretch full width */
    .stSelectbox label, .stNumberInput label { font-weight: 600; }  /* Make dropdown labels bold */
    h1 { color: black !important; font-weight: 800 !important; }    /* Make the main title black and extra bold */
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True tells Streamlit "trust this CSS, it is safe to render"

# @st.cache_data is a Streamlit feature that saves (caches) the result of this function
# Without it, the app would reload the CSV and retrain the model every time you click a button
# With it, this heavy work happens only once, then the result is reused
@st.cache_data
def load_and_train():
    # --- Step 1: Load the car dataset ---
    # Try to fetch the CSV file from the internet (GitHub)
    try:
        # The raw data lives at this web address (URL)
        URL = ("https://raw.githubusercontent.com/YBIFoundation/Dataset/"
               "main/Car%20Price.csv")
        # pd.read_csv() downloads the CSV and turns it into a pandas DataFrame (a table)
        df = pd.read_csv(URL)
    except Exception:
        # If the internet is down or the URL fails, load from the local backup file instead
        import os  # os helps us build file paths that work on any computer
        # os.path.dirname(__file__) gives the folder where this Python file lives
        # os.path.join combines that folder path with the filename "car_price.csv"
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "car_price.csv"))

    # --- Step 2: Pick only the columns we care about ---
    # The original CSV has many columns; we only need these six
    df = df[["Brand", "Year", "KM_Driven", "Fuel", "Transmission", "Selling_Price"]]

    # Rename columns to shorter, lowercase names -- easier to type and read
    df.columns = ["brand", "year", "mileage_km", "fuel_type", "transmission", "price"]

    # --- Step 3: Remove obvious errors and extreme outliers ---
    # Keep only cars priced between 50,000 and 10,000,000 GHS
    # This removes data-entry mistakes (e.g., a car listed for 1 GHS or 1 billion GHS)
    df = df[(df["price"] > 50000) & (df["price"] < 10000000)]

    # --- Step 4: Convert text categories into numbers the model can understand ---
    # Machine learning models cannot work with words like "Toyota" or "Petrol"
    # pd.get_dummies() creates binary (0/1) columns for each possible category value
    # Example: "brand" becomes brand_Toyota=1, brand_Honda=0, brand_BMW=0 ... for a Toyota car
    # drop_first=True drops the first category per feature to avoid redundant information
    # (if you know it is not Honda and not BMW, you already know it is Toyota)
    df_encoded = pd.get_dummies(
        df, columns=["brand", "fuel_type", "transmission"], drop_first=True
    )

    # --- Step 5: Separate the "what we know" (features) from "what we want to predict" (target) ---
    # X = the input features (everything except the price column)
    X = df_encoded.drop("price", axis=1)  # axis=1 means "drop a column" (axis=0 would mean "drop a row")
    # y = the target we are trying to predict (the car's selling price)
    y = df_encoded["price"]

    # --- Step 6: Split data into training set and test set ---
    # Training set (80%): Used to teach the model by showing it examples with known answers
    # Test set (20%): Held back to check how well the model learned (like a final exam)
    # random_state=42 ensures the split is the same every time (42 is just a fixed "seed" number)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Step 7: Create and train the model ---
    # LinearRegression() creates an empty "brain" with no knowledge yet
    model = LinearRegression()
    # .fit() teaches the model: "here are the features (X_train), here are the answers (y_train) -- find the pattern"
    # Internally, it finds the best line/plane through the data by minimising prediction errors
    model.fit(X_train, y_train)

    # Return three things:
    # 1) model -- the trained "brain" ready to make predictions
    # 2) X_train.columns.tolist() -- the names of all feature columns (needed later to build the user's input)
    # 3) df -- the raw (unencoded) data (needed to populate the dropdown menus with brand names, fuel types, etc.)
    return model, X_train.columns.tolist(), df


# Call the function once -- this triggers the caching, so it runs only the first time
# Unpack the three returned values into separate variables
model, feature_cols, df = load_and_train()

# Display the big title at the top of the webpage
st.title("Car Price Predictor")

# ==================== USER INPUT SECTION ====================

# Dropdown: pick a car brand from the unique brands found in the dataset
# sorted() puts them in alphabetical order
brand = st.selectbox("Brand", sorted(df["brand"].unique()))

# Dropdown: pick a fuel type (Petrol, Diesel, CNG, etc.)
fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))

# Dropdown: pick the transmission type (Manual or Automatic)
transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))

# Dropdown: pick the model year (2005 through 2024)
year = st.selectbox("Year", list(range(2005, 2025)))

# Dropdown: pick the mileage (kilometers driven)
mileage_km = st.selectbox("Mileage (km)", [10000, 30000, 50000, 70000, 100000, 150000, 200000])

# ==================== PREDICTION SECTION ====================

# Show a button; when the user clicks it, the code indented below runs
if st.button("Predict Price"):

    # --- Build the input row that the model expects ---
    # Start with a dictionary where every feature column is set to 0
    # feature_cols is the list of column names the model was trained on (e.g., "year", "mileage_km", "brand_Toyota", ...)
    row = {col: 0 for col in feature_cols}

    # Set the two numeric columns to the values the user chose
    row["year"] = year
    row["mileage_km"] = mileage_km

    # Set the correct one-hot encoded column to 1 for each categorical choice
    # For example, if the user chose brand = "Toyota", we find "brand_Toyota" and set it to 1
    # If the user chose fuel_type = "Petrol", we find "fuel_type_Petrol" and set it to 1
    for prefix, val in [("brand_", brand), ("fuel_type_", fuel_type), ("transmission_", transmission)]:
        col = prefix + val  # Combine prefix and value, e.g., "brand_" + "Toyota" = "brand_Toyota"
        if col in row:      # Safety check: only set if the column exists in our feature list
            row[col] = 1

    # --- Make the prediction ---
    # pd.DataFrame([row]) wraps our single-row dictionary in a DataFrame (table) that the model expects
    # model.predict() runs the trained model on our input and returns an array of predictions
    # [0] extracts the first (and only) prediction from that array
    price = model.predict(pd.DataFrame([row]))[0]

    # Display the predicted price in a green success box
    # f"### GHS {price:,.0f}" formats the price with thousand separators (e.g., 1,500,000) and no decimals
    # The ### makes it appear as a level-3 heading (bold and slightly larger)
    st.success(f"### GHS {price:,.0f}")

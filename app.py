# Bring in the Streamlit library so we can build a web app with buttons, dropdowns, and text
import streamlit as st

# Bring in pandas, a library for working with tables of data (like Excel spreadsheets)
import pandas as pd

# Bring in numpy for numerical operations
import numpy as np

# Import a function that splits our data into two parts: one for teaching the model, one for testing it
from sklearn.model_selection import train_test_split

# Import a classification model -- Random Forest, a collection of decision trees that vote on the answer
# Works well out-of-the-box, handles both numeric and categorical data, resists overfitting
from sklearn.ensemble import RandomForestClassifier

# Import classification evaluation metrics
# accuracy  = % of correct predictions overall
# precision = when we predict a category, how often are we right?
# recall    = of all true cases in a category, how many did we catch?
# f1_score  = balanced average of precision and recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

# Define price category bins with meaningful labels
# pd.cut converts a continuous number (price) into a category label based on which range it falls into
# Bins are in GHS, labels describe the market segment
PRICE_BINS = [0, 300000, 800000, 1800000, 99999999]        # Edges of each price range (in GHS)
PRICE_LABELS = ["Budget", "Mid-Range", "Premium", "Luxury"] # Human-readable category names

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

    # --- Step 4: Convert the continuous price into a category label ---
    # Instead of predicting an exact number, we now predict which price bracket the car belongs to
    # pd.cut takes each car's price and assigns a label based on which bin it falls into
    # Example: a car with price 450,000 falls into the "Mid-Range" category
    df["price_category"] = pd.cut(
        df["price"],           # The column to bin
        bins=PRICE_BINS,       # The edges of the price ranges
        labels=PRICE_LABELS,   # The human-readable names for each range
        include_lowest=True    # Make sure the lowest edge (0) is included
    )

    # --- Step 5: One-hot encode categorical features ---
    # Machine learning models cannot work with words like "Toyota" or "Petrol"
    # pd.get_dummies() creates binary (0/1) columns for each possible category value
    # Example: "brand" becomes brand_Toyota=1, brand_Honda=0, brand_BMW=0 ... for a Toyota car
    # drop_first=True drops the first category per feature to avoid redundant information
    # (if you know it is not Honda and not BMW, you already know it is Toyota)
    df_encoded = pd.get_dummies(
        df, columns=["brand", "fuel_type", "transmission"], drop_first=True
    )

    # --- Step 6: Separate features (X) from target (y) ---
    # X = the input features (everything except the original price column and the new category column)
    X = df_encoded.drop(["price", "price_category"], axis=1)
    # y = the target -- the price category we want to predict (Budget, Mid-Range, Premium, or Luxury)
    y = df["price_category"]

    # --- Step 7: Split data into training set and test set ---
    # Training set (80%): Used to teach the model by showing it examples with known answers
    # Test set (20%): Held back to check how well the model learned (like a final exam)
    # random_state=42 ensures the split is the same every time (42 is just a fixed "seed" number)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        # stratify=y ensures each price category has the same proportion in train and test sets
    )

    # --- Step 8: Create and train the classification model ---
    # RandomForestClassifier is a collection of decision trees that each vote on the category
    # n_estimators=100 means 100 trees vote; more trees = more stable but slower to train
    # random_state=42 ensures the same random trees are built every time (reproducible results)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # .fit() teaches the model: "here are the features (X_train), here are the correct categories (y_train)"
    model.fit(X_train, y_train)

    # --- Step 9: Evaluate the model using classification metrics ---
    # Predict categories for the 20% of cars the model has never seen before
    y_pred = model.predict(X_test)

    # Accuracy: What percentage of all predictions were correct?
    # Example: 0.75 means the model correctly classified 75% of cars in the test set
    accuracy = accuracy_score(y_test, y_pred)

    # Precision: When the model says "Luxury", how often is it actually a luxury car?
    # weighted average across all categories, accounting for class imbalance
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    # Recall: Of all the actual luxury cars, what percentage did the model correctly identify?
    # Weighted average -- high recall means the model misses few cars in each category
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # F1 Score: The harmonic mean of precision and recall -- a single balanced score
    # Ranges from 0 (worst) to 1 (best). F1 is high only when both precision AND recall are high
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Return everything the UI needs:
    # 1) model         -- trained Random Forest classifier
    # 2) feature_cols -- list of all encoded feature column names (for building the input row)
    # 3) df            -- raw dataframe with unique brand/fuel/transmission values for dropdowns
    # 4) accuracy      -- overall % of correct predictions on test set
    # 5) precision     -- weighted avg: when we predict a class, how often are we right?
    # 6) recall        -- weighted avg: of all true cases in a class, how many did we catch?
    # 7) f1            -- weighted harmonic mean of precision and recall
    return model, X_train.columns.tolist(), df, accuracy, precision, recall, f1


# Call the function once -- this triggers the caching, so it runs only the first time
# Unpack the seven returned values into separate variables
model, feature_cols, df, accuracy, precision, recall, f1 = load_and_train()

# Display the big title at the top of the webpage
st.title("Car Price Predictor")

# ==================== MODEL PERFORMANCE METRICS ====================
# Show how well the trained classification model performs on unseen data
# These metrics answer different questions about the model's reliability

st.subheader("Model Performance")

# Display the four metrics side by side in four columns
col1, col2, col3, col4 = st.columns(4)

# Column 1: Accuracy -- the simplest metric, overall % correct
# High accuracy means the model gets most predictions right
# But can be misleading if one category dominates (e.g., most cars are "Mid-Range")
col1.metric(
    label="Accuracy",
    value=f"{accuracy:.1%}",
    help="Percentage of all predictions that were correct. Overall correctness."
)

# Column 2: Precision -- trustworthiness of positive predictions
# When the model says "this is a Luxury car", precision tells us how likely that is to be true
# High precision = few false alarms (the model is cautious and only claims categories when sure)
col2.metric(
    label="Precision",
    value=f"{precision:.1%}",
    help="Of the times we predicted a category, how often were we right? High precision = few false alarms."
)

# Column 3: Recall -- ability to find all cases of a category
# Of all actual Budget cars, what % did the model correctly find?
# High recall = few missed cases (the model catches most cars in each category)
col3.metric(
    label="Recall",
    value=f"{recall:.1%}",
    help="Of all actual cars in a category, what % did we catch? High recall = few missed cars."
)

# Column 4: F1 Score -- the balanced single-number summary
# F1 combines precision and recall into one number
# High F1 only when BOTH precision and recall are high (prevents gaming one at expense of other)
col4.metric(
    label="F1 Score",
    value=f"{f1:.1%}",
    help="Harmonic mean of precision and recall. Best single metric for overall model quality."
)

# Add a light divider line between the metrics and the input section
st.divider()

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
if st.button("Predict Price Category"):

    # --- Build the input row that the model expects ---
    # Start with a dictionary where every feature column is set to 0
    # feature_cols is the list of column names the model was trained on
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
    # model.predict() returns the predicted category label (e.g., "Premium")
    # [0] extracts the first (and only) prediction from the array
    predicted_category = model.predict(pd.DataFrame([row]))[0]

    # --- Get prediction probabilities for each category ---
    # model.predict_proba() returns an array of confidence scores (0 to 1) for each category
    # These tell us how confident the model is in its prediction
    # probabilities[0] gives the first row's probabilities
    probabilities = model.predict_proba(pd.DataFrame([row]))[0]

    # Map each category label to its confidence percentage
    # model.classes_ lists the categories in the same order as the probabilities array
    confidence = dict(zip(model.classes_, probabilities))

    # Display the predicted category in a green success box
    st.success(f"### {predicted_category}")

    # Show a breakdown of how confident the model is for each possible category
    # This reveals if the model is torn between two categories or confidently picks one
    st.write("**Confidence Breakdown:**")
    for cat, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
        st.text(f"{cat}: {prob:.1%}")

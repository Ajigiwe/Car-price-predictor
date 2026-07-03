# Car Price Predictor

A simple car price prediction project built with Python for a school presentation.

## Files

- **`car_price_prediction.py`** — Linear regression model: loads data, cleans, encodes, trains, evaluates, plots, and predicts
- **`app.py`** — Streamlit web UI with dropdowns for brand, year, mileage, fuel type, and transmission

## Dataset

Used cars from CarDekho (Kaggle-sourced, ~4300 rows). Downloaded automatically from YBIFoundation's GitHub.

## How to Run

```bash
# Train and evaluate
python car_price_prediction.py

# Launch web UI
streamlit run app.py
```

## Requirements

`pandas`, `scikit-learn`, `matplotlib`, `streamlit`

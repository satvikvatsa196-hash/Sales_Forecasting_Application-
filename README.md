# Sales Forecasting System

A complete AI/ML-powered sales forecasting application built with **Streamlit**, featuring multiple time series models, interactive visualizations, and a sleek dark UI.

---

## Features

- **7 Forecasting Models**: Moving Average, Exponential Smoothing, ARIMA, SARIMA, Linear Regression, XGBoost, Prophet
- **Interactive UI**: Built with Streamlit + Plotly, dark futuristic theme
- **Data Flexibility**: Use built-in synthetic data or upload your own CSV/Excel
- **Model Comparison**: Side-by-side comparison with MAE, RMSE, MAPE, R²
- **Seasonality Analysis**: Weekly, monthly, quarterly, and yearly patterns
- **Residual Analysis**: Diagnose model fit
- **Download Forecasts**: Export predictions as CSV

---

## Project Structure

```
sales_forecasting/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   ├── __init__.py
│   └── forecasting_models.py       # All ML/statistical models
├── utils/
│   ├── __init__.py
│   ├── data_generator.py           # Synthetic data & preprocessing
│   └── visualizations.py           # Plotly chart builders
└── README.md
```

---

## Installation & Setup

### 1. Clone / navigate to the project
```bash
cd sales_forecasting
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on Prophet**: Prophet requires additional system dependencies.
> On Ubuntu/Debian: `sudo apt-get install libstan-math-dev`
> On Mac: `brew install cmdstan`
> Prophet will be skipped gracefully if unavailable.

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Models Overview

| Model | Type | Best For |
|-------|------|----------|
| Moving Average | Statistical | Smooth, stable series |
| Exponential Smoothing | Statistical | Trend + seasonal data |
| ARIMA | Statistical | Stationary time series |
| SARIMA | Statistical | Seasonal patterns |
| Linear Regression | ML | Simple trend forecasting |
| XGBoost | ML | Complex non-linear patterns |
| Prophet | ML (Meta) | Holiday effects, missing data |

---

## Data Format (for uploads)

Your CSV/Excel file should have at minimum:
- **Date column**: Any standard date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Sales column**: Numeric values (units, revenue, orders, etc.)

Example:
```csv
date,sales
2023-01-01,1250.5
2023-01-02,980.2
2023-01-03,1100.0
...
```

---

## Configuration (Sidebar)

- **Forecast Horizon**: 7 to 180 days ahead
- **Train/Test Split**: Controls how much data is used for training
- **MA Window**: Window size for Moving Average model
- **Confidence Interval**: Width of prediction uncertainty band
- **Residual Analysis**: Toggle diagnostic plots

---

## Metrics Explained

- **MAE** (Mean Absolute Error): Average absolute difference — lower is better
- **RMSE** (Root Mean Squared Error): Penalizes large errors more — lower is better
- **MAPE** (Mean Absolute Percentage Error): % error — lower is better
- **R²** (R-squared): Variance explained — closer to 1.0 is better

---

## Contributing

we can extend this project with:
- LSTM / Transformer models
- External regressors (weather, promotions)
- Multi-product / multi-store forecasting
- Real-time data integration

---

*Built using Python, Streamlit, Plotly, Statsmodels, XGBoost & Prophet*

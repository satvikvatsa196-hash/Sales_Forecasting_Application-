import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_sales_data(
    start_date: str = "2020-01-01",
    periods: int = 1000,
    freq: str = "D",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic sales data with:
    - Trend (linear + slight curve)
    - Weekly seasonality
    - Yearly seasonality
    - Holiday spikes
    - Random noise
    """
    np.random.seed(seed)

    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Base trend
    t = np.arange(periods)
    trend = 1000 + 0.5 * t + 0.0002 * t ** 2

    # Weekly seasonality (Mon-Sun)
    day_of_week = date_range.dayofweek
    weekly = np.where(day_of_week < 5, 1.2, 0.6)  # weekdays higher

    # Yearly seasonality using Fourier terms
    yearly_sin = 200 * np.sin(2 * np.pi * t / 365)
    yearly_cos = 100 * np.cos(2 * np.pi * t / 365)
    yearly_seasonality = yearly_sin + yearly_cos

    # Holiday effects (Black Friday, Christmas, etc.)
    holiday_effect = np.zeros(periods)
    for i, d in enumerate(date_range):
        # Black Friday (last Friday of November)
        if d.month == 11 and d.weekday() == 4 and d.day >= 23:
            holiday_effect[i] = 800
        # Christmas week
        elif d.month == 12 and d.day >= 20:
            holiday_effect[i] = 500
        # New Year bump
        elif d.month == 1 and d.day <= 5:
            holiday_effect[i] = 300
        # Summer sale (July)
        elif d.month == 7:
            holiday_effect[i] = 150

    # Noise
    noise = np.random.normal(0, 80, periods)

    # Combine
    sales = (trend * weekly + yearly_seasonality + holiday_effect + noise).clip(min=0)

    df = pd.DataFrame({
        "ds": date_range,
        "y": sales.round(2),
        "day_of_week": date_range.day_name(),
        "month": date_range.month_name(),
        "year": date_range.year,
        "quarter": date_range.quarter,
        "is_weekend": (date_range.dayofweek >= 5).astype(int),
    })

    # Add product categories
    categories = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports"]
    category_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    df["category"] = np.random.choice(categories, size=periods, p=category_weights)

    # Add revenue (sales * avg price)
    avg_prices = {"Electronics": 250, "Clothing": 65, "Food & Beverage": 25, "Home & Garden": 85, "Sports": 120}
    df["revenue"] = df.apply(lambda row: row["y"] * avg_prices[row["category"]] * np.random.uniform(0.9, 1.1), axis=1).round(2)

    # Add units sold
    df["units"] = (df["y"] * np.random.uniform(0.8, 1.2, periods)).round(0).astype(int)

    return df


def load_uploaded_data(file) -> pd.DataFrame:
    """Load and validate user-uploaded data."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")

        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def preprocess_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Preprocess data for forecasting."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.dropna(subset=["ds", "y"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    return df[["ds", "y"]]

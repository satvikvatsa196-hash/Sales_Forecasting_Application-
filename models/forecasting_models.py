import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute forecasting evaluation metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R² Score": round(r2, 4),
    }


class MovingAverageModel:
    """Simple Moving Average forecasting."""

    def __init__(self, window: int = 7):
        self.window = window
        self.name = f"Moving Average (window={window})"
        self.history = None

    def fit(self, df: pd.DataFrame):
        self.history = df["y"].values
        return self

    def predict(self, periods: int) -> np.ndarray:
        predictions = []
        history = list(self.history)
        for _ in range(periods):
            pred = np.mean(history[-self.window:])
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions)

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["y"].rolling(window=self.window, min_periods=1).mean().values


class ExponentialSmoothingModel:
    """Holt-Winters Exponential Smoothing."""

    def __init__(self, trend="add", seasonal="add", seasonal_periods=7):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.name = "Holt-Winters Exponential Smoothing"
        self.model = None

    def fit(self, df: pd.DataFrame):
        self.model = ExponentialSmoothing(
            df["y"].values,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit(optimized=True)
        self._n = len(df)
        return self

    def predict(self, periods: int) -> np.ndarray:
        return self.model.forecast(periods)

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.fittedvalues


class ARIMAModel:
    """ARIMA model for time series forecasting."""

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.name = f"ARIMA{order}"
        self.model = None

    def fit(self, df: pd.DataFrame):
        self.model = ARIMA(df["y"].values, order=self.order).fit()
        return self

    def predict(self, periods: int) -> np.ndarray:
        forecast = self.model.forecast(steps=periods)
        return forecast

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.fittedvalues


class SARIMAXModel:
    """SARIMA model with seasonal component."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = f"SARIMA{order}x{seasonal_order}"
        self.model = None

    def fit(self, df: pd.DataFrame):
        self.model = SARIMAX(
            df["y"].values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, periods: int) -> np.ndarray:
        return self.model.forecast(steps=periods)

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.fittedvalues


class ProphetModel:
    """Facebook Prophet model."""

    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.name = "Prophet"
        self.model = None
        self._df = None

    def fit(self, df: pd.DataFrame):
        try:
            from prophet import Prophet
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                interval_width=0.95,
            )
            self.model.fit(df[["ds", "y"]])
            self._df = df
        except ImportError:
            raise ImportError("Prophet is not installed. Run: pip install prophet")
        return self

    def predict(self, periods: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast["yhat"].tail(periods).values

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        forecast = self.model.predict(df[["ds"]])
        return forecast["yhat"].values

    def get_full_forecast(self, periods: int):
        future = self.model.make_future_dataframe(periods=periods)
        return self.model.predict(future)


class LinearRegressionModel:
    """Linear Regression with time-based features."""

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.name = "Linear Regression"
        self._last_t = None

    def _create_features(self, n, start=0):
        t = np.arange(start, start + n)
        X = np.column_stack([
            t,
            t ** 2,
            np.sin(2 * np.pi * t / 7),
            np.cos(2 * np.pi * t / 7),
            np.sin(2 * np.pi * t / 365),
            np.cos(2 * np.pi * t / 365),
        ])
        return X

    def fit(self, df: pd.DataFrame):
        n = len(df)
        X = self._create_features(n)
        self.model.fit(X, df["y"].values)
        self._last_t = n
        return self

    def predict(self, periods: int) -> np.ndarray:
        X = self._create_features(periods, start=self._last_t)
        return self.model.predict(X)

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._create_features(len(df))
        return self.model.predict(X)


class XGBoostModel:
    """XGBoost with lag features."""

    def __init__(self, n_estimators=200, max_depth=5, n_lags=14):
        import xgboost as xgb
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_lags = n_lags
        self.name = f"XGBoost (lags={n_lags})"
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        self._history = None

    def _create_features(self, y_array):
        X, targets = [], []
        for i in range(self.n_lags, len(y_array)):
            features = list(y_array[i - self.n_lags:i])
            features.append(i % 7)   # day of week
            features.append(i % 30)  # approx day of month
            features.append((i // 30) % 12)  # approx month
            X.append(features)
            targets.append(y_array[i])
        return np.array(X), np.array(targets)

    def fit(self, df: pd.DataFrame):
        y = df["y"].values
        self._history = y.copy()
        X, targets = self._create_features(y)
        self.model.fit(X, targets)
        return self

    def predict(self, periods: int) -> np.ndarray:
        history = list(self._history)
        predictions = []
        for i in range(periods):
            t = len(history)
            features = history[-self.n_lags:] + [t % 7, t % 30, (t // 30) % 12]
            pred = self.model.predict([features])[0]
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions)

    def in_sample_predict(self, df: pd.DataFrame) -> np.ndarray:
        y = df["y"].values
        X, _ = self._create_features(y)
        preds = self.model.predict(X)
        # Pad the front with actual values
        pad = np.full(self.n_lags, y[:self.n_lags].mean())
        return np.concatenate([pad, preds])


def get_model(name: str, params: dict = None):
    """Factory function to get model by name."""
    params = params or {}
    models = {
        "Moving Average": MovingAverageModel(**{k: v for k, v in params.items() if k == "window"}),
        "Exponential Smoothing": ExponentialSmoothingModel(),
        "ARIMA": ARIMAModel(),
        "SARIMA": SARIMAXModel(),
        "Prophet": ProphetModel(),
        "Linear Regression": LinearRegressionModel(),
        "XGBoost": XGBoostModel(),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name]

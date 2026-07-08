import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color palette
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "success": "#43D39E",
    "warning": "#FFB347",
    "info": "#45B8AC",
    "dark": "#1E1E2E",
    "surface": "#2A2A3E",
    "text": "#E0E0FF",
    "grid": "rgba(255,255,255,0.05)",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="'Rajdhani', sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
)


def plot_sales_overview(df: pd.DataFrame) -> go.Figure:
    """Plot the historical sales overview."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines",
        name="Historical Sales",
        line=dict(color=COLORS["primary"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(108, 99, 255, 0.1)",
    ))

    # Add rolling average
    rolling = df["y"].rolling(window=30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df["ds"], y=rolling,
        mode="lines",
        name="30-Day Moving Avg",
        line=dict(color=COLORS["warning"], width=2, dash="dash"),
    ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="📈 Historical Sales Overview", font=dict(size=18, color=COLORS["primary"])),
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        height=420,
    )
    return fig


def plot_forecast(df: pd.DataFrame, forecast: np.ndarray, future_dates: pd.DatetimeIndex,
                  model_name: str, confidence_interval: float = 0.15) -> go.Figure:
    """Plot forecast with confidence interval."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines",
        name="Historical",
        line=dict(color=COLORS["primary"], width=1.5),
    ))

    # Confidence interval
    upper = forecast * (1 + confidence_interval)
    lower = forecast * (1 - confidence_interval)

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(67, 211, 158, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        mode="lines+markers",
        name=f"{model_name} Forecast",
        line=dict(color=COLORS["success"], width=2.5),
        marker=dict(size=4),
    ))

    # Vertical line separating history and forecast
    fig.add_vline(
        x=df["ds"].iloc[-1].to_pydatetime(),
        line_dash="dot",
        line_color=COLORS["secondary"],
        
    )
    fig.add_annotation(
    x=df["ds"].iloc[-1].to_pydatetime(),
    y=1,
    yref="paper",
    text="Forecast Start",
    showarrow=False,
    font=dict(color=COLORS["secondary"])
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=f"🔮 Sales Forecast — {model_name}", font=dict(size=18, color=COLORS["success"])),
        xaxis_title="Date",
        yaxis_title="Predicted Sales",
        hovermode="x unified",
        height=450,
    )
    return fig


def plot_model_comparison(results: dict, df: pd.DataFrame) -> go.Figure:
    """Plot multiple model forecasts for comparison."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["ds"].tail(90), y=df["y"].tail(90),
        mode="lines",
        name="Historical (Last 90 Days)",
        line=dict(color=COLORS["primary"], width=2),
    ))

    color_list = [COLORS["success"], COLORS["warning"], COLORS["secondary"],
                  COLORS["info"], "#FF9FF3", "#FFC3A0", "#85C1E9"]

    for i, (model_name, (future_dates, forecast)) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            mode="lines",
            name=model_name,
            line=dict(color=color_list[i % len(color_list)], width=2),
        ))

    fig.add_vline(
        x=df["ds"].iloc[-1],
        line_dash="dot",
        line_color="rgba(255,255,255,0.4)",
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="🔬 Model Comparison", font=dict(size=18, color=COLORS["info"])),
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        height=450,
    )
    return fig


def plot_metrics_comparison(metrics_dict: dict) -> go.Figure:
    """Bar chart comparing metrics across models."""
    models = list(metrics_dict.keys())
    metric_names = ["MAE", "RMSE", "MAPE (%)"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metric_names)

    colors = [COLORS["primary"], COLORS["success"], COLORS["warning"],
              COLORS["secondary"], COLORS["info"], "#FF9FF3"]

    for col, metric in enumerate(metric_names, 1):
        values = [metrics_dict[m][metric] for m in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=[colors[i % len(colors)] for i in range(len(models))],
                showlegend=False,
            ),
            row=1, col=col,
        )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="📊 Model Performance Metrics", font=dict(size=18, color=COLORS["warning"])),
        height=380,
    )
    return fig


def plot_seasonality(df: pd.DataFrame) -> go.Figure:
    """Plot seasonality decomposition."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Sales by Day of Week", "Sales by Month",
                                        "Sales by Quarter", "Yearly Trend"])

    df = df.copy()
    df["dow"] = df["ds"].dt.day_name()
    df["month"] = df["ds"].dt.month_name()
    df["quarter"] = "Q" + df["ds"].dt.quarter.astype(str)
    df["year"] = df["ds"].dt.year

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = df.groupby("dow")["y"].mean().reindex(days_order)
    fig.add_trace(go.Bar(x=dow_avg.index, y=dow_avg.values,
                         marker_color=COLORS["primary"], name="Day of Week"), row=1, col=1)

    months_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    month_avg = df.groupby("month")["y"].mean().reindex(months_order)
    fig.add_trace(go.Bar(x=month_avg.index, y=month_avg.values,
                         marker_color=COLORS["success"], name="Month"), row=1, col=2)

    quarter_avg = df.groupby("quarter")["y"].mean()
    fig.add_trace(go.Bar(x=quarter_avg.index, y=quarter_avg.values,
                         marker_color=COLORS["warning"], name="Quarter"), row=2, col=1)

    yearly = df.groupby("year")["y"].mean()
    fig.add_trace(go.Scatter(x=yearly.index, y=yearly.values,
                             mode="lines+markers",
                             line=dict(color=COLORS["secondary"], width=2),
                             marker=dict(size=8),
                             name="Year"), row=2, col=2)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="📅 Seasonality Analysis", font=dict(size=18, color=COLORS["secondary"])),
        height=500,
        showlegend=False,
    )
    return fig


def plot_residuals(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> go.Figure:
    """Plot residual analysis."""
    residuals = actual - predicted

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals Over Time", "Residual Distribution"])

    fig.add_trace(
        go.Scatter(x=list(range(len(residuals))), y=residuals,
                   mode="lines", line=dict(color=COLORS["info"], width=1),
                   name="Residuals"),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["secondary"], row=1, col=1)

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=40,
                     marker_color=COLORS["primary"],
                     opacity=0.75, name="Distribution"),
        row=1, col=2
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=f"🔍 Residual Analysis — {model_name}", font=dict(size=18, color=COLORS["info"])),
        height=350,
        showlegend=False,
    )
    return fig

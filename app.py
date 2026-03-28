"""
╔══════════════════════════════════════════════════════════════╗
║          SALES FORECASTING SYSTEM — Streamlit App           ║
║          Powered by Time Series & ML Models                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import sys
import os

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 50%, #16213E 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132B 0%, #1E1E3A 100%);
    border-right: 1px solid rgba(108, 99, 255, 0.25);
}

/* ── Header Banner ── */
.app-header {
    background: linear-gradient(90deg, rgba(108,99,255,0.15) 0%, rgba(67,211,158,0.08) 100%);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #E0E0FF;
    margin: 0 0 0.3rem 0;
    letter-spacing: 0.5px;
}
.app-header p {
    color: rgba(224, 224, 255, 0.6);
    font-size: 1.1rem;
    margin: 0;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(108, 99, 255, 0.5);
    background: rgba(108, 99, 255, 0.07);
}
.metric-label {
    font-size: 0.85rem;
    color: rgba(224, 224, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #6C63FF;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta {
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* ── Section Titles ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #E0E0FF;
    padding: 0.5rem 0;
    border-bottom: 2px solid rgba(108, 99, 255, 0.3);
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

/* ── Model Badge ── */
.model-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(108,99,255,0.3), rgba(67,211,158,0.2));
    border: 1px solid rgba(108, 99, 255, 0.4);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.85rem;
    color: #E0E0FF;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 0.2rem;
}

/* ── Info Box ── */
.info-box {
    background: rgba(69, 184, 172, 0.08);
    border-left: 3px solid #45B8AC;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    color: rgba(224, 224, 255, 0.85);
    font-size: 0.95rem;
}

/* ── Warning Box ── */
.warning-box {
    background: rgba(255, 179, 71, 0.08);
    border-left: 3px solid #FFB347;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    color: rgba(224, 224, 255, 0.85);
}

/* ── Forecast Table ── */
.forecast-table-header {
    background: linear-gradient(90deg, rgba(108,99,255,0.2), transparent);
    padding: 0.6rem 1rem;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    color: #6C63FF;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
}

/* ── Streamlit Overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #43D39E) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
}

.stSelectbox label, .stSlider label, .stCheckbox label,
.stMultiSelect label, .stRadio label, .stNumberInput label {
    color: rgba(224, 224, 255, 0.7) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: rgba(224, 224, 255, 0.6) !important;
}
.stTabs [aria-selected="true"] {
    color: #6C63FF !important;
}

div[data-testid="stMetricValue"] {
    color: #43D39E !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D0D1A; }
::-webkit-scrollbar-thumb { background: rgba(108, 99, 255, 0.5); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Imports ────────────────────────────────────────────────────────────────
from utils.data_generator import generate_sample_sales_data, preprocess_data
from utils.visualizations import (
    plot_sales_overview, plot_forecast, plot_model_comparison,
    plot_metrics_comparison, plot_seasonality, plot_residuals
)
from models.forecasting_models import get_model, compute_metrics


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>📈 Sales Forecasting System</h1>
    <p>Intelligent time series forecasting powered by statistical & machine learning models</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Data Source
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["🎲 Generate Sample Data", "📤 Upload Your Data"],
        label_visibility="collapsed"
    )

    uploaded_df = None
    date_col, target_col = "ds", "y"

    if "Upload" in data_source:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="File must contain a date column and a numeric sales column."
        )
        if uploaded_file:
            try:
                from utils.data_generator import load_uploaded_data
                raw_df = load_uploaded_data(uploaded_file)
                st.success(f"✅ Loaded {len(raw_df):,} rows")
                cols = raw_df.columns.tolist()
                date_col = st.selectbox("📅 Date Column", cols)
                target_col = st.selectbox("💰 Target Column (Sales)", cols)
                uploaded_df = raw_df
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown("##### Sample Data Settings")
        n_days = st.slider("Days of history", 365, 1460, 730, 30)
        st.caption("Synthetic sales with trend, seasonality & noise.")

    st.markdown("---")

    # Forecast Settings
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Days)", 7, 180, 30)
    train_split = st.slider("Train/Test Split (%)", 70, 95, 80)

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 Model Selection")
    available_models = [
        "Moving Average",
        "Exponential Smoothing",
        "ARIMA",
        "SARIMA",
        "Linear Regression",
        "XGBoost",
        "Prophet",
    ]

    selected_models = st.multiselect(
        "Select Models to Run",
        available_models,
        default=["Moving Average", "Exponential Smoothing", "ARIMA", "Linear Regression"],
        help="Select one or more models to compare."
    )

    st.markdown("---")

    # Advanced Options
    with st.expander("🔧 Advanced Options"):
        ma_window = st.slider("MA Window Size", 3, 30, 7)
        conf_interval = st.slider("Confidence Interval (%)", 5, 30, 15)
        show_residuals = st.checkbox("Show Residual Analysis", True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

    st.markdown("""
    <div style='margin-top: 2rem; padding: 1rem; background: rgba(108,99,255,0.08);
    border-radius: 10px; font-size: 0.8rem; color: rgba(224,224,255,0.5);'>
    <b style='color: rgba(224,224,255,0.7);'>Models Available:</b><br>
    📊 Moving Average<br>
    📉 Exp. Smoothing<br>
    🔢 ARIMA / SARIMA<br>
    📐 Linear Regression<br>
    🌲 XGBoost<br>
    🔮 Prophet
    </div>
    """, unsafe_allow_html=True)


# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def get_sample_data(n_days):
    return generate_sample_sales_data(periods=n_days)

if "Upload" in data_source and uploaded_df is not None:
    try:
        df = preprocess_data(uploaded_df, date_col, target_col)
    except Exception as e:
        st.error(f"Data preprocessing error: {e}")
        st.stop()
elif "Upload" in data_source:
    st.markdown("""
    <div class="info-box">
        📤 Please upload a CSV or Excel file from the sidebar to get started.
        Your file should contain at least a <b>date column</b> and a <b>sales/target column</b>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
else:
    df = get_sample_data(n_days)
    df = df[["ds", "y"]]


# ─── KPI Row ─────────────────────────────────────────────────────────────────
total_sales = df["y"].sum()
avg_daily = df["y"].mean()
max_day = df["y"].max()
growth = ((df["y"].iloc[-30:].mean() - df["y"].iloc[:30].mean()) / df["y"].iloc[:30].mean()) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Total Records", f"{len(df):,}", delta=None)
with col2:
    st.metric("💰 Total Sales", f"{total_sales:,.0f}", delta=None)
with col3:
    st.metric("📊 Avg Daily Sales", f"{avg_daily:,.1f}")
with col4:
    st.metric("📈 Overall Growth", f"{growth:+.1f}%",
              delta="vs First 30 Days",
              delta_color="normal")

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "🔬 Model Comparison",
    "📅 Seasonality",
    "📋 Data Table",
])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Historical Sales Overview</div>', unsafe_allow_html=True)
    fig_overview = plot_sales_overview(df)
    st.plotly_chart(fig_overview, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
        stats = df["y"].describe().round(2)
        stats_df = pd.DataFrame({
            "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
            "Value": [
                f"{int(stats['count']):,}",
                f"{stats['mean']:,.2f}",
                f"{stats['std']:,.2f}",
                f"{stats['min']:,.2f}",
                f"{stats['25%']:,.2f}",
                f"{stats['50%']:,.2f}",
                f"{stats['75%']:,.2f}",
                f"{stats['max']:,.2f}",
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-title">📅 Date Range Info</div>', unsafe_allow_html=True)
        date_info = pd.DataFrame({
            "Property": ["Start Date", "End Date", "Total Days", "Date Frequency",
                         "Missing Dates", "Zero Sales Days"],
            "Value": [
                str(df["ds"].min().date()),
                str(df["ds"].max().date()),
                str((df["ds"].max() - df["ds"].min()).days + 1),
                pd.infer_freq(df["ds"]) or "Unknown",
                str(((df["ds"].max() - df["ds"].min()).days + 1) - len(df)),
                str((df["y"] == 0).sum()),
            ]
        })
        st.dataframe(date_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════
with tab2:
    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar.")
    else:
        if not run_btn:
            st.markdown("""
            <div class="info-box">
                🚀 Configure your forecast settings in the sidebar and click <b>Run Forecast</b> to begin.
                You can select multiple models, adjust the horizon, and more.
            </div>
            """, unsafe_allow_html=True)

        if run_btn:
            # Train/test split
            split_idx = int(len(df) * train_split / 100)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            future_dates = pd.date_range(
                start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq="D"
            )

            primary_model_name = selected_models[0]

            with st.spinner(f"⏳ Training {primary_model_name}..."):
                try:
                    params = {"window": ma_window} if primary_model_name == "Moving Average" else {}
                    model = get_model(primary_model_name, params)
                    model.fit(train_df)
                    forecast = model.predict(forecast_horizon)
                    forecast = np.clip(forecast, 0, None)

                    # Metrics on test set
                    if len(test_df) > 0:
                        test_pred_count = min(len(test_df), 60)
                        test_model = get_model(primary_model_name, params)
                        test_model.fit(train_df)
                        in_sample = test_model.in_sample_predict(train_df)
                        metrics = compute_metrics(
                            train_df["y"].values[-test_pred_count:],
                            in_sample[-test_pred_count:]
                        )
                    else:
                        metrics = {}

                    # Plot
                    fig_forecast = plot_forecast(df, forecast, future_dates,
                                                 primary_model_name, conf_interval / 100)
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Metrics display
                    if metrics:
                        st.markdown('<div class="section-title">📐 Model Performance</div>', unsafe_allow_html=True)
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("MAE", f"{metrics['MAE']:,.2f}")
                        mc2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
                        mc3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
                        mc4.metric("R² Score", f"{metrics['R² Score']:.4f}")

                    # Forecast Table
                    st.markdown('<div class="section-title">📋 Forecast Values</div>', unsafe_allow_html=True)
                    forecast_df = pd.DataFrame({
                        "Date": future_dates.strftime("%Y-%m-%d"),
                        "Predicted Sales": forecast.round(2),
                        "Lower Bound": (forecast * (1 - conf_interval / 100)).round(2),
                        "Upper Bound": (forecast * (1 + conf_interval / 100)).round(2),
                        "Day of Week": future_dates.day_name(),
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                    # Download
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Forecast CSV",
                        data=csv,
                        file_name=f"forecast_{primary_model_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )

                    # Residuals
                    if show_residuals:
                        st.markdown('<div class="section-title">🔍 Residual Analysis</div>', unsafe_allow_html=True)
                        in_sample_preds = model.in_sample_predict(train_df)
                        min_len = min(len(train_df["y"].values), len(in_sample_preds))
                        fig_res = plot_residuals(
                            train_df["y"].values[-min_len:],
                            in_sample_preds[-min_len:],
                            primary_model_name
                        )
                        st.plotly_chart(fig_res, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Model error ({primary_model_name}): {e}")
                    import traceback
                    st.expander("🔍 Error Details").code(traceback.format_exc())


# ══════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
    if not run_btn:
        st.markdown("""
        <div class="info-box">
            🔬 Click <b>Run Forecast</b> in the sidebar to compare all selected models side by side.
        </div>
        """, unsafe_allow_html=True)

    if run_btn and selected_models:
        st.markdown('<div class="section-title">🔬 Multi-Model Comparison</div>', unsafe_allow_html=True)

        split_idx = int(len(df) * train_split / 100)
        train_df = df.iloc[:split_idx]
        future_dates = pd.date_range(
            start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=forecast_horizon, freq="D"
        )

        comparison_results = {}
        all_metrics = {}
        progress = st.progress(0, text="Training models...")

        for i, model_name in enumerate(selected_models):
            progress.progress((i + 1) / len(selected_models), text=f"Training {model_name}...")
            try:
                params = {"window": ma_window} if model_name == "Moving Average" else {}
                m = get_model(model_name, params)
                m.fit(train_df)
                fc = np.clip(m.predict(forecast_horizon), 0, None)
                comparison_results[model_name] = (future_dates, fc)

                in_s = m.in_sample_predict(train_df)
                n = min(len(train_df["y"]), len(in_s))
                all_metrics[model_name] = compute_metrics(
                    train_df["y"].values[-n:], in_s[-n:]
                )
            except Exception as e:
                st.warning(f"⚠️ {model_name} failed: {e}")

        progress.empty()

        if comparison_results:
            fig_comp = plot_model_comparison(comparison_results, df)
            st.plotly_chart(fig_comp, use_container_width=True)

            if len(all_metrics) > 1:
                fig_metrics = plot_metrics_comparison(all_metrics)
                st.plotly_chart(fig_metrics, use_container_width=True)

            # Metrics table
            st.markdown('<div class="section-title">📊 Performance Summary Table</div>', unsafe_allow_html=True)
            metrics_table = pd.DataFrame(all_metrics).T.reset_index()
            metrics_table.columns = ["Model", "MAE", "RMSE", "MAPE (%)", "R² Score"]

            # Highlight best model
            best_model = metrics_table.loc[metrics_table["MAE"].idxmin(), "Model"]

            st.dataframe(
                metrics_table.style.highlight_min(
                    subset=["MAE", "RMSE", "MAPE (%)"],
                    color="rgba(67, 211, 158, 0.3)"
                ).highlight_max(
                    subset=["R² Score"],
                    color="rgba(67, 211, 158, 0.3)"
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(f"""
            <div class="info-box">
                🏆 <b>Best Performing Model (lowest MAE):</b> <span class="model-badge">{best_model}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — SEASONALITY
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📅 Seasonality & Trend Analysis</div>', unsafe_allow_html=True)
    fig_season = plot_seasonality(df)
    st.plotly_chart(fig_season, use_container_width=True)

    # Rolling stats
    st.markdown('<div class="section-title">📈 Rolling Statistics</div>', unsafe_allow_html=True)
    import plotly.graph_objects as go
    roll_fig = go.Figure()
    roll_fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines",
                                   name="Raw Sales", line=dict(color="rgba(108,99,255,0.4)", width=1)))
    for window, color in [(7, "#43D39E"), (30, "#FFB347"), (90, "#FF6584")]:
        roll = df["y"].rolling(window).mean()
        roll_fig.add_trace(go.Scatter(x=df["ds"], y=roll, mode="lines",
                                       name=f"{window}-Day MA", line=dict(color=color, width=2)))
    roll_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0FF", family="Rajdhani"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=350,
        title=dict(text="Moving Averages Overlay", font=dict(size=16, color="#E0E0FF")),
        hovermode="x unified",
    )
    st.plotly_chart(roll_fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — DATA TABLE
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">📋 Raw Data Preview</div>', unsafe_allow_html=True)

    display_df = df.copy()
    display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m-%d")
    display_df["y"] = display_df["y"].round(2)
    display_df.columns = ["Date", "Sales"]

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search = st.text_input("🔍 Search by date", "")
    with col_f2:
        page_size = st.selectbox("Rows per page", [20, 50, 100, 500], index=0)
    with col_f3:
        sort_col = st.selectbox("Sort by", ["Date", "Sales"])

    if search:
        display_df = display_df[display_df["Date"].str.contains(search)]

    display_df = display_df.sort_values(sort_col, ascending=(sort_col == "Date"))

    st.dataframe(display_df.head(page_size), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(page_size, len(display_df))} of {len(display_df):,} records")

    csv = display_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Full Dataset",
        data=csv,
        file_name="sales_data.csv",
        mime="text/csv",
    )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(224,224,255,0.3); font-size: 0.85rem; padding: 1rem 0;'>
    📈 Sales Forecasting System &nbsp;|&nbsp; Models: MA · ES · ARIMA · SARIMA · LR · XGBoost · Prophet
</div>
""", unsafe_allow_html=True)

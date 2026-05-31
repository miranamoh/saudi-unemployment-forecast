import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Saudi Unemployment Forecasting",
    layout="centered"
)
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1f3a;
        color: white;
    }
    
    .stMetric {
        background-color: #252b4a;
        border-radius: 10px;
        padding: 10px;
    }
    
    h1, h2, h3 {
        color: #5b9bd5 !important;
    }
    
    .stDataFrame {
        background-color: #252b4a;
    }
    
    [data-testid="stExpander"] {
        background-color: #252b4a;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return joblib.load("sarima_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# ==================== HEADER ====================
st.title("Saudi Unemployment Rate Forecasting ")
st.markdown("---")

# DATA (from results.json) 
import json

with open('results.json', 'r') as f:
    results = json.load(f)

mae            = results['mae']
historical     = {
    "Quarter": results['historical_quarters'],
    "Actual Rate (%)": results['historical_rates']
}
df_hist        = pd.DataFrame(historical)
actual_test    = results['actual_test']
predicted_test = results['predicted_test']
test_quarters  = results['test_quarters']

# ==================== FORECAST ====================
st.subheader("National Unemployment Rate Forecast")

if not model_loaded:
    st.error("Model file not found. Please ensure sarima_model.pkl is in the app directory.")
    st.stop()


forecast_values   = model.forecast(steps=2)
forecast_quarters = ["2026-Q1", "2026-Q2"]

forecast_data = []
for i in range(2):
    pred = max(0, forecast_values[i])
    forecast_data.append({
        "Quarter": forecast_quarters[i],
        "Predicted Rate (%)": round(pred, 2),
        "Lower Bound (%)": round(max(0, pred - mae), 2),
        "Upper Bound (%)": round(pred + mae, 2)
    })
df_forecast = pd.DataFrame(forecast_data)

# ── Metrics ──
col1, col2 = st.columns(2)
with col1:
    delta1 = df_forecast.iloc[0]["Predicted Rate (%)"] - 7.24
    st.metric(
        label="2026-Q1 Forecast",
        value=f"{df_forecast.iloc[0]['Predicted Rate (%)']:.2f}%",
        delta=f"{delta1:+.2f} percentage points vs 2025-Q4"
    )
    st.caption(f"Confidence interval: [{df_forecast.iloc[0]['Lower Bound (%)']:.2f}% — {df_forecast.iloc[0]['Upper Bound (%)']:.2f}%]")

with col2:
    delta2 = df_forecast.iloc[1]["Predicted Rate (%)"] - df_forecast.iloc[0]["Predicted Rate (%)"]
    st.metric(
        label="2026-Q2 Forecast",
        value=f"{df_forecast.iloc[1]['Predicted Rate (%)']:.2f}%",
        delta=f"{delta2:+.2f} percentage points vs 2026-Q1"
    )
    st.caption(f"Confidence interval: [{df_forecast.iloc[1]['Lower Bound (%)']:.2f}% — {df_forecast.iloc[1]['Upper Bound (%)']:.2f}%]")

st.markdown("---")

# ==================== CHART 1: Historical + Forecast ====================
st.subheader(" Historical Trend and Forecast")

fig1, ax1 = plt.subplots(figsize=(11, 5))

ax1.plot(df_hist["Quarter"], df_hist["Actual Rate (%)"],
         color="#1f4e79", marker="o", linewidth=2, markersize=6, label="Historical (Actual)")

connect_q  = [df_hist["Quarter"].iloc[-1]] + forecast_quarters
connect_v  = [df_hist["Actual Rate (%)"].iloc[-1]] + [df_forecast.iloc[i]["Predicted Rate (%)"] for i in range(2)]
connect_lo = [df_hist["Actual Rate (%)"].iloc[-1]] + [df_forecast.iloc[i]["Lower Bound (%)"] for i in range(2)]
connect_hi = [df_hist["Actual Rate (%)"].iloc[-1]] + [df_forecast.iloc[i]["Upper Bound (%)"] for i in range(2)]

ax1.plot(connect_q, connect_v,
         color="#2e75b6", marker="s", linewidth=2, markersize=7, linestyle="--", label="Forecast (SARIMA)")
ax1.fill_between(connect_q, connect_lo, connect_hi,
                 alpha=0.15, color="#2e75b6", label=f"Confidence Interval (+/- {mae:.4f} pp)")
ax1.axhline(y=7.0, color="green", linestyle=":", linewidth=1.5, alpha=0.8, label="Vision 2030 Target (7%)")
ax1.axvline(x=11.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
ax1.text(11.6, 11.5, "Forecast", fontsize=8, color="gray")

for i in range(2):
    ax1.annotate(
        f"{df_forecast.iloc[i]['Predicted Rate (%)']:.2f}%",
        xy=(forecast_quarters[i], df_forecast.iloc[i]["Predicted Rate (%)"]),
        xytext=(0, 12), textcoords="offset points",
        ha="center", fontsize=9, fontweight="bold", color="#2e75b6"
    )

ax1.set_ylabel("Unemployment Rate (%)", fontsize=11)
ax1.set_xlabel("Quarter", fontsize=11)
ax1.tick_params(axis="x", rotation=45)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(4, 13)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# ==================== CHART 2A: Vision 2030 Comparison ====================
st.subheader("Forecast vs Vision 2030 Target")

vision_target = 7.0
all_quarters  = forecast_quarters
all_predicted = [df_forecast.iloc[i]["Predicted Rate (%)"] for i in range(2)]
all_lower     = [df_forecast.iloc[i]["Lower Bound (%)"] for i in range(2)]
all_upper     = [df_forecast.iloc[i]["Upper Bound (%)"] for i in range(2)]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Forecast vs Vision 2030 Target", fontsize=13, fontweight="bold")

# ── Bar chart ──
bar_colors = ["#c0392b" if v > vision_target else "#27ae60" for v in all_predicted]
bars = ax2a.bar(all_quarters, all_predicted, color=bar_colors, width=0.4, zorder=3)
ax2a.axhline(y=vision_target, color="green", linestyle="--", linewidth=2, label="Vision 2030 Target (7%)")
ax2a.errorbar(all_quarters, all_predicted,
              yerr=[[p - lo for p, lo in zip(all_predicted, all_lower)],
                    [hi - p  for p, hi in zip(all_predicted, all_upper)]],
              fmt="none", color="gray", capsize=6, linewidth=1.5)

for bar, val in zip(bars, all_predicted):
    diff = val - vision_target
    sign = "+" if diff > 0 else ""
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f"{val:.2f}%\n({sign}{diff:.2f} pp)", ha="center", fontsize=9, fontweight="bold")

ax2a.set_ylabel("Unemployment Rate (%)")
ax2a.set_title("Predicted Rate vs Target")
ax2a.set_ylim(0, 11)
ax2a.legend(fontsize=9)
ax2a.grid(axis="y", alpha=0.3, zorder=0)

# ── Gauge-style text summary ──
ax2b.axis("off")
for i, (q, pred) in enumerate(zip(all_quarters, all_predicted)):
    diff   = pred - vision_target
    status = "Above Target" if diff > 0 else "Below Target"
    color  = "#c0392b" if diff > 0 else "#27ae60"
    y      = 0.75 - i * 0.40

    ax2b.text(0.5, y + 0.12, q, ha="center", fontsize=13, fontweight="bold",
              transform=ax2b.transAxes)
    ax2b.text(0.5, y - 0.01, f"Forecast: {pred:.2f}%", ha="center", fontsize=11,
              transform=ax2b.transAxes)
    ax2b.text(0.5, y - 0.12, f"{status}  ({'+' if diff > 0 else ''}{diff:.2f} pp)",
              ha="center", fontsize=11, color=color, fontweight="bold",
              transform=ax2b.transAxes)

ax2b.text(0.5, 0.05, f"Vision 2030 Target: {vision_target}%",
          ha="center", fontsize=10, color="green", style="italic",
          transform=ax2b.transAxes)
ax2b.set_title("Status vs Target")

plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# ==================== CHART 3: Actual vs Predicted ====================
st.subheader("Model Accuracy: Actual vs Predicted (Test Set 2023–2025)")

fig3, ax3 = plt.subplots(figsize=(11, 5))

ax3.plot(test_quarters, actual_test,
         color="#1f4e79", marker="o", linewidth=2, markersize=6, label="Actual")
ax3.plot(test_quarters, predicted_test,
         color="#e74c3c", marker="s", linewidth=2, markersize=6, linestyle="--", label="Predicted (SARIMA)")

for i, (a, p) in enumerate(zip(actual_test, predicted_test)):
    ax3.plot([test_quarters[i], test_quarters[i]], [a, p],
             color="gray", linewidth=1, linestyle=":", alpha=0.6)

ax3.axhline(y=7.0, color="green", linestyle=":", linewidth=1.5, alpha=0.7, label="Vision 2030 Target (7%)")
ax3.set_ylabel("Unemployment Rate (%)", fontsize=11)
ax3.set_xlabel("Quarter", fontsize=11)
ax3.tick_params(axis="x", rotation=45)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.set_ylim(4, 12)

ax3.text(0.5, 0.96, f"MAE = {results['mae']:.4f} pp  |  RMSE = {results['rmse']:.4f}  |  R2 = {results['r2']:.4f}",
         transform=ax3.transAxes, ha="center", fontsize=9,
         style="italic", color="gray",
         bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="lightgray"))

plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")

# ==================== FORECAST TABLE ====================
st.subheader("Forecast Details")
st.dataframe(df_forecast, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== MODEL INFO ====================
with st.expander("Model Information"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "SARIMA")
    col2.metric("MAE", f"{results['mae']:.4f} pp")
    col3.metric("RMSE", f"{results['rmse']:.4f}")
    col4.metric("R2", f"{results['r2']:.4f}")
    st.write("**Training Period:** Q2 2016 — Q2 2023 (27 quarters)")
    st.write("**Test Period:** Q3 2023 — Q4 2025 (12 quarters)")
    st.write("**SARIMA Order:** (1,1,1)(1,1,1,4)")
# ==================== DISCLAIMER ====================
st.warning(
    "Disclaimer: These forecasts are statistical projections generated by a SARIMA "
    "time-series model trained on historical data. They are not guaranteed outcomes. "
    "All results should be interpreted as estimates to support decision-making."
)

st.caption("Data source: DataSaudi — Ministry of Economy and Planning of Saudi Arabia")

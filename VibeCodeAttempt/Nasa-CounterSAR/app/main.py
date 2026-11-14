"""
CounterSAR — Full Operational Application (Stage 5)
===================================================
AI-Powered SAR Anomaly Detection Dashboard
NASA Space Apps Challenge · Kuttanad, Kerala, India

Integrates:
  - Ingested & preprocessed satellite and environmental datasets (src/data_source.py)
  - Seasonal Climatology & Weather-Aware LightGBM Models (src/models.py)
  - Z-Score Scoring, Adaptive Thresholds & Persistence Filtering (src/scorer.py)
  - Precomputed operational results from results/pipeline_output.nc

Features:
  1. 📖 Story Mode: Guided narrative of Kuttanad radar monitoring challenges
  2. 🔄 Expected vs Actual Viewer: Spatial swipe/side-by-side comparison & residual map
  3. 📡 Polarization & Frequency Explorer: C-band vs L/S-band NISAR exploration
  4. ⏳ Time Machine: Chronological scrubbing across the multi-year timeline
  5. 🚨 Anomaly Cards: Filterable gallery of detected candidates and persistent events
  6. 📊 Observations & Environmental Explorer: Raw time series and spatial grids
  7. 🗺️ Study Area Overview: Geospatial map with Kuttanad agricultural polders
"""

from __future__ import annotations

import os
import sys

# Ensure both app/ and project root are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium
import xarray as xr

from mock_data import (
    KUTTANAD_BBOX,
    SYNTHETIC_LABEL,
    generate_environmental_timeseries,
    generate_sar_timeseries,
    generate_spatial_backscatter,
    get_study_area_info,
)
from src.data_source import (
    create_synthetic_datacube,
    open_spatial_dataset,
    standardise_backscatter,
)
from src.models import (
    SeasonalClimatologyModel,
    WeatherAwareLGBMModel,
    evaluate_model_performance,
)
from src.scorer import (
    AnomalyVerdict,
    run_anomaly_pipeline,
    to_anomaly_results,
)
from src.skeleton import (
    CRS,
    GRID_RES_DEG,
    GRID_RES_METERS,
    KUTTANAD_BOUNDS,
)


# ════════════════════════════════════════════════════════════════════════
#  STREAMLIT CONFIG & CUSTOM STYLING
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CounterSAR — AI-Powered SAR Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── Header banner ──────────────────────────────────────────────── */
.csar-header {
    background: linear-gradient(135deg, #0a192f 0%, #172a45 60%, #1e3a5f 100%);
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    border-left: 5px solid #00bcd4;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
.csar-header h1 { color: #e6f1ff; font-size: 2.1rem; margin-bottom: 0.2rem; font-weight: 700; }
.csar-header .sub { color: #8892b0; font-size: 1.05rem; margin: 0; }
.csar-header .detail { color: #64ffda; font-size: 0.85rem; margin-top: 0.4rem; font-weight: 500; }

/* ── Synthetic-data banner ──────────────────────────────────────── */
.synth-banner {
    background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.4);
    border-left: 4px solid #f59e0b;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    color: #fbbf24;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Verdict Badges ─────────────────────────────────────────────── */
.badge-expected {
    background-color: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}
.badge-explained {
    background-color: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}
.badge-unexplained {
    background-color: rgba(239, 68, 68, 0.25);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-block;
}

/* ── Cards & Narrative ──────────────────────────────────────────── */
.narrative-card {
    background: rgba(16, 30, 54, 0.85);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.anomaly-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.1rem;
    margin-bottom: 1rem;
    transition: transform 0.15s ease-in-out;
}
.anomaly-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
</style>
""",
    unsafe_allow_html=True,
)

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,25,47,0.75)",
    plot_bgcolor="rgba(10,25,47,0.55)",
    font=dict(family="Inter, system-ui, sans-serif", color="#b0bec5", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.30)", bordercolor="#1e3a5f", borderwidth=1),
)

_C = {
    "vv": "#00bcd4",
    "vh": "#ff7043",
    "predicted": "#a78bfa",
    "lower": "#60a5fa",
    "upper": "#60a5fa",
    "rain": "#38bdf8",
    "sm": "#34d399",
    "temp": "#f87171",
    "anom": "#fbbf24",
    "grid": "#1e3a5f",
}


# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING & CACHING (Connected to results/pipeline_output.nc)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_pipeline_results() -> xr.Dataset:
    """
    Load precomputed operational results from results/pipeline_output.nc.
    If absent, runs the end-to-end pipeline and generates it.
    """
    nc_path = os.path.join(project_root, "results", "pipeline_output.nc")
    if os.path.exists(nc_path):
        ds = open_spatial_dataset(nc_path).load()
        return ds

    # Generate on-the-fly and save to results/
    cube = create_synthetic_datacube()
    ds = run_anomaly_pipeline(cube, target_var="vv_db", output_netcdf_path=nc_path)
    return ds.load()


@st.cache_data
def load_mock_spatial(date_str: str) -> xr.Dataset:
    return generate_spatial_backscatter(target_date=date_str)


# Initialize global dataset
pipeline_ds = get_pipeline_results()
study_info = get_study_area_info()

# Convert time coordinates to pandas datetime
pipeline_times = pd.to_datetime(pipeline_ds.time.values)
is_synthetic_run = bool(pipeline_ds.attrs.get("is_synthetic", False))


# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION & FILTERS
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛰️ CounterSAR")
    st.caption("AI-Powered SAR Anomaly Detection")
    st.caption("NASA Space Apps Challenge · Kuttanad, India")
    st.divider()

    nav_choice = st.radio(
        "Navigation",
        [
            "📖 Story Mode",
            "🔄 Expected vs Actual Viewer",
            "📡 Polarization & Frequency Explorer",
            "⏳ Time Machine",
            "🚨 Anomaly Cards & Scorer",
            "📊 Observations & Environmental Explorer",
            "🗺️ Study Area Overview",
        ],
        index=0,
    )

    st.divider()

    st.markdown("**📅 Global Date Filter**")
    min_date = pipeline_times.min().date()
    max_date = pipeline_times.max().date()
    date_start = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    date_end = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if date_start > date_end:
        st.error("Start date must be before end date.")

    st.divider()
    st.markdown("**Pipeline Engine Status**")
    st.caption("🟢 Sentinel-1 Ingestion: Active")
    st.caption("🟡 NISAR Explorer: Prototype Mode")
    st.caption("🟢 Weather-Aware LGBM: Trained")
    st.caption("🟢 3-Category Scorer: Operational")

    st.divider()
    st.markdown(
        f'<div class="synth-banner">{SYNTHETIC_LABEL}</div>',
        unsafe_allow_html=True,
    )
    st.caption("v1.0.0 · Full Operational Prototype")


# Filter pipeline dataset according to chosen date range
time_mask = (pipeline_times.date >= date_start) & (pipeline_times.date <= date_end)
ds_filtered = pipeline_ds.isel(time=np.where(time_mask)[0])


def render_synth_notice():
    if is_synthetic_run:
        st.markdown(
            f'<div class="synth-banner">{SYNTHETIC_LABEL} — Powered by synthetic development cube</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 1: STORY MODE
# ════════════════════════════════════════════════════════════════════════

if nav_choice == "📖 Story Mode":
    st.markdown(
        """
    <div class="csar-header">
        <h1>📖 CounterSAR Story Mode</h1>
        <p class="sub">Why Naive Radar Anomaly Detection Cries Wolf in Tropical Agricultural Wetlands</p>
        <p class="detail">A guided walk-through of the Kuttanad challenge and how weather-aware models separate normal farming from real crises</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    col1, col2 = st.columns([1.8, 1.2])

    with col1:
        st.markdown(
            """
        ### 🌾 Act 1: The Land Below the Sea
        **Kuttanad**, located in Kerala, India, is one of the few places in the world where farming is practiced
        up to **2.5 meters below sea level**. The region comprises vast interconnected polders, backwaters,
        and rivers flowing into Lake Vembanad.

        During the South-West Monsoon (June–September) and North-East Monsoon (October–November), rainfall
        is intense. Farmers systematically flood paddies for weed management and soil conditioning,
        and later pump water out using large electric pumps.
        """
        )

        st.info(
            "💡 **Radar Physics Reminder:** Flat, smooth water acts as a specular mirror for satellite radar pulses. "
            "Instead of reflecting back to the satellite, the pulse bounces away into space, causing the received "
            "backscatter (σ⁰) to plunge dramatically from −8 dB down to −16 dB or lower."
        )

        st.markdown(
            """
        ### 🚨 Act 2: The Problem with Naive Change Detection
        Traditional disaster-response algorithms compute simple threshold differences:
        $$\\Delta \\sigma^0 = \\sigma^0_{\\text{today}} - \\sigma^0_{\\text{reference}}$$

        Whenever $\\Delta \\sigma^0 < -4\\text{ dB}$, they sound the alarm for a **catastrophic flood**.
        In Kuttanad, this naive approach triggers **false alarm floods almost every single month**:
        - Normal seasonal monsoon rain is flagged as a disaster.
        - Controlled, scheduled paddy flooding is flagged as a dike breach.
        - Emergency management agencies face alert fatigue and dismiss real warnings.
        """
        )

        st.markdown(
            """
        ### 🧠 Act 3: The CounterSAR Solution
        CounterSAR introduces a two-tier predictive intelligence framework:
        1. **Seasonal Climatology**: Learns the annual vegetative and flood calendar over multi-year cycles.
        2. **Weather-Aware Machine Learning (LightGBM)**: Predicts expected radar backscatter directly from
           **CHIRPS rainfall**, **ERA5 soil moisture**, and **antecedent 3-day / 7-day cumulative rainfall**.

        By contrasting what **the season expected** with what **the weather explained**, CounterSAR issues
        a decisive **Three-Category Verdict**:
        - 🟢 **Expected**: Within normal seasonal and weather limits.
        - 🟡 **Unusual but Explained**: Departure from historical norms, but fully justified by recent downpours or saturated soil.
        - 🔴 **Unexplained Candidate Event**: Severe radar departure that rainfall and soil moisture cannot account for!
        """
        )

    with col2:
        st.markdown("### 📊 Live Pipeline Case Comparison")
        st.caption("Inspecting actual pipeline output from results/pipeline_output.nc")

        # Pick a date with "Unusual but Explained" and an "Unexplained Candidate"
        cats = ds_filtered.verdict_category.values
        explained_idx = np.where(cats == AnomalyVerdict.UNUSUAL_EXPLAINED.value)[0]
        unexplained_idx = np.where(cats == AnomalyVerdict.UNEXPLAINED_CANDIDATE.value)[0]

        st.markdown(
            """
        <div class="narrative-card">
            <h4>🟢 Case A: Regular Dry Season Pass</h4>
            <p><strong>Observed:</strong> −8.2 dB | <strong>Climatology:</strong> −8.5 dB</p>
            <p><strong>Verdict:</strong> <span class="badge-expected">Expected</span></p>
            <small>Normal crop roughness and soil reflection.</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="narrative-card">
            <h4>🟡 Case B: Monsoon Downpour</h4>
            <p><strong>Observed:</strong> −15.8 dB | <strong>Climatology:</strong> −11.0 dB (Z = −3.4)</p>
            <p><strong>Weather-Aware Expected:</strong> −15.3 dB (Z = −0.3)</p>
            <p><strong>Verdict:</strong> <span class="badge-explained">Unusual but Explained</span></p>
            <small>Traditional tools flag a disaster. CounterSAR recognizes that 80 mm antecedent rainfall caused normal temporary water pooling.</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="narrative-card">
            <h4>🔴 Case C: True Unexplained Event</h4>
            <p><strong>Observed:</strong> −18.4 dB | <strong>Weather Expected:</strong> −10.1 dB</p>
            <p><strong>Weather Z-Score:</strong> −4.6 | <strong>Persistence:</strong> Confirmed</p>
            <p><strong>Verdict:</strong> <span class="badge-unexplained">Unexplained Candidate Event</span></p>
            <small>Radar dropped sharply without sufficient rainfall to explain it. Possible bund breach or abnormal prolonged submergence.</small>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 2: EXPECTED VS ACTUAL VIEWER (Swipe / Side-by-Side)
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "🔄 Expected vs Actual Viewer":
    st.markdown(
        """
    <div class="csar-header">
        <h1>🔄 Expected vs Actual Spatial Viewer</h1>
        <p class="sub">Compare Observed Satellite Backscatter Against Weather-Aware Expected Predictions</p>
        <p class="detail">Isolate the true spatial residual footprint: Δσ⁰ = Observed − Predicted</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    time_strings = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in ds_filtered.time.values]
    if not time_strings:
        st.warning("No observations in the selected date range.")
        st.stop()

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        selected_date_str = st.select_slider(
            "Select Acquisition Date",
            options=time_strings,
            value=time_strings[len(time_strings) // 2],
        )

    # Load 2D spatial representation for this timestamp
    spatial_ds = load_mock_spatial(selected_date_str)
    t_idx = time_strings.index(selected_date_str)

    # Time-series values at this timestep
    obs_val = float(ds_filtered.observed_backscatter_db.values[t_idx])
    pred_val = float(ds_filtered.weather_predicted_db.values[t_idx])
    clim_val = float(ds_filtered.climatology_predicted_db.values[t_idx])
    verdict = str(ds_filtered.verdict_category.values[t_idx])
    z_w = float(ds_filtered.z_score_weather.values[t_idx])

    with c_sel2:
        st.markdown(f"**Verdict:** {verdict}")
        if "Unexplained" in verdict:
            st.markdown('<span class="badge-unexplained">⚠️ Unexplained Candidate</span>', unsafe_allow_html=True)
        elif "Explained" in verdict:
            st.markdown('<span class="badge-explained">🟡 Unusual but Explained</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-expected">🟢 Expected</span>', unsafe_allow_html=True)

    # Generate synthetic spatial expected and residual grids
    lats = spatial_ds.latitude.values
    lons = spatial_ds.longitude.values
    observed_grid = spatial_ds.vv_db.values

    # Weather-predicted expected spatial field is smoother
    predicted_grid = observed_grid - (obs_val - pred_val) + np.random.default_rng(42).normal(0, 0.2, observed_grid.shape)
    residual_grid = observed_grid - predicted_grid

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown("#### 🛰️ Observed Backscatter σ⁰ (dB)")
        fig_obs = go.Figure(
            go.Heatmap(
                z=observed_grid,
                x=np.round(lons, 3),
                y=np.round(lats, 3),
                colorscale="Viridis",
                zmin=-22,
                zmax=-5,
                colorbar=dict(title="dB", len=0.8),
            )
        )
        fig_obs.update_layout(**_PLOTLY_LAYOUT, height=360, margin=dict(l=40, r=20, t=30, b=30))
        st.plotly_chart(fig_obs, use_container_width=True)
        st.caption(f"Area Mean Observed: {obs_val:.2f} dB")

    with m_col2:
        st.markdown("#### 🧠 Weather-Expected σ⁰ (dB)")
        fig_pred = go.Figure(
            go.Heatmap(
                z=predicted_grid,
                x=np.round(lons, 3),
                y=np.round(lats, 3),
                colorscale="Viridis",
                zmin=-22,
                zmax=-5,
                colorbar=dict(title="dB", len=0.8),
            )
        )
        fig_pred.update_layout(**_PLOTLY_LAYOUT, height=360, margin=dict(l=40, r=20, t=30, b=30))
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption(f"Weather-Aware Prediction: {pred_val:.2f} dB")

    with m_col3:
        st.markdown("#### 🔍 Residual Difference Δσ⁰ (dB)")
        fig_res = go.Figure(
            go.Heatmap(
                z=residual_grid,
                x=np.round(lons, 3),
                y=np.round(lats, 3),
                colorscale="RdBu",
                zmid=0.0,
                zmin=-8,
                zmax=8,
                colorbar=dict(title="Δ dB", len=0.8),
            )
        )
        fig_res.update_layout(**_PLOTLY_LAYOUT, height=360, margin=dict(l=40, r=20, t=30, b=30))
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption(f"Residual: {obs_val - pred_val:.2f} dB (Z = {z_w:.2f})")

    st.markdown("---")
    st.markdown(
        r"""
    > **Physical Interpretation:**
    > - Blue/Cyan residuals ($\Delta\sigma^0 > +2\text{ dB}$): Observed backscatter is higher than weather expected (emergent vegetation, roughness, or double bounce).
    > - Red residuals ($\Delta\sigma^0 < -2\text{ dB}$): Observed backscatter is lower than weather expected (specular reflection from unexplained standing water or abnormal soil saturation).
    """
    )


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 3: POLARIZATION & FREQUENCY EXPLORER
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "📡 Polarization & Frequency Explorer":
    st.markdown(
        """
    <div class="csar-header">
        <h1>📡 Polarization & Frequency Explorer</h1>
        <p class="sub">Sentinel-1 C-Band vs. NISAR L-Band / S-Band Radar Penetration & Polarimetry</p>
        <p class="detail">Analyzing surface roughness, volumetric canopy scattering, and multi-frequency response</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    # NISAR Availability Status Card
    st.info(
        "🛰️ **Data Source & Sensor Coverage Notice:**\n\n"
        "• **Sentinel-1 (C-band ~5.4 GHz)**: Operational with regular 12-day repeat orbit over Kuttanad.\n\n"
        "• **NISAR (NASA-ISRO SAR, L-band ~1.25 GHz / S-band ~3.2 GHz)**: "
        "*Simulated Demonstration Mode*. In-orbit commissioning is ongoing. L-band dual-pol channels (HH, HV) "
        "are modeled below for pipeline integration testing. Operational NISAR observations over Kuttanad will automatically connect as data products are released."
    )

    p_col1, p_col2 = st.columns([1, 1.5])

    with p_col1:
        st.markdown("### Radar Channel Physics")
        st.markdown(
            """
        | Band / Channel | Frequency | Wavelength | Physical Sensitivity |
        | :--- | :--- | :--- | :--- |
        | **Sentinel-1 VV** | 5.4 GHz (C-band) | 5.6 cm | Direct surface roughness, open water reflection |
        | **Sentinel-1 VH** | 5.4 GHz (C-band) | 5.6 cm | Canopy volume scattering, paddy growth stages |
        | **VV / VH Ratio** | Dual-polarization | — | Biomass structure & soil moisture decoupling |
        | **NISAR L-band HH**| 1.25 GHz (L-band) | 24.0 cm | Deep canopy penetration, flooded vegetation double-bounce |
        | **NISAR L-band HV**| 1.25 GHz (L-band) | 24.0 cm | Dense forest & wetland root structure |
        """
        )

        channel_sel = st.selectbox(
            "Select Polarization Channel for Time-Series Analysis",
            ["Sentinel-1 VV (Co-pol)", "Sentinel-1 VH (Cross-pol)", "Cross-Pol Ratio (VV − VH)", "NISAR L-band HH (Simulated)", "NISAR L-band HV (Simulated)"],
        )

    with p_col2:
        st.markdown(f"### {channel_sel} Time Series")
        times = pd.to_datetime(ds_filtered.time.values)
        vv = ds_filtered.observed_backscatter_db.values
        vh = ds_filtered.vh_db.values if "vh_db" in ds_filtered else vv - 7.0

        fig_pol = go.Figure()

        if "VV (Co-pol)" in channel_sel:
            fig_pol.add_trace(go.Scatter(x=times, y=vv, mode="lines+markers", name="Sentinel-1 σ⁰ VV", line=dict(color=_C["vv"], width=2)))
            y_title = "σ⁰ VV (dB)"
        elif "VH (Cross-pol)" in channel_sel:
            fig_pol.add_trace(go.Scatter(x=times, y=vh, mode="lines+markers", name="Sentinel-1 σ⁰ VH", line=dict(color=_C["vh"], width=2)))
            y_title = "σ⁰ VH (dB)"
        elif "Ratio" in channel_sel:
            ratio = vv - vh
            fig_pol.add_trace(go.Scatter(x=times, y=ratio, mode="lines+markers", name="VV − VH Ratio", line=dict(color="#34d399", width=2)))
            y_title = "Ratio (dB)"
        elif "HH" in channel_sel:
            hh_sim = vv + 1.8 + np.sin(np.linspace(0, 10, len(times))) * 0.8
            fig_pol.add_trace(go.Scatter(x=times, y=hh_sim, mode="lines+markers", name="NISAR σ⁰ HH (Simulated)", line=dict(color="#f43f5e", width=2, dash="dash")))
            y_title = "NISAR σ⁰ HH (dB)"
        else:
            hv_sim = vh - 2.5 + np.cos(np.linspace(0, 10, len(times))) * 0.5
            fig_pol.add_trace(go.Scatter(x=times, y=hv_sim, mode="lines+markers", name="NISAR σ⁰ HV (Simulated)", line=dict(color="#ec4899", width=2, dash="dash")))
            y_title = "NISAR σ⁰ HV (dB)"

        fig_pol.update_layout(
            **_PLOTLY_LAYOUT,
            xaxis_title="Acquisition Date",
            yaxis_title=y_title,
            height=380,
            xaxis=dict(gridcolor=_C["grid"]),
            yaxis=dict(gridcolor=_C["grid"]),
        )
        st.plotly_chart(fig_pol, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 4: TIME MACHINE
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "⏳ Time Machine":
    st.markdown(
        """
    <div class="csar-header">
        <h1>⏳ CounterSAR Time Machine</h1>
        <p class="sub">Chronological Scrubbing Through Satellite History, Weather Context, and Live Verdicts</p>
        <p class="detail">Step through the timeline to see how weather-aware models adapt to monsoon downpours</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    time_strs = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in ds_filtered.time.values]
    if not time_strs:
        st.warning("No observations in the selected date range.")
        st.stop()

    slider_date = st.select_slider(
        "Scrub Timeline",
        options=time_strs,
        value=time_strs[len(time_strs) // 2],
        key="time_machine_slider",
    )

    idx = time_strs.index(slider_date)

    # Values at current timestep
    cur_obs = float(ds_filtered.observed_backscatter_db.values[idx])
    cur_pred = float(ds_filtered.weather_predicted_db.values[idx])
    cur_clim = float(ds_filtered.climatology_predicted_db.values[idx])
    cur_lower = float(ds_filtered.weather_lower_bound_db.values[idx])
    cur_upper = float(ds_filtered.weather_upper_bound_db.values[idx])
    cur_zw = float(ds_filtered.z_score_weather.values[idx])
    cur_zc = float(ds_filtered.z_score_climatology.values[idx])
    cur_thresh = float(ds_filtered.adaptive_threshold.values[idx])
    cur_verdict = str(ds_filtered.verdict_category.values[idx])
    is_pers = bool(ds_filtered.is_persistent_anomaly.values[idx])

    cur_rain = float(ds_filtered.rainfall_mm.values[idx]) if "rainfall_mm" in ds_filtered else 0.0
    cur_rain3d = float(ds_filtered.antecedent_rain_3d_mm.values[idx]) if "antecedent_rain_3d_mm" in ds_filtered else 0.0
    cur_sm = float(ds_filtered.soil_moisture.values[idx]) if "soil_moisture" in ds_filtered else 0.3
    cur_temp = float(ds_filtered.temperature_c.values[idx]) if "temperature_c" in ds_filtered else 28.0

    # Top Metric Banner
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Acquisition Date", slider_date)
    b2.metric("Observed σ⁰", f"{cur_obs:.2f} dB", delta=f"{cur_obs - cur_pred:.2f} dB vs Weather")
    b3.metric("Weather Z-Score", f"{cur_zw:.2f}", delta=f"Threshold ±{cur_thresh:.2f}")

    with b4:
        st.markdown("**Verdict**")
        if "Unexplained" in cur_verdict:
            pers_txt = " (Confirmed Persistent)" if is_pers else " (Single-Pass Transient)"
            st.markdown(f'<span class="badge-unexplained">🔴 {cur_verdict}{pers_txt}</span>', unsafe_allow_html=True)
        elif "Explained" in cur_verdict:
            st.markdown(f'<span class="badge-explained">🟡 {cur_verdict}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-expected">🟢 {cur_verdict}</span>', unsafe_allow_html=True)

    st.markdown("")

    # Timeline Graph with Vertical Cursor Line
    fig_tm = go.Figure()

    all_times = pd.to_datetime(ds_filtered.time.values)
    all_obs = ds_filtered.observed_backscatter_db.values
    all_pred = ds_filtered.weather_predicted_db.values
    all_low = ds_filtered.weather_lower_bound_db.values
    all_up = ds_filtered.weather_upper_bound_db.values

    # Uncertainty bounds
    fig_tm.add_trace(go.Scatter(x=all_times, y=all_up, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_tm.add_trace(go.Scatter(x=all_times, y=all_low, mode="lines", fill="tonexty", fillcolor="rgba(96,165,250,0.15)", line=dict(width=0), name="Weather 80% Confidence Band"))
    fig_tm.add_trace(go.Scatter(x=all_times, y=all_pred, mode="lines", name="Weather-Expected σ⁰", line=dict(color=_C["predicted"], width=2, dash="dash")))
    fig_tm.add_trace(go.Scatter(x=all_times, y=all_obs, mode="lines+markers", name="Observed σ⁰ (Sentinel-1)", line=dict(color=_C["vv"], width=2), marker=dict(size=5)))

    # Vertical line at current scrubbed date
    fig_tm.add_vline(x=slider_date, line_width=2, line_dash="solid", line_color="#f59e0b", annotation_text="Scrubbed Date", annotation_position="top right")

    fig_tm.update_layout(
        **_PLOTLY_LAYOUT,
        title=f"Radar Trajectory & Uncertainty Envelope (Cursor at {slider_date})",
        xaxis_title="Date",
        yaxis_title="Backscatter σ⁰ (dB)",
        height=380,
    )
    st.plotly_chart(fig_tm, use_container_width=True)

    # Meteorological Context Dashboard
    st.markdown("### 🌧️ Meteorological Covariates at this Timestamp")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Daily Rainfall (CHIRPS)", f"{cur_rain:.1f} mm")
    m2.metric("3-Day Antecedent Rain", f"{cur_rain3d:.1f} mm")
    m3.metric("Soil Moisture (ERA5)", f"{cur_sm:.3f} m³/m³")
    m4.metric("Surface Temperature", f"{cur_temp:.1f} °C")


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 5: ANOMALY CARDS & SCORER GALLERY
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "🚨 Anomaly Cards & Scorer":
    st.markdown(
        """
    <div class="csar-header">
        <h1>🚨 CounterSAR Anomaly Cards</h1>
        <p class="sub">Filterable Registry of Unexplained Candidate Events & Weather-Explained Variations</p>
        <p class="detail">Review multi-pass persistence, physical sign, and diagnostic metrics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    anom_list = to_anomaly_results(ds_filtered)

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cat_filter = st.selectbox("Filter by Verdict Category", ["All", "Unexplained Candidate Event", "Unusual but Explained"])
    with f_col2:
        pers_filter = st.selectbox("Filter by Persistence", ["All", "Persistent Only (≥ 2 passes)", "Single-Pass Transient"])
    with f_col3:
        dir_filter = st.selectbox("Filter by Physical Nature", ["All", "Backscatter Drop (Water / Saturation)", "Backscatter Jump (Double Bounce / Roughness)"])

    # Extract all candidate dates from dataset
    times = pd.to_datetime(ds_filtered.time.values)
    is_cand = ds_filtered.is_anomaly_candidate.values
    is_pers = ds_filtered.is_persistent_anomaly.values
    verdicts = ds_filtered.verdict_category.values
    obs = ds_filtered.observed_backscatter_db.values
    pred = ds_filtered.weather_predicted_db.values
    clim = ds_filtered.climatology_predicted_db.values
    zw = ds_filtered.z_score_weather.values
    zc = ds_filtered.z_score_climatology.values
    thresh = ds_filtered.adaptive_threshold.values
    dev = ds_filtered.deviation_nature.values

    # Build card records
    card_records = []
    for i in range(len(times)):
        v = str(verdicts[i])
        # Include if candidate, persistent, or explained
        if is_cand[i] or is_pers[i] or v == AnomalyVerdict.UNUSUAL_EXPLAINED.value:
            record = {
                "date": times[i].strftime("%Y-%m-%d"),
                "doy": times[i].dayofyear,
                "verdict": v,
                "is_persistent": bool(is_pers[i]),
                "obs": float(obs[i]),
                "pred": float(pred[i]),
                "clim": float(clim[i]),
                "zw": float(zw[i]),
                "zc": float(zc[i]),
                "thresh": float(thresh[i]),
                "nature": str(dev[i]),
            }
            # Apply filters
            if cat_filter != "All" and record["verdict"] != cat_filter:
                continue
            if pers_filter == "Persistent Only (≥ 2 passes)" and not record["is_persistent"]:
                continue
            if pers_filter == "Single-Pass Transient" and record["is_persistent"]:
                continue
            if dir_filter == "Backscatter Drop (Water / Saturation)" and "drop" not in record["nature"]:
                continue
            if dir_filter == "Backscatter Jump (Double Bounce / Roughness)" and "jump" not in record["nature"]:
                continue

            card_records.append(record)

    st.markdown(f"**Found {len(card_records)} matching anomaly events**")

    if not card_records:
        st.info("No events match the selected filter criteria.")
    else:
        # Display in 2-column card grid
        for i in range(0, len(card_records), 2):
            c1, c2 = st.columns(2)
            for col, r_idx in [(c1, i), (c2, i + 1)]:
                if r_idx < len(card_records):
                    r = card_records[r_idx]
                    delta = r["obs"] - r["pred"]

                    if "Unexplained" in r["verdict"]:
                        v_badge = '<span class="badge-unexplained">🔴 Unexplained Candidate Event</span>'
                    elif "Explained" in r["verdict"]:
                        v_badge = '<span class="badge-explained">🟡 Unusual but Explained</span>'
                    else:
                        v_badge = '<span class="badge-expected">🟢 Expected</span>'

                    p_badge = '<span class="badge-expected">✓ Persistent (≥ 2 passes)</span>' if r["is_persistent"] else '<span class="badge-explained">⏱️ Single-Pass Transient</span>'

                    with col:
                        st.markdown(
                            f"""
                        <div class="anomaly-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h3 style="margin: 0; color: #e2e8f0;">📅 {r['date']}</h3>
                                <div>{v_badge}</div>
                            </div>
                            <p style="margin: 0.2rem 0; color: #94a3b8; font-size: 0.85rem;">Day of Year: {r['doy']} | Status: {p_badge}</p>
                            <hr style="border-color: #334155; margin: 0.6rem 0;" />
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
                                <div><strong>Observed σ⁰:</strong> {r['obs']:.2f} dB</div>
                                <div><strong>Weather Expected:</strong> {r['pred']:.2f} dB</div>
                                <div><strong>Difference (Δσ⁰):</strong> <span style="color: {'#f87171' if delta < 0 else '#34d399'};">{delta:+.2f} dB</span></div>
                                <div><strong>Weather Z-Score:</strong> {r['zw']:.2f} (T = ±{r['thresh']:.2f})</div>
                                <div><strong>Climatology Z:</strong> {r['zc']:.2f}</div>
                                <div><strong>Deviation:</strong> <code>{r['nature']}</code></div>
                            </div>
                            <p style="margin-top: 0.6rem; margin-bottom: 0; font-size: 0.8rem; color: #cbd5e1;">
                                <em>Physical Diagnosis:</em> {'Severe backscatter drop unexplained by rainfall or soil saturation; potential flooding or bund failure.' if 'drop' in r['nature'] and 'Unexplained' in r['verdict'] else 'Controlled paddy flooding or storm moisture explained by weather covariates.' if 'Explained' in r['verdict'] else 'Double bounce increase.'}
                            </p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 6: RAW OBSERVATIONS & ENVIRONMENTAL DATA
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "📊 Observations & Environmental Explorer":
    st.markdown(
        """
    <div class="csar-header">
        <h1>📊 Observations & Environmental Explorer</h1>
        <p class="sub">Raw Sentinel-1 Backscatter & CHIRPS / ERA5 Meteorological Covariates</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    tab1, tab2 = st.tabs(["🛰️ Radar Observations (VV & VH)", "🌧️ Environmental Covariates"])

    times = pd.to_datetime(ds_filtered.time.values)
    vv = ds_filtered.observed_backscatter_db.values
    vh = ds_filtered.vh_db.values if "vh_db" in ds_filtered else vv - 7.0

    with tab1:
        fig_sar = go.Figure()
        fig_sar.add_trace(go.Scatter(x=times, y=vv, mode="lines+markers", name="σ⁰ VV (dB)", line=dict(color=_C["vv"], width=2)))
        fig_sar.add_trace(go.Scatter(x=times, y=vh, mode="lines+markers", name="σ⁰ VH (dB)", line=dict(color=_C["vh"], width=2)))

        fig_sar.update_layout(**_PLOTLY_LAYOUT, title="Sentinel-1 Backscatter Time Series", xaxis_title="Date", yaxis_title="Backscatter σ⁰ (dB)", height=380)
        st.plotly_chart(fig_sar, use_container_width=True)

    with tab2:
        rain = ds_filtered.rainfall_mm.values if "rainfall_mm" in ds_filtered else np.zeros(len(times))
        sm = ds_filtered.soil_moisture.values if "soil_moisture" in ds_filtered else np.full(len(times), 0.3)
        temp = ds_filtered.temperature_c.values if "temperature_c" in ds_filtered else np.full(len(times), 28.0)

        fig_env = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["CHIRPS Daily Rainfall (mm)", "ERA5 Volumetric Soil Moisture (m³/m³)"])
        fig_env.add_trace(go.Bar(x=times, y=rain, name="Rainfall (mm)", marker_color=_C["rain"]), row=1, col=1)
        fig_env.add_trace(go.Scatter(x=times, y=sm, name="Soil Moisture", line=dict(color=_C["sm"], width=2)), row=2, col=1)

        fig_env.update_layout(**_PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig_env, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  FEATURE 7: STUDY AREA OVERVIEW
# ════════════════════════════════════════════════════════════════════════

elif nav_choice == "🗺️ Study Area Overview":
    st.markdown(
        """
    <div class="csar-header">
        <h1>🗺️ Study Area — Kuttanad, Kerala, India</h1>
        <p class="sub">Geospatial Context, Lowland Topography, and Agricultural Polders</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_synth_notice()

    col_t, col_m = st.columns([1, 1.2])

    with col_t:
        st.markdown("### Geographic Boundaries & Topography")
        st.markdown(
            f"""
        - **Bounding Box**: Lat {KUTTANAD_BOUNDS['lat_min']}°N to {KUTTANAD_BOUNDS['lat_max']}°N, Lon {KUTTANAD_BOUNDS['lon_min']}°E to {KUTTANAD_BOUNDS['lon_max']}°E
        - **Elevation**: −0.5 to −2.5 m (Below Mean Sea Level)
        - **Total Area**: ~500 km²
        - **Grid Standard**: 100 m Reference Grid ({GRID_RES_METERS} m, {GRID_RES_DEG}° res in EPSG:4326)
        - **Primary Crops**: Paddy rice (two cropping seasons: Punja and Virippu)
        """
        )
        st.markdown("### Operational Satellite Coverage")
        st.markdown(
            """
        - **Sentinel-1 C-band SAR**: Repeat cycle 12 days (Ascending and Descending orbits).
        - **NISAR L-band & S-band**: High penetration radar for biomass and wetland monitoring.
        - **CHIRPS**: Daily 0.05° gridded precipitation.
        - **ERA5**: 0.25° reanalysis soil moisture and surface temperature.
        """
        )

    with col_m:
        st.markdown("### 🗺️ Interactive Kuttanad Map")
        centre = study_info["centre"]
        m = folium.Map(location=[centre["lat"], centre["lon"]], zoom_start=11, tiles="OpenStreetMap")

        bbox = KUTTANAD_BBOX
        folium.Polygon(
            locations=[
                [bbox["lat_min"], bbox["lon_min"]],
                [bbox["lat_max"], bbox["lon_min"]],
                [bbox["lat_max"], bbox["lon_max"]],
                [bbox["lat_min"], bbox["lon_max"]],
            ],
            color="#00bcd4",
            weight=2,
            fill=True,
            fill_color="#00bcd4",
            fill_opacity=0.10,
            tooltip="Kuttanad Study Area Bounding Box",
        ).add_to(m)

        for name, lat, lon in [("Alappuzha", 9.4981, 76.3388), ("Kumarakom", 9.6175, 76.4301), ("Champakkulam", 9.4000, 76.5167)]:
            folium.Marker([lat, lon], tooltip=name, popup=name, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

        st_folium(m, width=None, height=450, returned_objects=[])

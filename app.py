import streamlit as st
import plotly.express as px
import joblib
import os
from src.ingestion import fetch_telemetry_logic
from src.processing import master_diagnostic
from src.maintenance import predict_thermal_failure
from src.s3_manager import S3Manager

# 1. PAGE SETUP (Must be the first Streamlit command)
st.set_page_config(page_title="Apex Guardian 2026", layout="wide")


# --- CACHE LAYER 1: S3 & MODEL (Global Resources) ---
@st.cache_resource
def initialize_mlops_system():
    """
    Connects to AWS S3 and ensures the latest 'Champion' model is available locally.
    This runs once per app restart.
    """
    s3 = S3Manager()
    # Pull the latest brain from the cloud feature store
    s3.download_production_model()
    return s3


# --- CACHE LAYER 2: TELEMETRY & INFERENCE (Data Storage) ---
@st.cache_data(ttl=3600)
def get_processed_data(year=2026):
    """
    Handles the full data lifecycle: Ingestion -> Processing -> AI Inference.
    Cached for 1 hour to prevent API rate-limiting.
    """
    # 1. Pull raw telemetry (Logic version avoids Streamlit warnings)
    raw_data, event_name = fetch_telemetry_logic(year)

    # 2. Run Feature Engineering (Rulebook)
    processed_data = master_diagnostic(raw_data)

    # 3. Run AI Anomaly Detection (Inference)
    final_data = predict_thermal_failure(processed_data)

    return final_data, event_name


# --- 2. DATA ORCHESTRATION ---
with st.spinner("☁️ Syncing with AWS S3 & Running AI Diagnostics..."):
    # Initialize the cloud connection
    s3_client = initialize_mlops_system()

    # Load the high-speed cached data
    final_data, event_name = get_processed_data(year=2026)

# Inject the SMART ML Classifications into the "Diagnostic" column
# This ensures your 3D plot and Map show the ML labels correctly
anomaly_mask = final_data["Maintenance_Alert"] == True
final_data.loc[anomaly_mask, "Diagnostic"] = final_data.loc[
    anomaly_mask, "ML_Diagnosis"
]

# --- 3. UI PRESENTATION ---
st.title("🏎️ 2026 F1 Apex Guardian")
st.markdown("Live Grid-Wide Energy & Active Aero Monitor")

# --- THE "HOW TO READ THIS" SECTION FOR RECRUITERS ---
with st.expander("📖 HOW TO READ THIS DASHBOARD (Click to expand)"):
    st.markdown(
        """
    **The 2026 Problem:** F1 cars now rely on a 50/50 split between gas and electric power. 
    This AI tool detects when cars are losing speed on straights and diagnoses *why*:
    * 🔴 **Super Clipping:** The car is intentionally harvesting battery power (Strategy).
    * 🔵 **Z-Mode Transition:** The car's wings are closing to prepare for a corner (Aerodynamics).
    * 📉 **Driver Lift & Coast:** The AI caught the driver lifting off the throttle early to save fuel.
    * 🔋 **Severe High-Speed Derating:** The ML model detected a critical failure in electrical deployment at 300+ km/h.
    """
    )

# Status Badge
st.info(
    f"📊 **Active Data Source:** 2026 {event_name} | **Status:** Live Automated Cloud Pipeline"
)

# --- 4. SIDEBAR & MLOPS BADGE ---
st.sidebar.header("🔧 Telemetry Controls")

all_drivers = final_data["Driver"].unique().tolist()
selected_drivers = st.sidebar.multiselect(
    "Select Drivers to Compare:",
    options=all_drivers,
    default=all_drivers[:3],  # Default to top 3 for speed
)

if not selected_drivers:
    st.warning("Please select at least one driver from the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("☁️ MLOps Architecture")
st.sidebar.success(
    "**Feature Store:** AWS S3\n\n**Registry:** DagsHub / MLflow\n\n**CI/CD:** GitHub Actions\n\n**Model:** Isolation Forest"
)

# --- 5. AI INSIGHTS ENGINE ---
st.markdown("---")
st.subheader("🧠 AI Diagnostic Insights")

filtered_data = final_data[final_data["Driver"].isin(selected_drivers)]
anomaly_frames = len(filtered_data[filtered_data["Maintenance_Alert"] == True])

# Anomaly Logic
if anomaly_frames > 10:
    st.error(
        f"**Critical ML Alert:** {anomaly_frames} frames of abnormal power delivery at high speed (>280 km/h)."
    )
elif anomaly_frames > 0:
    st.warning(
        f"**Minor ML Alert:** {anomaly_frames} high-speed anomaly frames detected."
    )
else:
    st.success(
        "**Health Check:** AI confirms optimal high-speed power delivery. No derating detected."
    )

# --- 6. METRICS & VISUALS ---
col1, col2, col3 = st.columns(3)
top_speed_row = filtered_data.loc[filtered_data["Speed"].idxmax()]
col1.metric(
    "Max Speed",
    f"{int(top_speed_row['Speed'])} km/h",
    f"Driver: {top_speed_row['Driver']}",
)
col2.metric(
    "Harvesting Events",
    len(filtered_data[filtered_data["Diagnostic"] == "SUPER CLIPPING"]),
)
col3.metric("ML Anomalies", anomaly_frames)

st.subheader("📊 Speed vs. Distance Trace")
fig_speed = px.line(
    filtered_data, x="Distance", y="Speed", color="Driver", template="plotly_dark"
)
st.plotly_chart(fig_speed, use_container_width=True)

st.subheader("🗺️ Spatial Diagnostic Map")
master_color_map = {
    "Optimal": "#2c3e50",
    "SUPER CLIPPING": "#e74c3c",
    "Z-MODE TRANSITION": "#3498db",
    "📉 Driver Lift & Coast": "#00ffcc",
    "🔋 Severe High-Speed Derating": "#ff00ff",
    "⚠️ Unknown Power Loss": "#ffaa00",
}

fig_map = px.scatter(
    filtered_data,
    x="X",
    y="Y",
    color="Diagnostic",
    hover_data=["Driver", "Speed", "Throttle"],
    color_discrete_map=master_color_map,
)
fig_map.update_traces(marker=dict(size=4, opacity=0.8))
st.plotly_chart(fig_map, use_container_width=True)

# 7. THE 3D FEATURE SPACE
st.subheader("🧊 AI Decision Space (Under the Hood)")
if "Acceleration" in filtered_data.columns and "RPM" in filtered_data.columns:
    # Subsample for 3D performance if data is massive
    plot_data = filtered_data.iloc[::2] if len(filtered_data) > 15000 else filtered_data
    fig_3d = px.scatter_3d(
        plot_data,
        x="Speed",
        y="RPM",
        z="Acceleration",
        color="Diagnostic",
        color_discrete_map=master_color_map,
        opacity=0.7,
    )
    fig_3d.update_traces(marker=dict(size=2))
    st.plotly_chart(fig_3d, use_container_width=True)

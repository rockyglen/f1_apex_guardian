import streamlit as st
import plotly.express as px
from src.ingestion import get_all_drivers_telemetry
from src.processing import master_diagnostic
from src.maintenance import predict_thermal_failure

# 1. PAGE SETUP
st.set_page_config(page_title="Apex Guardian 2026", layout="wide")
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

# 2. DATA INGESTION & INFERENCE (CLOUD SYNCED)
with st.spinner("☁️ Connecting to AWS S3 Feature Store & Running Cloud Inference..."):
    # Pull Data dynamically
    raw_data, event_name = get_all_drivers_telemetry(year=2026)

    # Run the Rulebook (Heuristics)
    processed_data = master_diagnostic(raw_data)

    # Run the AI Brain (Inference pulls from S3 automatically)
    final_data = predict_thermal_failure(processed_data)

# Inject the SMART ML Classifications into the "Diagnostic" column
anomaly_mask = final_data["Maintenance_Alert"] == True
final_data.loc[anomaly_mask, "Diagnostic"] = final_data.loc[
    anomaly_mask, "ML_Diagnosis"
]

# Prove the pipeline is live with ONE dynamic info box
st.info(
    f"📊 **Active Data Source:** 2026 {event_name} | **Status:** Live Automated Cloud Pipeline"
)

# 3. THE SIDEBAR SELECTOR & MLOPS BADGE
st.sidebar.header("🔧 Telemetry Controls")

all_drivers = final_data["Driver"].unique().tolist()
selected_drivers = st.sidebar.multiselect(
    "Select Drivers to Compare:", options=all_drivers, default=all_drivers
)

if not selected_drivers:
    st.warning("Please select at least one driver from the sidebar.")
    st.stop()

# --- THE MLOPS FLEX FOR RECRUITERS ---
st.sidebar.markdown("---")
st.sidebar.subheader("☁️ MLOps Architecture")
st.sidebar.success(
    "**Feature Store:** AWS S3\n\n**Registry:** DagsHub / MLflow\n\n**CI/CD:** GitHub Actions\n\n**Model:** Isolation Forest"
)
st.sidebar.markdown("---")

filtered_data = final_data[final_data["Driver"].isin(selected_drivers)]

# --- THE AI INSIGHTS ENGINE ---
st.markdown("---")
st.subheader("🧠 AI Diagnostic Insights")

# Strategy Insight
most_clipping_driver = filtered_data[filtered_data["Diagnostic"] == "SUPER CLIPPING"][
    "Driver"
].mode()

if not most_clipping_driver.empty:
    worst_clipper = most_clipping_driver[0]
    st.info(
        f"**Strategy Insight:** Among selected drivers, **{worst_clipper}** is spending the most time in 'Super Clipping' mode (Battery Harvesting)."
    )

# Anomaly Insight
anomaly_frames = len(filtered_data[filtered_data["Maintenance_Alert"] == True])

if anomaly_frames > 5:
    st.error(
        f"**Critical ML Alert:** The Anomaly Detector flagged {anomaly_frames} frames of abnormal power delivery at high speed (>280 km/h). See the map below."
    )
elif anomaly_frames > 0:
    st.warning(
        f"**Minor ML Alert:** {anomaly_frames} high-speed anomaly frames detected. Likely statistical noise or track bumps."
    )
else:
    st.success(
        "**Health Check:** AI Anomaly Detector confirms optimal high-speed power delivery. No derating detected."
    )

st.markdown("---")

# 4. TOP LEVEL METRICS
col1, col2, col3 = st.columns(3)
top_speed_row = filtered_data.loc[filtered_data["Speed"].idxmax()]
col1.metric(
    "Max Speed (Selected)",
    f"{int(top_speed_row['Speed'])} km/h",
    f"Driver: {top_speed_row['Driver']}",
)
total_clipping = len(filtered_data[filtered_data["Diagnostic"] == "SUPER CLIPPING"])
col2.metric("Harvesting Events (Selected)", total_clipping)
col3.metric(
    "ML Anomalies Detected",
    anomaly_frames,
)

# 5. INSIGHT 1: THE TELEMETRY TRACE
st.subheader("📊 Speed vs. Distance Trace")
st.markdown(
    "*Use this graph to physically see the 'Speed Sag' where the red line drops before the braking zone.*"
)
fig_speed = px.line(filtered_data, x="Distance", y="Speed", color="Driver")
st.plotly_chart(fig_speed, width="stretch")

# 6. INSIGHT 2: THE TRACK MAP
st.subheader("🗺️ Spatial Diagnostic Map")
st.markdown(
    "*Use this map to physically see WHERE the anomalies are happening on the track. Look for the purple markers at the ends of the longest straights.*"
)

# CENTRALIZED COLOR MAP FOR CONSISTENCY
master_color_map = {
    "Optimal": "#2c3e50",
    "SUPER CLIPPING": "#e74c3c",
    "Z-MODE TRANSITION": "#3498db",
    "WARNING: UNKNOWN POWER LOSS": "#f1c40f",
    "📉 Driver Lift & Coast": "#00ffcc",  # Teal
    "🔋 Severe High-Speed Derating": "#ff00ff",  # Neon Purple
    "⚠️ Unknown Power Loss": "#ffaa00",  # Orange
}

fig_map = px.scatter(
    filtered_data,
    x="X",
    y="Y",
    color="Diagnostic",
    hover_data=["Driver", "Speed", "Throttle", "Acceleration"],
    color_discrete_map=master_color_map,
)
fig_map.update_traces(marker=dict(size=4, opacity=0.8))
fig_map.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600,
)
st.plotly_chart(fig_map, width="stretch")

# 7. INSIGHT 3: THE ML FEATURE SPACE (3D)
st.markdown("---")
st.subheader("🧊 AI Decision Space (Under the Hood)")
st.markdown(
    "*How the AI thinks: This 3D plot visualizes the physical relationships between Speed, RPM, and Acceleration. Notice how the purple ML anomalies isolate themselves from the 'normal' clusters of engine behavior.*"
)

if "Acceleration" in filtered_data.columns and "RPM" in filtered_data.columns:
    plot_data = filtered_data.iloc[::2] if len(filtered_data) > 10000 else filtered_data

    fig_3d = px.scatter_3d(
        plot_data,
        x="Speed",
        y="RPM",
        z="Acceleration",
        color="Diagnostic",
        hover_data=["Driver", "Throttle"],
        color_discrete_map=master_color_map,  # THE QA FIX
        opacity=0.7,
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(
        height=700,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="Speed (km/h)",
            yaxis_title="Engine RPM",
            zaxis_title="Acceleration (m/s²)",
        ),
    )
    st.plotly_chart(fig_3d, width="stretch")
else:
    st.warning(
        "⚠️ Feature Engineering metrics (RPM/Acceleration) not found. Run processing pipeline first."
    )

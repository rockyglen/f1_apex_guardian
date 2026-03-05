import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import mlflow
import mlflow.sklearn
import os
import dagshub
from dotenv import load_dotenv

# Direct imports from current directory
from ingestion import get_all_drivers_telemetry
from processing import master_diagnostic
from drift_detector import DriftDetector
from s3_manager import S3Manager
from challenger import evaluate_challenger

# Load environment variables for AWS and DagsHub
load_dotenv()

repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")
token = os.getenv("DAGSHUB_TOKEN")

if token and repo_owner and repo_name:
    # Set the remote tracking URI directly
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    # Manually inject the credentials into the environment for the MLflow client
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    print(f"📡 Remote MLflow Tracking initialized: {repo_name}")
else:
    print("⚠️ Local MLflow Tracking (Credentials missing)")


def train_anomaly_detector(df, event_name):
    """
    Trains the Isolation Forest and returns the model.
    Note: mlflow.start_run is now handled in the main block.
    """
    print(f"🏋️‍♂️ Starting Pipeline for: {event_name}")

    # 1. Domain Scoping (Focused on high-speed straights)
    speed_limit = 280
    throttle_limit = 90
    high_speed_data = df[
        (df["Throttle"] >= throttle_limit) & (df["Speed"] >= speed_limit)
    ].copy()

    features = ["Speed", "Throttle", "RPM", "Acceleration"]
    training_data = high_speed_data[features].dropna()

    n_est = 100
    contam = 0.01

    # Log Parameters to the active run
    mlflow.log_param("event_name", event_name)
    mlflow.log_param("speed_threshold", speed_limit)
    mlflow.log_param("n_estimators", n_est)

    # Train Model
    model = IsolationForest(n_estimators=n_est, contamination=contam, random_state=42)
    print(f"🧠 Model is learning from {len(training_data)} telemetry points...")
    model.fit(training_data)

    # Evaluation Metrics
    test_df = training_data.copy()
    test_df["Anomaly"] = model.predict(test_df[features])
    anomalies = test_df[test_df["Anomaly"] == -1]

    metrics = {
        "total_training_rows": len(test_df),
        "anomalies_found": len(anomalies),
        "avg_speed_anomaly": (anomalies["Speed"].mean() if not anomalies.empty else 0),
        "avg_accel_anomaly": (
            anomalies["Acceleration"].mean() if not anomalies.empty else 0
        ),
    }
    mlflow.log_metrics(metrics)

    # 2. THE CHALLENGER PHASE
    is_promotable = evaluate_challenger(metrics)

    if is_promotable:
        mlflow.set_tag("model_status", "production")

        # Local save
        os.makedirs("models", exist_ok=True)
        local_model_path = "models/thermal_detector.pkl"
        joblib.dump(model, local_model_path)

        # S3 Production Upload
        s3.upload_model(local_model_path)
        print("🚀 Champion model uploaded to S3 Production folder.")

        # Log to DagsHub Artifacts
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        print("✅ Model promoted to Production.")
    else:
        mlflow.set_tag("model_status", "rejected")
        print("❌ Model failed Challenger check. Keeping existing Champion.")

    return model


if __name__ == "__main__":
    s3 = S3Manager()

    # 1. INGESTION
    raw_data, event_name = get_all_drivers_telemetry(year=2026)
    print("⚙️ Running Feature Engineering...")
    processed_data = master_diagnostic(raw_data)

    # 2. MASTER MLFLOW RUN (One run to rule them all)
    with mlflow.start_run(run_name=f"Pipeline_{event_name.replace(' ', '_')}"):

        # FEATURE STORE SYNC
        current_feature_file = f"features_{event_name.replace(' ', '_')}.parquet"
        os.makedirs("data", exist_ok=True)
        processed_data.to_parquet(f"data/{current_feature_file}")
        s3.upload_features(processed_data, current_feature_file)

        # DRIFT DETECTION
        golden_path = "data/golden_reference.parquet"
        if os.path.exists(golden_path):
            reference_data = pd.read_parquet(golden_path)
            detector = DriftDetector(reference_data)
            is_drifting, _ = detector.check_drift(processed_data)

            if is_drifting:
                print(
                    "🚨 DATA DRIFT DETECTED: Physics profile has shifted significantly."
                )
                mlflow.log_param("drift_alert", True)
            else:
                mlflow.log_param("drift_alert", False)
        else:
            # Create baseline if it doesn't exist
            processed_data.to_parquet(golden_path)
            s3.upload_features(processed_data, "golden_reference.parquet")
            mlflow.set_tag("data_status", "new_baseline_created")

        # 3. TRAINING & PROMOTION
        train_anomaly_detector(processed_data, event_name)

    print(f"🏁 Pipeline complete. View results on DagsHub.")

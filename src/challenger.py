import joblib
import mlflow
from s3_manager import S3Manager


def evaluate_challenger(new_model_metrics):
    s3 = S3Manager()

    # 1. Pull the current "Champion" from S3
    has_champion = s3.download_latest_model()

    if not has_champion:
        print("🥇 No champion found. New model is now the Champion.")
        return True

    # 2. Simple logic: If the new model found NO anomalies, it's likely a failure
    if new_model_metrics["anomalies_found"] == 0:
        print("❌ Challenger rejected: Found 0 anomalies. Keeping current model.")
        return False

    # 3. Add more complex drift/metric checks here
    print("✅ Challenger passed! Promoting to Production.")
    return True

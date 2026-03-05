import boto3
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


class S3Manager:
    def __init__(self):
        # AWS Credentials pulled from .env or GitHub Secrets
        self.s3 = boto3.client("s3")
        self.bucket = os.getenv("S3_BUCKET_NAME")

    def upload_features(self, df, filename):
        """Uploads processed telemetry to the S3 Feature Store with MLflow metadata."""
        local_path = f"data/{filename}"
        os.makedirs("data", exist_ok=True)
        df.to_parquet(local_path)

        # Get the current MLflow run ID for version tracking/lineage
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "no_run"

        # Upload with Metadata tags so DagsHub/S3 can track the lineage
        self.s3.upload_file(
            local_path,
            self.bucket,
            f"feature_store/{filename}",
            ExtraArgs={"Metadata": {"mlflow_run_id": run_id}},
        )
        print(f"🚀 Features versioned & synced to S3: feature_store/{filename}")

    def download_latest_model(self):
        """
        Pulls the current 'Champion' model for the Challenger check.
        Named specifically to match challenger.py requirements.
        """
        try:
            os.makedirs("models", exist_ok=True)
            self.s3.download_file(
                self.bucket,
                "production/thermal_detector.pkl",
                "models/production_model.pkl",
            )
            return True
        except Exception as e:
            # We don't print an error if it's just missing; that means it's the first run
            return False

    def download_production_model(self):
        """Pulls the current 'Champion' model specifically for the Dashboard/Inference."""
        try:
            os.makedirs("models", exist_ok=True)
            self.s3.download_file(
                self.bucket,
                "production/thermal_detector.pkl",
                "models/thermal_detector.pkl",
            )
            return True
        except Exception as e:
            print(f"⚠️ S3 Production Download Failed: {e}")
            return False

    def upload_model(self, local_path, s3_path="production/thermal_detector.pkl"):
        """Promotes a local model to the S3 Production folder."""
        try:
            self.s3.upload_file(local_path, self.bucket, s3_path)
            print(f"🚀 Model promoted to S3: {s3_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to upload model to S3: {e}")
            return False

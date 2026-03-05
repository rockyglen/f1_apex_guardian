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
        local_path = f"data/{filename}"
        df.to_parquet(local_path)

        # Get the current MLflow run ID for version tracking
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "no_run"

        # Upload with Metadata tags so DagsHub/S3 can track the lineage
        self.s3.upload_file(
            local_path,
            self.bucket,
            f"feature_store/{filename}",
            ExtraArgs={"Metadata": {"mlflow_run_id": run_id}},
        )
        print(f"🚀 Features versioned & synced to S3: feature_store/{filename}")

    def download_production_model(self):
        """Pulls the current 'Champion' model for the Inference Pipeline."""
        try:
            os.makedirs("models", exist_ok=True)
            self.s3.download_file(
                self.bucket,
                "production/thermal_detector.pkl",
                "models/thermal_detector.pkl",
            )
            return True
        except Exception as e:
            print(f"⚠️ S3 Download Failed: {e}")
            return False

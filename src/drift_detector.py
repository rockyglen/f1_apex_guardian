import pandas as pd
from scipy.stats import ks_2samp
import mlflow
import os


class DriftDetector:
    def __init__(self, reference_df):
        """
        reference_df: This should be your 'Golden' Bahrain dataset.
        """
        self.reference_df = reference_df
        self.features = ["Speed", "Throttle", "RPM", "Acceleration"]

    def check_drift(self, current_df, threshold=0.05):
        """
        Compares Current Race Data vs. Reference Bahrain Data.
        Returns True if drift is detected (p-value < threshold).
        """
        drift_report = {}
        drift_detected = False

        print("\n🔍 Checking for Data Drift...")

        for feature in self.features:
            # Clean data for the test
            ref_data = self.reference_df[feature].dropna()
            curr_data = current_df[feature].dropna()

            # Perform Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(ref_data, curr_data)

            # If p-value is very small, the distributions are different
            is_drifting = p_value < threshold
            drift_report[feature] = {"p_value": p_value, "drift_detected": is_drifting}

            if is_drifting:
                drift_detected = True
                print(f"⚠️ DRIFT DETECTED in {feature} (p-value: {p_value:.4f})")
            else:
                print(f"✅ {feature} is stable.")

        # Log these results to DagsHub/MLflow
        for feat, metrics in drift_report.items():
            mlflow.log_metric(f"drift_p_value_{feat}", metrics["p_value"])

        return drift_detected, drift_report

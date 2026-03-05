import pandas as pd
import joblib
import os
from src.s3_manager import S3Manager


def predict_thermal_failure(df):
    """
    THE INFERENCE PIPELINE:
    Loads the trained AI Brain, runs predictions, and CLASSIFIES the anomalies.
    """
    s3 = S3Manager()
    model_path = "models/thermal_detector.pkl"

    # 1. Default setup
    df["Maintenance_Alert"] = False
    df["ML_Diagnosis"] = "Normal"  # NEW: Our specific diagnosis column

    if not os.path.exists(model_path):
        print("⚠️ No AI Brain found! Run model.py first.")
        return df

    # 2. Load the Brain
    model = joblib.load(model_path)
    features = ["Speed", "Throttle", "RPM", "Acceleration"]

    # 3. Filter for the exact conditions the AI was trained on
    high_speed_mask = (df["Speed"] >= 280) & (df["Throttle"] >= 90)
    inference_data = df[high_speed_mask][features].dropna()

    # 4. Run the Prediction
    if not inference_data.empty:
        predictions = model.predict(inference_data)
        df.loc[inference_data.index, "Maintenance_Alert"] = predictions == -1

        # The AI gets confused by strong corner exits. If the car is
        # accelerating beautifully (> 2.0 m/s²), it's a false positive. Veto it.
        false_positive_mask = (df["Maintenance_Alert"] == True) & (
            df["Acceleration"] > 2.0
        )
        df.loc[false_positive_mask, "Maintenance_Alert"] = False

        # --- NEW: THE DIAGNOSTIC CLASSIFIER ---
        # Look only at the rows the AI flagged
        anomalies = df[df["Maintenance_Alert"] == True]

        for index, row in anomalies.iterrows():
            # Rule 1: Did the driver lift their foot slightly? (Throttle < 95%)
            if row["Throttle"] < 95:
                df.at[index, "ML_Diagnosis"] = "📉 Driver Lift & Coast"

            # Rule 2: Is the car flying, but acceleration is dead? (Derating)
            elif row["Speed"] > 290 and row["Acceleration"] < 1.5:
                df.at[index, "ML_Diagnosis"] = "🔋 Severe High-Speed Derating"

            # Rule 3: The Catch-All (Mechanical/Aero issue)
            else:
                df.at[index, "ML_Diagnosis"] = "⚠️ Unknown Power Loss"

    return df

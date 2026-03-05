import pandas as pd  # Think of this as 'Excel on Steroids'


def detect_super_clipping(df):
    # STEP A: Create 'Acceleration'
    # .diff() compares one row to the previous row.
    # If speed was 300 and now it's 298, the result is -2.
    df["Acceleration"] = df["Speed"].diff()

    # STEP B: The Smoother
    # .rolling(window=5).mean() takes the last 5 rows and averages them.
    # This ignores tiny bumps in the track so we don't get 'fake' alarms.
    df["Accel_Smooth"] = df["Acceleration"].rolling(window=5).mean()

    # STEP C: The Logic Filter
    # In Pandas, we use '&' for 'AND'. We are looking for three things:
    # 1. Driver is at 100% gas (Throttle >= 99)
    # 2. Car is losing speed (Accel_Smooth <= 0)
    # 3. Car is moving fast (> 250 km/h)
    clipping_mask = (
        (df["Throttle"] >= 99) & (df["Accel_Smooth"] <= 0) & (df["Speed"] > 250)
    )

    # STEP D: Add a Label
    # We create a new column called 'Is_Clipping' and set it to True where the mask fits.
    df["Is_Clipping"] = clipping_mask

    return df


def detect_active_aero(df):
    """
    Finds where the car switches between X-mode (Straight) and Z-mode (Corner).
    Layman: We look for a 'free boost' in speed that isn't from the engine RPM.
    """
    # 1. Calculate RPM Delta (How much is the engine pushing?)
    df["RPM_Change"] = df["RPM"].diff()

    # 2. THE X-MODE DETECTOR:
    # If acceleration is INCREASING, but RPM is stable or dropping,
    # it means the car just got 'slicker' (X-mode opened).
    # We use a threshold of 55% drag reduction as our physics baseline.
    x_mode_condition = (df["Acceleration"] > 1.0) & (df["RPM_Change"] <= 0)

    # 3. THE Z-MODE DETECTOR:
    # If the car slows down suddenly on a straight but the driver is
    # still at 100% throttle, and it's NOT clipping, it's Z-mode (closing for a corner).
    z_mode_condition = (df["Acceleration"] < -1.0) & (df["Throttle"] >= 99)

    df["Aero_Mode"] = "Z-mode (Corner)"  # Default
    df.loc[x_mode_condition, "Aero_Mode"] = "X-mode (Straight)"

    return df


def master_diagnostic(df):
    """
    Layman: This is the 'Judge'. It looks at the speed drop
    and decides if it's a Strategy, Aero, or a Failure.
    """
    # 1. First, detect the basics
    df = detect_super_clipping(df)
    df = detect_active_aero(df)

    # 2. THE MASTER DECISION
    df["Diagnostic"] = "Optimal"

    # Case A: If it's clipping AND far from a corner -> STRATEGIC RECHARGE
    df.loc[
        df["Is_Clipping"] & (df["Aero_Mode"] == "X-mode (Straight)"), "Diagnostic"
    ] = "SUPER CLIPPING"

    # Case B: If it's slowing down AND Aero switched to Z-mode -> AERO RECOVERY
    df.loc[
        (df["Acceleration"] < -2.0) & (df["Aero_Mode"] == "Z-mode (Corner)"),
        "Diagnostic",
    ] = "Z-MODE TRANSITION"

    # Case C: If it's slowing down but NOT clipping and NOT Z-mode -> POTENTIAL FAILURE
    failure_mask = (
        (df["Acceleration"] < -1.0)
        & (~df["Is_Clipping"])
        & (df["Aero_Mode"] == "X-mode (Straight)")
    )
    df.loc[failure_mask, "Diagnostic"] = "WARNING: UNKNOWN POWER LOSS"

    return df

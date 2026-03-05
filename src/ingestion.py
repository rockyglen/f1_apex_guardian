import fastf1
import os
import pandas as pd
import streamlit as st
from datetime import datetime


def setup_cache():
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Enable FastF1's internal disk cache to avoid re-downloading from their API
    fastf1.Cache.enable_cache(cache_dir)


def get_latest_event_context(year=2026):
    """Checks the F1 calendar to find the most recent valid session."""
    schedule = fastf1.get_event_schedule(year)
    now = datetime.now()

    # Find races that have already concluded
    past_races = schedule[(schedule["EventDate"] < now) & (schedule["RoundNumber"] > 0)]

    if not past_races.empty:
        latest_race = past_races.iloc[-1]
        return "Race", latest_race["RoundNumber"], latest_race["EventName"]
    else:
        # 2026 Season hasn't started: Fallback to Bahrain Testing
        return "Testing", 2, "Bahrain Pre-Season Testing"


def fetch_telemetry_logic(year=2026):
    """The raw data logic, independent of Streamlit's UI decorators."""
    setup_cache()
    event_type, round_num, event_name = get_latest_event_context(year)

    if event_type == "Race":
        session = fastf1.get_session(year, round_num, "R")
    else:
        # Test 2, Day 3 is the standard deep-dive testing session
        session = fastf1.get_testing_session(year, 2, 3)

    session.load(telemetry=True)
    all_telemetry = []

    for drv in session.drivers:
        try:
            driver_info = session.get_driver(drv)
            driver_label = (
                f"{driver_info['Abbreviation']} ({driver_info['DriverNumber']})"
            )

            # Extract fastest lap and attach telemetry
            lap = session.laps.pick_drivers(drv).pick_fastest()
            tel = lap.get_telemetry().add_distance()
            tel["Driver"] = driver_label
            all_telemetry.append(tel)
        except Exception as e:
            continue

    if not all_telemetry:
        raise ValueError(f"Telemetry unavailable for {event_name}")

    return pd.concat(all_telemetry, ignore_index=True), event_name


# --- THE UI WRAPPER ---
@st.cache_data(ttl=3600)
def get_all_drivers_telemetry(year=2026):
    """Dashboard-only function that adds a caching layer."""
    return fetch_telemetry_logic(year)

import fastf1
import os
import pandas as pd
import streamlit as st
from datetime import datetime


def setup_cache():
    """
    Layman: Creates the memory folder.
    """
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)


def get_latest_event_context(year=2026):
    """
    LAYMAN: The AI Scout checks the F1 calendar. If a real race has happened,
    it returns the race details. If the season hasn't started yet, it defaults to Testing.
    """
    schedule = fastf1.get_event_schedule(year)
    now = datetime.now()

    # Filter for official races (RoundNumber > 0) that have already happened
    past_races = schedule[(schedule["EventDate"] < now) & (schedule["RoundNumber"] > 0)]

    if not past_races.empty:
        # A real race has happened! Grab the latest one.
        latest_race = past_races.iloc[-1]
        return "Race", latest_race["RoundNumber"], latest_race["EventName"]
    else:
        # No races yet. Fall back to testing so the pipeline survives.
        return "Testing", None, "Bahrain Pre-Season Testing"


@st.cache_data(ttl=3600)
def get_all_drivers_telemetry(year=2026):
    """
    Layman: Dynamically pulls the absolute latest valid data available,
    whether it's a Grand Prix or Pre-Season testing.
    """
    setup_cache()

    # 1. Ask the Calendar Scout what data we should pull today
    event_type, round_num, event_name = get_latest_event_context(year)
    print(f"🏎️ Auto-Pulling data for: {event_name}")

    # 2. Pull the correct session based on the Scout's report
    if event_type == "Race":
        # 'R' stands for the main Sunday Race
        session = fastf1.get_session(year, round_num, "R")
    else:
        # Hardcoded fallback to the testing data we know works (Test 2, Day 3)
        session = fastf1.get_testing_session(year, 2, 3)

    session.load(telemetry=True)
    all_telemetry = []

    # 3. Process the drivers
    for drv in session.drivers:
        try:
            driver_info = session.get_driver(drv)
            driver_label = (
                f"{driver_info['Abbreviation']} ({driver_info['DriverNumber']})"
            )

            lap = session.laps.pick_drivers(drv).pick_fastest()
            tel = lap.get_telemetry().add_distance()
            tel["Driver"] = driver_label

            all_telemetry.append(tel)

        except Exception as e:
            print(f"⚠️ Skipped driver {drv} because: {e}")
            continue

    if not all_telemetry:
        raise ValueError(f"Failed to collect telemetry for {event_name}.")

    master_df = pd.concat(all_telemetry, ignore_index=True)

    return master_df, event_name

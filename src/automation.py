import fastf1
from datetime import datetime


def get_latest_race_info():
    """
    LAYMAN: This function looks at today's date and finds
    the most recent race that actually happened.
    """
    # Get the 2026 schedule
    schedule = fastf1.get_event_schedule(2026)

    # Filter for races that have already passed today's date
    past_races = schedule[schedule["EventDate"] < datetime.now()]

    # Grab the very last one from that list
    latest_event = past_races.iloc[-1]

    return latest_event["EventName"], latest_event["RoundNumber"]

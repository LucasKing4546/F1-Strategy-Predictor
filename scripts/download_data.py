import sys
import time

import fastf1
import pandas as pd
import requests
from fastf1.exceptions import RateLimitExceededError
from fastf1.logger import LoggingManager

from f1strat.data.cache import enable_cache

SEASONS = range(2023, 2027)
SESSION_SLOTS = range(1, 6)
DATA_DELAY = pd.Timedelta(hours=6)
RATE_LIMIT_WAIT_S = 10 * 60
CHECKED_DATA = ("laps", "results", "weather_data", "race_control_messages", "car_data", "pos_data")


def finished_sessions(year: int, now: pd.Timestamp):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        for slot in SESSION_SLOTS:
            name, start = event[f"Session{slot}"], event[f"Session{slot}DateUtc"]
            if isinstance(name, str) and name and pd.notna(start) and start + DATA_DELAY < now:
                yield event, name


def load(session) -> None:
    while True:
        try:
            session.load(laps=True, telemetry=True, weather=True, messages=True)
            return
        except (RateLimitExceededError, requests.ConnectionError) as exc:
            print(f"  {type(exc).__name__}, waiting {RATE_LIMIT_WAIT_S // 60} min", flush=True)
            time.sleep(RATE_LIMIT_WAIT_S)
        except Exception as exc:
            print(f"  strict load failed ({type(exc).__name__}: {exc}), retrying leniently", flush=True)
            LoggingManager.debug = False
            try:
                session.load(laps=True, telemetry=True, weather=True, messages=True)
            finally:
                LoggingManager.debug = True
            return


def missing_data(session) -> list[str]:
    gaps = []
    for name in CHECKED_DATA:
        try:
            if len(getattr(session, name)) == 0:
                gaps.append(name)
        except Exception:
            gaps.append(name)
    return gaps


def main() -> int:
    enable_cache()
    fastf1.set_log_level("WARNING")
    LoggingManager.debug = True
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    incomplete = []
    for year in SEASONS:
        for event, name in finished_sessions(year, now):
            label = f"{year} R{event.RoundNumber} {event.EventName} {name}"
            print(label, flush=True)
            session = event.get_session(name)
            try:
                load(session)
                gaps = missing_data(session)
            except Exception as exc:
                gaps = [f"{type(exc).__name__}: {exc}"]
            if gaps:
                incomplete.append(f"{label}: {', '.join(gaps)}")
    print(f"\n{len(incomplete)} incomplete sessions", *incomplete, sep="\n")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())

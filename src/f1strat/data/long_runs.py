import fastf1
import numpy as np
import pandas as pd

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
COOLDOWN_FACTOR = 1.15
PACE_TOLERANCE = 0.03
MAX_LAP_GAP = 2
MIN_LAPS = 5

ID_COLUMNS = ["season", "round", "circuit", "session_type"]
RUN_COLUMNS = [
    "driver",
    "team",
    "stint",
    "compound",
    "start_tyre_age",
    "n_laps",
    "first_lap",
    "last_lap",
    "median_lap_time_s",
    "deg_slope_s_per_lap",
]


def _usable_laps(laps: pd.DataFrame) -> pd.DataFrame:
    lap_time_s = laps["LapTime"].dt.total_seconds()
    keep = (
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & lap_time_s.notna()
        & laps["TyreLife"].notna()
        & (laps["TrackStatus"] == "1")
        & laps["Compound"].isin(DRY_COMPOUNDS)
        & ~laps["Deleted"].eq(True)
    )
    return laps.assign(lap_time_s=lap_time_s)[keep]


def _candidate_runs(stint: pd.DataFrame) -> list[pd.DataFrame]:
    is_cooldown = stint["lap_time_s"] > stint["lap_time_s"].min() * COOLDOWN_FACTOR
    segment = is_cooldown.cumsum()[~is_cooldown]
    stint = stint[~is_cooldown]
    median = stint["lap_time_s"].median()
    in_pace = (stint["lap_time_s"] - median).abs() <= median * PACE_TOLERANCE
    stint, segment = stint[in_pace], segment[in_pace]
    breaks = (stint["LapNumber"].diff() > MAX_LAP_GAP) | (segment.diff() != 0)
    return [run for _, run in stint.groupby(breaks.cumsum())]


def _summarize(run: pd.DataFrame) -> dict:
    first = run.iloc[0]
    slope = np.polyfit(run["TyreLife"], run["lap_time_s"], 1)[0]
    return {
        "driver": first["Driver"],
        "team": first["Team"],
        "stint": int(first["Stint"]),
        "compound": first["Compound"],
        "start_tyre_age": int(run["TyreLife"].min()),
        "n_laps": len(run),
        "first_lap": int(run["LapNumber"].min()),
        "last_lap": int(run["LapNumber"].max()),
        "median_lap_time_s": float(run["lap_time_s"].median()),
        "deg_slope_s_per_lap": float(slope),
    }


def detect_long_runs(laps: pd.DataFrame) -> pd.DataFrame:
    if laps["Deleted"].isna().any():
        raise ValueError("Deleted flags missing; load the session with messages=True")
    rows = [
        _summarize(run)
        for _, stint in _usable_laps(laps).groupby(["Driver", "Stint"])
        for run in _candidate_runs(stint)
        if len(run) >= MIN_LAPS
    ]
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def practice_session_names(event: fastf1.events.Event) -> list[str]:
    names = (event[f"Session{i}"] for i in range(1, 6))
    return [name for name in names if str(name).startswith("Practice")]


def practice_long_runs(year: int, event_name: str) -> pd.DataFrame:
    event = fastf1.get_event(year, event_name)
    tables = []
    for name in practice_session_names(event):
        session = event.get_session(name)
        session.load(telemetry=False, messages=True)
        if session.weather_data["Rainfall"].any():
            continue
        runs = detect_long_runs(session.laps)
        tables.append(
            runs.assign(
                season=year,
                round=int(event["RoundNumber"]),
                circuit=event["Location"],
                session_type=f"FP{name[-1]}",
            )
        )
    if not tables:
        return pd.DataFrame(columns=ID_COLUMNS + RUN_COLUMNS)
    return pd.concat(tables, ignore_index=True)[ID_COLUMNS + RUN_COLUMNS]

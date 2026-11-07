import fastf1
import pandas as pd
from fastf1.core import Session
from fastf1.events import Event

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
PIT_LANE_GRID = 0
RACE_META_COLUMNS = [
    "season", "round", "event_name", "circuit", "event_format",
    "scheduled_laps", "completed_laps", "has_rainfall", "wet_tyre_laps", "is_wet",
]
GRID_COLUMNS = [
    "season", "round", "driver", "team", "grid_position", "is_pit_lane_start", "finish_position", "status",
]


def _race_rainfall(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.Series:
    during = weather["Time"].between(laps["LapStartTime"].min(), laps["Time"].max())
    if not during.any():
        raise ValueError("no weather samples during the race; load the session with weather=True")
    return weather.loc[during, "Rainfall"]


def build_race_meta(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    season: int,
    round_number: int,
    event_name: str,
    circuit: str,
    event_format: str,
    scheduled_laps: int,
) -> pd.DataFrame:
    if laps.empty:
        raise ValueError("laps is empty; load the session with laps=True")
    rainfall = _race_rainfall(laps, weather)
    if rainfall.isna().any():
        raise ValueError("Rainfall is missing during the race")
    has_rainfall = bool(rainfall.astype(bool).any())
    wet_tyre_laps = int(laps["Compound"].isin(WET_COMPOUNDS).sum())
    row = {
        "season": season,
        "round": round_number,
        "event_name": event_name,
        "circuit": circuit,
        "event_format": event_format,
        "scheduled_laps": int(scheduled_laps),
        "completed_laps": int(laps["LapNumber"].max()),
        "has_rainfall": has_rainfall,
        "wet_tyre_laps": wet_tyre_laps,
        "is_wet": has_rainfall or wet_tyre_laps > 0,
    }
    return pd.DataFrame([row], columns=RACE_META_COLUMNS)


def build_grid(results: pd.DataFrame, laps: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    starters = results[results["GridPosition"].notna()]
    missing = set(laps["Driver"]) - set(starters["Abbreviation"])
    if missing:
        raise ValueError(f"drivers with laps but no grid position: {', '.join(sorted(missing))}")
    pit_lane = starters["GridPosition"].eq(PIT_LANE_GRID)
    grid = pd.DataFrame(
        {
            "season": season,
            "round": round_number,
            "driver": starters["Abbreviation"],
            "team": starters["TeamName"],
            "grid_position": starters["GridPosition"].mask(pit_lane).astype("Int64"),
            "is_pit_lane_start": pit_lane,
            "finish_position": starters["Position"].astype("Int64"),
            "status": starters["Status"],
        },
        columns=GRID_COLUMNS,
    )
    return grid.sort_values(["grid_position", "driver"], na_position="last").reset_index(drop=True)


def _race(year: int, event_name: str) -> tuple[Event, Session]:
    event = fastf1.get_event(year, event_name)
    session = event.get_session("Race")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return event, session


def race_meta(year: int, event_name: str) -> pd.DataFrame:
    event, session = _race(year, event_name)
    return build_race_meta(
        session.laps,
        session.weather_data,
        season=year,
        round_number=int(event["RoundNumber"]),
        event_name=event["EventName"],
        circuit=event["Location"],
        event_format=event["EventFormat"],
        scheduled_laps=session.total_laps,
    )


def race_grid(year: int, event_name: str) -> pd.DataFrame:
    event, session = _race(year, event_name)
    return build_grid(session.results, session.laps, season=year, round_number=int(event["RoundNumber"]))

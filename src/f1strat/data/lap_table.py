import fastf1
import pandas as pd

from f1strat.data.long_runs import DRY_COMPOUNDS

KNOWN_COMPOUNDS = DRY_COMPOUNDS + ("INTERMEDIATE", "WET")
GREEN = "1"
TRAFFIC_GAP_S = 1.5
OUTLIER_TOLERANCE = 0.07

WEATHER_COLUMNS = {
    "TrackTemp": "track_temp",
    "AirTemp": "air_temp",
    "Humidity": "humidity",
    "Rainfall": "rainfall",
}
POSITION_COLUMNS = [
    "position",
    "gap_to_leader_s",
    "gap_ahead_s",
    "gap_behind_s",
    "car_ahead",
    "car_behind",
]
LAP_TABLE_COLUMNS = [
    "season", "round", "circuit", "session_type", "driver", "team", "lap_number",
    "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "lap_end_time_s", "is_accurate", "is_deleted",
    "compound", "tyre_age", "fresh_tyre", "stint", "compounds_used_so_far",
    "is_pit_in_lap", "is_pit_out_lap", "pit_lane_time_s", "track_status",
    "is_sc", "is_vsc", "is_yellow", "is_red_flag", "is_restart_lap", "is_clean_lap",
    *POSITION_COLUMNS, "in_traffic",
    "laps_remaining", "race_progress", "fuel_proxy",
    *WEATHER_COLUMNS.values(),
    "driver_ref_pace_s", "lap_delta_s",
]


def _seconds(values: pd.Series) -> pd.Series:
    return values.dt.total_seconds()


def _base(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.duplicated(["Driver", "LapNumber"]).any():
        raise ValueError("laps has duplicate (Driver, LapNumber) rows")
    if laps["Deleted"].isna().any():
        raise ValueError("Deleted flags missing; load the session with messages=True")
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    by_driver = laps.groupby("Driver")
    compound = laps["Compound"].where(laps["Compound"].isin(KNOWN_COMPOUNDS))
    first_use = compound.isin(DRY_COMPOUNDS) & ~pd.DataFrame(
        {"driver": laps["Driver"], "compound": compound}
    ).duplicated()
    next_is_consecutive = by_driver["LapNumber"].shift(-1) == laps["LapNumber"] + 1
    pit_lane = by_driver["PitOutTime"].shift(-1) - laps["PitInTime"]
    return pd.DataFrame(
        {
            "driver": laps["Driver"],
            "team": laps["Team"],
            "lap_number": laps["LapNumber"].astype(int),
            "lap_time_s": _seconds(laps["LapTime"]),
            "sector1_s": _seconds(laps["Sector1Time"]),
            "sector2_s": _seconds(laps["Sector2Time"]),
            "sector3_s": _seconds(laps["Sector3Time"]),
            "lap_end_time_s": _seconds(laps["Time"]),
            "is_accurate": laps["IsAccurate"].eq(True),
            "is_deleted": laps["Deleted"].eq(True),
            "compound": compound,
            "tyre_age": laps["TyreLife"],
            "fresh_tyre": laps["FreshTyre"],
            "stint": laps["Stint"].astype("Int64"),
            "compounds_used_so_far": first_use.groupby(laps["Driver"]).cumsum(),
            "is_pit_in_lap": laps["PitInTime"].notna(),
            "is_pit_out_lap": laps["PitOutTime"].notna(),
            "pit_lane_time_s": _seconds(pit_lane.where(next_is_consecutive)),
            "track_status": laps["TrackStatus"].fillna("").astype(str),
        }
    )


def _add_flags(table: pd.DataFrame) -> pd.DataFrame:
    status = table["track_status"]
    is_sc = status.str.contains("4")
    is_red_flag = status.str.contains("5")
    neutralised = is_sc | is_red_flag
    previous_is_consecutive = table.groupby("driver")["lap_number"].diff() == 1
    previous_neutralised = neutralised.groupby(table["driver"]).shift(fill_value=False) & previous_is_consecutive
    return table.assign(
        is_sc=is_sc,
        is_vsc=status.str.contains("[67]"),
        is_yellow=status.str.contains("2"),
        is_red_flag=is_red_flag,
        is_restart_lap=previous_neutralised & ~neutralised,
    )


def _add_positions(table: pd.DataFrame) -> pd.DataFrame:
    timed = table[table["lap_end_time_s"].notna()].sort_values(["lap_number", "lap_end_time_s"])
    by_lap = timed.groupby("lap_number")
    crossing = by_lap["lap_end_time_s"]
    positions = pd.DataFrame(
        {
            "position": by_lap.cumcount() + 1,
            "gap_to_leader_s": timed["lap_end_time_s"] - crossing.transform("min"),
            "gap_ahead_s": crossing.diff(),
            "gap_behind_s": -crossing.diff(-1),
            "car_ahead": by_lap["driver"].shift(),
            "car_behind": by_lap["driver"].shift(-1),
        }
    )
    table = table.join(positions)
    return table.assign(
        position=table["position"].astype("Int64"),
        in_traffic=table["gap_ahead_s"] < TRAFFIC_GAP_S,
    )


def _add_progress(table: pd.DataFrame) -> pd.DataFrame:
    total_laps = int(table["lap_number"].max())
    laps_remaining = total_laps - table["lap_number"]
    return table.assign(
        laps_remaining=laps_remaining,
        race_progress=table["lap_number"] / total_laps,
        fuel_proxy=laps_remaining / total_laps,
    )


def _add_weather(table: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    samples = (
        weather.assign(lap_end_time_s=_seconds(weather["Time"]))
        .sort_values("lap_end_time_s")[["lap_end_time_s", *WEATHER_COLUMNS]]
        .rename(columns=WEATHER_COLUMNS)
    )
    timed = table.loc[table["lap_end_time_s"].notna(), ["lap_end_time_s"]].sort_values("lap_end_time_s")
    matched = pd.merge_asof(timed.reset_index(), samples, on="lap_end_time_s", direction="backward")
    table = table.join(matched.set_index("index")[list(WEATHER_COLUMNS.values())])
    return table.astype({"rainfall": "boolean"})


def _add_clean_and_pace(table: pd.DataFrame) -> pd.DataFrame:
    candidate = (
        table["track_status"].eq(GREEN)
        & ~table["is_pit_in_lap"]
        & ~table["is_pit_out_lap"]
        & table["is_accurate"]
        & ~table["is_deleted"]
        & (table["lap_number"] > 1)
        & ~table["is_restart_lap"]
        & table["compound"].isin(DRY_COMPOUNDS)
        & table["lap_time_s"].notna()
    )
    lap_time = table["lap_time_s"]
    stint_median = lap_time.where(candidate).groupby([table["driver"], table["stint"]]).transform("median")
    is_clean = candidate & ((lap_time - stint_median).abs() <= stint_median * OUTLIER_TOLERANCE)
    ref_pace = lap_time.where(is_clean).groupby(table["driver"]).transform("median")
    return table.assign(is_clean_lap=is_clean, driver_ref_pace_s=ref_pace, lap_delta_s=lap_time - ref_pace)


def build_lap_table(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    season: int,
    round_number: int,
    circuit: str,
    session_type: str = "R",
) -> pd.DataFrame:
    table = _add_flags(_base(laps))
    table = _add_positions(table)
    table = _add_progress(table)
    table = _add_weather(table, weather)
    table = _add_clean_and_pace(table)
    table = table.assign(season=season, round=round_number, circuit=circuit, session_type=session_type)
    return table[LAP_TABLE_COLUMNS]


def race_lap_table(year: int, event_name: str) -> pd.DataFrame:
    event = fastf1.get_event(year, event_name)
    session = event.get_session("Race")
    session.load(telemetry=False, messages=True)
    return build_lap_table(
        session.laps,
        session.weather_data,
        season=year,
        round_number=int(event["RoundNumber"]),
        circuit=event["Location"],
    )

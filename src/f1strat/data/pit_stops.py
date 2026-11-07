import pandas as pd

from f1strat.data.lap_table import race_lap_table
from f1strat.data.race_state import RACE_KEY

PIT_STOP_COLUMNS = [
    "season", "round", "driver", "team", "lap", "pit_lane_time_s",
    "compound_before", "compound_after", "tyre_age_before", "is_tyre_change",
    "is_sc", "is_vsc", "is_red_flag",
]


def build_pit_stops(table: pd.DataFrame) -> pd.DataFrame:
    if len(table[RACE_KEY].drop_duplicates()) != 1:
        raise ValueError("table must hold exactly one race session")
    table = table.sort_values(["driver", "lap_number"]).reset_index(drop=True)
    after = table.groupby("driver")[["lap_number", "compound", "stint", "is_pit_out_lap"]].shift(-1)
    completed = (
        table["is_pit_in_lap"]
        & after["is_pit_out_lap"].eq(True)
        & after["lap_number"].eq(table["lap_number"] + 1)
    )
    unmatched = table[table["is_pit_in_lap"] & ~completed & after["lap_number"].notna()]
    if not unmatched.empty:
        laps = ", ".join(f"{row.driver} lap {row.lap_number}" for row in unmatched.itertuples())
        raise ValueError(f"in-lap without a following out-lap: {laps}")
    stops, after = table[completed], after[completed]
    return pd.DataFrame(
        {
            "season": stops["season"],
            "round": stops["round"],
            "driver": stops["driver"],
            "team": stops["team"],
            "lap": stops["lap_number"],
            "pit_lane_time_s": stops["pit_lane_time_s"],
            "compound_before": stops["compound"],
            "compound_after": after["compound"],
            "tyre_age_before": stops["tyre_age"],
            "is_tyre_change": after["stint"].astype("Int64") != stops["stint"],
            "is_sc": stops["is_sc"],
            "is_vsc": stops["is_vsc"],
            "is_red_flag": stops["is_red_flag"],
        },
        columns=PIT_STOP_COLUMNS,
    ).reset_index(drop=True)


def race_pit_stops(year: int, event_name: str) -> pd.DataFrame:
    return build_pit_stops(race_lap_table(year, event_name))

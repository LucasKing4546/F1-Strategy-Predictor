from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from f1strat.data.long_runs import DRY_COMPOUNDS
from f1strat.domain.entities import CarState, LapTiming, RaceState

RACE_KEY = ["season", "round", "session_type"]
TIMING_COLUMNS = ["lap_time_s", "sector1_s", "sector2_s", "sector3_s"]
T = TypeVar("T")


def _optional(value: object, cast: Callable[[Any], T]) -> T | None:
    return None if pd.isna(value) else cast(value)


def _timing(values: pd.Series) -> LapTiming:
    return LapTiming(*(_optional(values[column], float) for column in TIMING_COLUMNS))


def _car(row: pd.Series, compounds_used: frozenset[str], best: pd.Series) -> CarState:
    return CarState(
        driver=row["driver"],
        team=row["team"],
        position=int(row["position"]),
        gap_to_leader_s=float(row["gap_to_leader_s"]),
        gap_ahead_s=_optional(row["gap_ahead_s"], float),
        gap_behind_s=_optional(row["gap_behind_s"], float),
        car_ahead=_optional(row["car_ahead"], str),
        car_behind=_optional(row["car_behind"], str),
        compound=_optional(row["compound"], str),
        tyre_age=_optional(row["tyre_age"], int),
        stint=_optional(row["stint"], int),
        compounds_used=compounds_used,
        pit_in=bool(row["is_pit_in_lap"]),
        pit_out=bool(row["is_pit_out_lap"]),
        last=_timing(row),
        last_deleted=bool(row["is_deleted"]),
        best=_timing(best),
    )


def race_state(table: pd.DataFrame, lap: int) -> RaceState:
    if len(table[RACE_KEY].drop_duplicates()) != 1:
        raise ValueError("table must hold exactly one race session")
    rows = table[(table["lap_number"] == lap) & table["position"].notna()].sort_values("position")
    if rows.empty:
        raise ValueError(f"no timed cars on lap {lap}")
    so_far = table[table["lap_number"] <= lap]
    used = so_far[so_far["compound"].isin(DRY_COMPOUNDS)].groupby("driver")["compound"].agg(frozenset)
    valid = so_far.loc[~so_far["is_deleted"], ["driver", *TIMING_COLUMNS]]
    best = valid.groupby("driver")[TIMING_COLUMNS].min().reindex(rows["driver"])
    first = rows.iloc[0]
    return RaceState(
        season=int(first["season"]),
        round=int(first["round"]),
        circuit=first["circuit"],
        lap=int(lap),
        laps_remaining=int(first["laps_remaining"]),
        is_sc=bool(rows["is_sc"].any()),
        is_vsc=bool(rows["is_vsc"].any()),
        is_red_flag=bool(rows["is_red_flag"].any()),
        fastest=_timing(valid[TIMING_COLUMNS].min()),
        cars=tuple(
            _car(row, used.get(row["driver"], frozenset()), best.loc[row["driver"]])
            for _, row in rows.iterrows()
        ),
    )

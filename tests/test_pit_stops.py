import numpy as np
import pandas as pd
import pytest

from f1strat.data.pit_stops import PIT_STOP_COLUMNS, build_pit_stops


def _driver(driver, compounds, stints, pit_in=(), red=()):
    rows, age = [], 0
    for index, (compound, stint) in enumerate(zip(compounds, stints)):
        lap = index + 1
        age = 1 if index == 0 or stint != stints[index - 1] else age + 1
        completed = lap in pit_in and lap < len(compounds)
        rows.append(
            {
                "season": 2026,
                "round": 1,
                "session_type": "R",
                "driver": driver,
                "team": f"Team {driver}",
                "lap_number": lap,
                "compound": compound,
                "stint": stint,
                "tyre_age": float(age),
                "is_pit_in_lap": lap in pit_in,
                "is_pit_out_lap": lap - 1 in pit_in,
                "pit_lane_time_s": 22.0 + lap if completed else np.nan,
                "is_sc": False,
                "is_vsc": False,
                "is_red_flag": lap in red,
            }
        )
    return rows


def _table(*drivers):
    rows = [row for driver in drivers for row in driver]
    return pd.DataFrame(rows).astype({"stint": "Int64"})


def test_tyre_change_stop():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,)))

    stops = build_pit_stops(table)

    assert list(stops.columns) == PIT_STOP_COLUMNS
    stop = stops.iloc[0]
    assert (stop["driver"], stop["team"], stop["lap"]) == ("AAA", "Team AAA", 3)
    assert (stop["compound_before"], stop["compound_after"]) == ("MEDIUM", "HARD")
    assert stop["tyre_age_before"] == 3
    assert stop["pit_lane_time_s"] == pytest.approx(25.0)
    assert stop["is_tyre_change"]


def test_stops_of_several_drivers_are_ordered_by_driver_and_lap():
    table = _table(
        _driver("BBB", ["SOFT"] * 2 + ["HARD"] * 4, [1] * 2 + [2] * 4, pit_in=(2,)),
        _driver("AAA", ["SOFT"] * 2 + ["MEDIUM"] * 2 + ["HARD"] * 2, [1, 1, 2, 2, 3, 3], pit_in=(2, 4)),
    )

    stops = build_pit_stops(table)

    assert list(zip(stops["driver"], stops["lap"])) == [("AAA", 2), ("AAA", 4), ("BBB", 2)]
    assert list(stops["compound_after"]) == ["MEDIUM", "HARD", "HARD"]


def test_drive_through_is_not_a_tyre_change():
    table = _table(_driver("AAA", ["MEDIUM"] * 6, [1] * 6, pit_in=(3,)))

    stop = build_pit_stops(table).iloc[0]

    assert (stop["compound_before"], stop["compound_after"]) == ("MEDIUM", "MEDIUM")
    assert not stop["is_tyre_change"]


def test_retirement_in_the_pits_is_not_a_stop():
    table = _table(
        _driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,)),
        _driver("BBB", ["MEDIUM"] * 4, [1] * 4, pit_in=(4,)),
    )

    stops = build_pit_stops(table)

    assert list(stops["driver"]) == ["AAA"]


def test_red_flag_stop_is_flagged():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,), red=(3,)))

    stop = build_pit_stops(table).iloc[0]

    assert stop["is_red_flag"]
    assert not stop["is_sc"]


def test_flags_come_from_the_in_lap():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,), red=(4,)))

    stop = build_pit_stops(table).iloc[0]

    assert not stop["is_red_flag"]


def test_back_to_back_stops():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] + ["SOFT"] * 2, [1, 1, 1, 2, 3, 3], pit_in=(3, 4)))

    stops = build_pit_stops(table)

    assert list(stops["lap"]) == [3, 4]
    assert list(stops["compound_after"]) == ["HARD", "SOFT"]


def test_unknown_stint_gives_unknown_tyre_change():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,)))
    table.loc[3, "stint"] = pd.NA

    stops = build_pit_stops(table)

    assert stops["is_tyre_change"].dtype == "boolean"
    assert pd.isna(stops["is_tyre_change"].iloc[0])


def test_stop_with_unknown_compound_after_is_kept():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + [None] * 3, [1] * 3 + [2] * 3, pit_in=(3,)))

    stop = build_pit_stops(table).iloc[0]

    assert pd.isna(stop["compound_after"])
    assert stop["is_tyre_change"]


def test_in_lap_without_out_lap_before_the_end_raises():
    table = _table(_driver("AAA", ["MEDIUM"] * 3 + ["HARD"] * 3, [1] * 3 + [2] * 3, pit_in=(3,)))
    table = table[table["lap_number"] != 4]

    with pytest.raises(ValueError, match="AAA lap 3"):
        build_pit_stops(table)


def test_race_without_stops_is_empty():
    stops = build_pit_stops(_table(_driver("AAA", ["MEDIUM"] * 4, [1] * 4)))

    assert stops.empty
    assert list(stops.columns) == PIT_STOP_COLUMNS


def test_table_with_two_races_raises():
    first = _driver("AAA", ["MEDIUM"] * 3, [1] * 3)
    second = [{**row, "round": 2} for row in first]

    with pytest.raises(ValueError, match="one race"):
        build_pit_stops(_table(first, second))

import dataclasses

import numpy as np
import pandas as pd
import pytest

from f1strat.data.race_state import race_state
from f1strat.domain.entities import LapTiming, Mark
from test_lap_table import _table

ALL_NONE = (Mark.NONE,) * 4


def _set(table, driver, lap, **values):
    rows = (table["driver"] == driver) & (table["lap_number"] == lap)
    for column, value in values.items():
        table.loc[rows, column] = value
    return table


def test_last_lap_and_bests_only_use_laps_so_far():
    table = _table()
    early, late = race_state(table, 2), race_state(table, 8)

    assert early.fastest.lap_s == pytest.approx(90.0)
    assert late.fastest.lap_s == pytest.approx(89.0)
    assert late.car("AAA").last.lap_s == pytest.approx(90.0)
    assert late.car("AAA").best.lap_s == pytest.approx(90.0)
    assert late.marks("BBB") == (Mark.OVERALL,) * 4
    assert late.marks("AAA") == (Mark.PERSONAL,) * 4


def test_slower_lap_has_no_mark():
    state = race_state(_table(), 5)

    assert state.car("AAA").last.lap_s == pytest.approx(99.0)
    assert state.marks("AAA") == ALL_NONE


def test_sector_bests_are_independent_of_laps():
    table = _set(_table(), "AAA", 3, sector1_s=20.0)
    table = _set(table, "BBB", 4, sector2_s=20.0)
    state = race_state(table, 4)

    assert state.car("AAA").best == LapTiming(90.0, 20.0, 30.0, 30.0)
    assert state.fastest == LapTiming(90.0, 20.0, 20.0, 30.0)
    assert state.marks("AAA") == (Mark.OVERALL, Mark.NONE, Mark.PERSONAL, Mark.OVERALL)


def test_deleted_lap_does_not_count_as_best():
    table = _set(_table(), "CCC", 5, lap_time_s=80.0, sector1_s=10.0, is_deleted=True)
    state = race_state(table, 5)

    assert state.car("CCC").last.lap_s == pytest.approx(80.0)
    assert state.car("CCC").best.lap_s == pytest.approx(100.0)
    assert state.fastest.lap_s == pytest.approx(90.0)
    assert state.fastest.sector1_s == pytest.approx(30.0)
    assert state.marks("CCC") == ALL_NONE


def test_deleted_lap_matching_a_best_has_no_mark():
    table = _set(_table(), "AAA", 8, is_deleted=True)

    assert race_state(table, 8).car("AAA").last_deleted
    assert race_state(table, 8).marks("AAA") == ALL_NONE


def test_driver_with_only_deleted_laps_has_no_bests():
    table = _table()
    table.loc[table["driver"] == "CCC", "is_deleted"] = True
    state = race_state(table, 2)

    assert state.car("CCC").best == LapTiming(None, None, None, None)
    assert state.marks("CCC") == ALL_NONE


def test_missing_sector_is_none_with_no_mark():
    table = _set(_table(), "AAA", 2, sector1_s=np.nan)
    state = race_state(table, 2)

    assert state.car("AAA").last.sector1_s is None
    assert state.marks("AAA")[1] is Mark.NONE


def test_car_lookup_by_driver():
    state = race_state(_table(), 1)

    assert state.car("BBB").position == 2
    with pytest.raises(KeyError, match="ZZZ"):
        state.car("ZZZ")


def test_neutralised_under_any_caution():
    table = _table()

    assert race_state(table, 3).is_neutralised
    assert race_state(table, 6).is_neutralised
    assert not race_state(table, 2).is_neutralised


def test_cars_ordered_by_position_with_gaps():
    state = race_state(_table(), 1)
    leader, second, third = state.cars

    assert [car.driver for car in state.cars] == ["AAA", "BBB", "CCC"]
    assert [car.position for car in state.cars] == [1, 2, 3]
    assert leader.gap_ahead_s is None and leader.car_ahead is None
    assert leader.gap_to_leader_s == pytest.approx(0.0)
    assert (second.car_ahead, second.car_behind) == ("AAA", "CCC")
    assert second.gap_ahead_s == pytest.approx(1.0)
    assert third.gap_to_leader_s == pytest.approx(5.0)
    assert third.gap_behind_s is None and third.car_behind is None


def test_race_level_fields():
    state = race_state(_table(), 1)

    assert (state.season, state.round, state.circuit, state.lap) == (2025, 14, "Budapest", 1)
    assert state.laps_remaining == 7
    assert not (state.is_sc or state.is_vsc or state.is_red_flag)


def test_track_flags():
    table = _table()

    assert race_state(table, 3).is_vsc
    assert race_state(table, 6).is_sc
    assert not race_state(table, 7).is_sc


def test_flag_on_one_car_sets_race_flag():
    table = _table()
    table.loc[(table["driver"] == "CCC") & (table["lap_number"] == 1), "is_sc"] = True

    assert race_state(table, 1).is_sc


def test_pit_status_and_tyres():
    table = _table()
    in_lap, out_lap = race_state(table, 4).car("BBB"), race_state(table, 5).car("BBB")

    assert in_lap.pit_in and not in_lap.pit_out
    assert (in_lap.compound, in_lap.stint, in_lap.compounds_used) == ("MEDIUM", 1, frozenset({"MEDIUM"}))
    assert out_lap.pit_out and not out_lap.pit_in
    assert (out_lap.compound, out_lap.tyre_age, out_lap.stint) == ("HARD", 1, 2)
    assert out_lap.compounds_used == frozenset({"MEDIUM", "HARD"})


def test_unknown_compound_is_none_but_counts_earlier_use():
    car = race_state(_table(), 2).car("CCC")

    assert car.compound is None and car.tyre_age is None
    assert car.compounds_used == frozenset({"SOFT"})


def test_car_without_crossing_time_is_left_out():
    state = race_state(_table(), 3)

    assert [car.driver for car in state.cars] == ["AAA", "BBB"]
    assert state.cars[1].car_behind is None


def test_values_are_plain_python_types():
    car = race_state(_table(), 5).car("BBB")

    assert type(car.position) is int and type(car.tyre_age) is int and type(car.stint) is int
    assert type(car.gap_to_leader_s) is float and type(car.pit_out) is bool


def test_state_is_frozen():
    state = race_state(_table(), 1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.cars[0].position = 2


def test_lap_without_timed_cars_raises():
    with pytest.raises(ValueError, match="lap 9"):
        race_state(_table(), 9)


def test_table_with_several_races_raises():
    table = _table()

    with pytest.raises(ValueError, match="one race"):
        race_state(pd.concat([table, table.assign(round=15)]), 1)

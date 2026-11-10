import numpy as np
import pandas as pd
import pytest

from f1strat.data.race_meta import GRID_COLUMNS, RACE_META_COLUMNS, build_grid, build_race_meta, rain_slowdown

LAP_S = 100.0


def _laps(compounds_by_driver):
    rows = [
        {
            "Driver": driver,
            "LapNumber": float(lap),
            "LapStartTime": pd.Timedelta(seconds=(lap - 1) * LAP_S),
            "Time": pd.Timedelta(seconds=lap * LAP_S),
            "Compound": compound,
        }
        for driver, compounds in compounds_by_driver.items()
        for lap, compound in enumerate(compounds, start=1)
    ]
    return pd.DataFrame(rows)


def _weather(samples):
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([time_s for time_s, _ in samples], unit="s"),
            "Rainfall": [rain for _, rain in samples],
        }
    )


def _meta(laps, weather, scheduled_laps=3, rain_slowdown_s=0.0):
    return build_race_meta(
        _laps(laps),
        _weather(weather),
        season=2026,
        round_number=12,
        event_name="Dutch Grand Prix",
        circuit="Zandvoort",
        event_format="sprint_qualifying",
        scheduled_laps=scheduled_laps,
        rain_slowdown_s=rain_slowdown_s,
    )


def _results(rows):
    return pd.DataFrame(rows, columns=["Abbreviation", "TeamName", "GridPosition", "Position", "Status"])


DRY = {"AAA": ["MEDIUM"] * 3, "BBB": ["SOFT", "SOFT", "HARD"]}
DRY_WEATHER = [(50, False), (250, False)]


def test_dry_race():
    meta = _meta(DRY, DRY_WEATHER)

    assert list(meta.columns) == RACE_META_COLUMNS
    row = meta.iloc[0]
    assert (row["season"], row["round"], row["circuit"]) == (2026, 12, "Zandvoort")
    assert (row["event_name"], row["event_format"]) == ("Dutch Grand Prix", "sprint_qualifying")
    assert (row["scheduled_laps"], row["completed_laps"]) == (3, 3)
    assert (row["has_rainfall"], row["wet_tyre_laps"], row["is_wet"]) == (False, 0, False)
    assert row["rain_slowdown_s"] == 0.0


def test_rain_that_slows_the_laps_makes_the_race_wet():
    row = _meta(DRY, [(50, False), (150, True)], rain_slowdown_s=2.2).iloc[0]

    assert row["has_rainfall"]
    assert row["is_wet"]


def test_rain_that_does_not_slow_the_laps_keeps_the_race_dry():
    row = _meta(DRY, [(50, False), (150, True)], rain_slowdown_s=0.2).iloc[0]

    assert row["has_rainfall"]
    assert not row["is_wet"]


def test_rain_with_unmeasurable_effect_makes_the_race_wet():
    row = _meta(DRY, [(50, False), (150, True)], rain_slowdown_s=np.nan).iloc[0]

    assert row["is_wet"]


@pytest.mark.parametrize("rain_s", [-600, 450])
def test_rainfall_before_the_start_or_after_the_flag_does_not_count(rain_s):
    row = _meta(DRY, [(rain_s, True), *DRY_WEATHER]).iloc[0]

    assert not row["has_rainfall"]
    assert not row["is_wet"]


def test_wet_tyres_without_rainfall_make_the_race_wet():
    laps = {"AAA": ["INTERMEDIATE", "INTERMEDIATE", "MEDIUM"], "BBB": ["WET", "MEDIUM", "MEDIUM"]}

    row = _meta(laps, DRY_WEATHER).iloc[0]

    assert not row["has_rainfall"]
    assert row["wet_tyre_laps"] == 3
    assert row["is_wet"]


def test_shortened_race_keeps_scheduled_laps():
    row = _meta(DRY, DRY_WEATHER, scheduled_laps=5).iloc[0]

    assert (row["scheduled_laps"], row["completed_laps"]) == (5, 3)


@pytest.mark.parametrize("weather", [[], [(-600, False)]])
def test_no_weather_during_the_race_raises(weather):
    with pytest.raises(ValueError, match="weather"):
        _meta(DRY, weather)


def test_unknown_rainfall_during_the_race_raises():
    with pytest.raises(ValueError, match="Rainfall"):
        _meta(DRY, [(50, False), (150, None)])


def test_empty_laps_raises():
    with pytest.raises(ValueError, match="laps"):
        _meta({}, DRY_WEATHER)


def test_grid_with_pit_lane_starts_and_withdrawal():
    results = _results(
        [
            ("AAA", "Team A", 2.0, 1.0, "Finished"),
            ("EEE", "Team E", 0.0, 2.0, "Finished"),
            ("BBB", "Team B", 0.0, 3.0, "Lapped"),
            ("CCC", "Team C", 1.0, 4.0, "Retired"),
            ("DDD", "Team D", np.nan, np.nan, "Withdrew"),
        ]
    )
    laps = _laps({driver: ["MEDIUM"] for driver in ("AAA", "BBB", "CCC", "EEE")})

    grid = build_grid(results, laps, season=2026, round_number=12)

    assert list(grid.columns) == GRID_COLUMNS
    assert list(grid["driver"]) == ["CCC", "AAA", "BBB", "EEE"]
    bbb = grid.set_index("driver").loc["BBB"]
    assert pd.isna(bbb["grid_position"])
    assert bbb["is_pit_lane_start"]
    assert bbb["finish_position"] == 3
    assert bbb["status"] == "Lapped"
    assert grid["grid_position"].dtype == "Int64"
    assert (grid["season"] == 2026).all() and (grid["round"] == 12).all()


@pytest.mark.parametrize("bbb_result", [[], [("BBB", "Team B", np.nan, np.nan, "Withdrew")]])
def test_driver_with_laps_but_no_grid_position_raises(bbb_result):
    results = _results([("AAA", "Team A", 1.0, 1.0, "Finished"), *bbb_result])
    laps = _laps({"AAA": ["MEDIUM"], "BBB": ["MEDIUM"]})

    with pytest.raises(ValueError, match="BBB"):
        build_grid(results, laps, season=2026, round_number=12)


def _clean_laps(deltas, rain_laps, clean=None):
    laps = range(1, len(deltas) + 1)
    return pd.DataFrame(
        {
            "lap_number": list(laps),
            "rainfall": pd.array([lap in rain_laps for lap in laps], dtype="boolean"),
            "is_clean_lap": [True] * len(deltas) if clean is None else clean,
            "lap_delta_s": deltas,
        }
    )


def test_rain_slowdown_compares_rain_laps_with_nearby_dry_laps():
    deltas = [5.0] + [0.0] * 6 + [2.0, 2.5] + [0.0] * 6 + [9.0]
    table = _clean_laps(deltas, rain_laps={8, 9})

    assert rain_slowdown(table) == pytest.approx(2.25)


def test_no_rain_laps_means_no_slowdown():
    assert rain_slowdown(_clean_laps([0.0] * 5, rain_laps=set())) == 0.0


def test_rain_only_on_unclean_laps_cannot_be_measured():
    table = _clean_laps([0.0] * 5, rain_laps={3}, clean=[True, True, False, True, True])

    assert np.isnan(rain_slowdown(table))

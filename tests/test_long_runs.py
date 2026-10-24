import pandas as pd
import pytest

from f1strat.data.long_runs import detect_long_runs


def _lap(driver, stint, lap, time_s, age, compound="MEDIUM", status="1", pit_in=False, pit_out=False):
    return {
        "Driver": driver,
        "Team": "Team",
        "Stint": float(stint),
        "LapNumber": float(lap),
        "LapTime": pd.to_timedelta(time_s, unit="s"),
        "Compound": compound,
        "TyreLife": float(age),
        "PitInTime": pd.Timedelta(seconds=1) if pit_in else pd.NaT,
        "PitOutTime": pd.Timedelta(seconds=1) if pit_out else pd.NaT,
        "TrackStatus": status,
        "Deleted": False,
    }


def _long_run():
    laps = [_lap("AAA", 1, 1, 100.0, 1, pit_out=True)]
    for lap in range(2, 15):
        age = lap
        time_s = 82.0 + 0.1 * age
        status = "12" if lap == 10 else "1"
        if lap in (6, 12):
            time_s = 86.0
        laps.append(_lap("AAA", 1, lap, time_s, age, status=status))
    laps.append(_lap("AAA", 1, 15, 95.0, 15, pit_in=True))
    return laps


def _quali_sim():
    times = [76.5, 110.0, 76.6, 110.0, 76.4, 110.0, 76.7, 110.0, 76.5]
    return [_lap("BBB", 1, lap, t, lap, compound="SOFT") for lap, t in enumerate(times, start=2)]


def _short_stint():
    return [_lap("CCC", 1, lap, 82.0 + 0.1 * lap, lap) for lap in range(2, 6)]


def _unknown_compound():
    return [_lap("DDD", 1, lap, 82.0 + 0.1 * lap, lap, compound="None") for lap in range(2, 12)]


def test_detects_only_the_race_pace_run():
    laps = pd.DataFrame(_long_run() + _quali_sim() + _short_stint() + _unknown_compound())

    runs = detect_long_runs(laps)

    assert runs["driver"].tolist() == ["AAA"]
    run = runs.iloc[0]
    assert run["n_laps"] == 10
    assert (run["first_lap"], run["last_lap"]) == (2, 14)
    assert run["start_tyre_age"] == 2
    assert run["compound"] == "MEDIUM"
    assert run["deg_slope_s_per_lap"] == pytest.approx(0.1)


def test_returns_empty_table_with_columns():
    runs = detect_long_runs(pd.DataFrame(_quali_sim()))

    assert runs.empty
    assert "deg_slope_s_per_lap" in runs.columns


def test_missing_deleted_flags_raise():
    laps = pd.DataFrame(_long_run()).assign(Deleted=None)

    with pytest.raises(ValueError, match="Deleted"):
        detect_long_runs(laps)

import numpy as np
import pandas as pd
import pytest

from f1strat.data.lap_table import build_lap_table

STATUS = {3: "26", 6: "4"}


def _rows(driver, times, compounds):
    rows, end, stint, age = [], 0.0, 0, 0
    for lap, (time_s, compound) in enumerate(zip(times, compounds), start=1):
        end += time_s
        if lap == 1 or compound != compounds[lap - 2]:
            stint, age = stint + 1, 0
        age += 1
        rows.append(
            {
                "Driver": driver,
                "Team": f"Team {driver}",
                "LapNumber": float(lap),
                "LapTime": pd.to_timedelta(time_s, unit="s"),
                "Time": pd.to_timedelta(end, unit="s"),
                "Sector1Time": pd.to_timedelta(time_s / 3, unit="s"),
                "Sector2Time": pd.to_timedelta(time_s / 3, unit="s"),
                "Sector3Time": pd.to_timedelta(time_s / 3, unit="s"),
                "Compound": compound,
                "TyreLife": float(age),
                "FreshTyre": True,
                "Stint": float(stint),
                "TrackStatus": STATUS.get(lap, "1"),
                "IsAccurate": lap > 1,
                "Deleted": False,
            }
        )
    return rows


def _race():
    laps = pd.DataFrame(
        _rows("AAA", [95, 90, 90, 90, 99, 120, 92, 90], ["MEDIUM"] * 8)
        + _rows("BBB", [96, 90, 90, 112, 110, 115, 91, 89], ["MEDIUM"] * 4 + ["HARD"] * 4)
        + _rows("CCC", [100, 110, 110, 110, 110, 130, 110], ["SOFT"] * 7)
    )
    for column in ("PitInTime", "PitOutTime"):
        laps[column] = pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]")
    bbb = laps["Driver"] == "BBB"
    laps.loc[bbb & (laps["LapNumber"] == 4), "PitInTime"] = pd.Timedelta(seconds=380)
    laps.loc[bbb & (laps["LapNumber"] == 5), "PitOutTime"] = pd.Timedelta(seconds=402)
    laps.loc[bbb & laps["LapNumber"].isin([4, 5]), "IsAccurate"] = False
    ccc = laps["Driver"] == "CCC"
    laps.loc[ccc & (laps["LapNumber"] == 2), ["Compound", "TyreLife"]] = ["None", np.nan]
    laps.loc[ccc & (laps["LapNumber"] == 3), "Time"] = pd.NaT
    laps.loc[ccc & (laps["LapNumber"] == 5), "Deleted"] = True
    return laps


def _weather():
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 400], unit="s"),
            "TrackTemp": [40.0, 45.0],
            "AirTemp": [25.0, 26.0],
            "Humidity": [50.0, 55.0],
            "Rainfall": [False, False],
        }
    )


def _table():
    return build_lap_table(_race(), _weather(), season=2025, round_number=14, circuit="Budapest")


def _row(table, driver, lap):
    return table[(table["driver"] == driver) & (table["lap_number"] == lap)].iloc[0]


def test_columns_match_lap_table_schema():
    assert _table().columns.tolist() == [
        "season", "round", "circuit", "session_type", "driver", "team", "lap_number",
        "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "lap_end_time_s", "is_accurate",
        "compound", "tyre_age", "fresh_tyre", "stint", "compounds_used_so_far",
        "is_pit_in_lap", "is_pit_out_lap", "pit_lane_time_s", "track_status",
        "is_sc", "is_vsc", "is_yellow", "is_red_flag", "is_restart_lap", "is_clean_lap",
        "position", "gap_to_leader_s", "gap_ahead_s", "gap_behind_s", "car_ahead", "car_behind", "in_traffic",
        "laps_remaining", "race_progress", "fuel_proxy",
        "track_temp", "air_temp", "humidity", "rainfall",
        "driver_ref_pace_s", "lap_delta_s",
    ]


def test_identifiers_timing_and_progress():
    table = _table()
    row = _row(table, "AAA", 2)

    assert len(table) == 23
    assert (row["season"], row["round"], row["circuit"], row["session_type"]) == (2025, 14, "Budapest", "R")
    assert row["team"] == "Team AAA"
    assert row["lap_time_s"] == pytest.approx(90.0)
    assert row["sector1_s"] == pytest.approx(30.0)
    assert row["lap_end_time_s"] == pytest.approx(185.0)
    assert row["laps_remaining"] == 6
    assert row["race_progress"] == pytest.approx(0.25)
    assert row["fuel_proxy"] == pytest.approx(0.75)


def test_positions_and_gaps_come_from_crossing_times():
    table = _table()
    leader, second, third = (_row(table, d, 1) for d in ("AAA", "BBB", "CCC"))

    assert [leader["position"], second["position"], third["position"]] == [1, 2, 3]
    assert np.isnan(leader["gap_ahead_s"]) and leader["gap_behind_s"] == pytest.approx(1.0)
    assert second["gap_to_leader_s"] == pytest.approx(1.0)
    assert (second["car_ahead"], second["car_behind"]) == ("AAA", "CCC")
    assert third["gap_ahead_s"] == pytest.approx(4.0)
    assert [leader["in_traffic"], second["in_traffic"], third["in_traffic"]] == [False, True, False]
    assert _row(table, "CCC", 7)["gap_to_leader_s"] == pytest.approx(104.0)


def test_missing_crossing_time_gets_no_position():
    table = _table()

    assert pd.isna(_row(table, "CCC", 3)["position"])
    assert pd.isna(_row(table, "BBB", 3)["car_behind"])
    assert _row(table, "BBB", 3)["position"] == 2


def test_track_status_flags_and_restart_laps():
    table = _table()
    vsc_lap, after_vsc = _row(table, "AAA", 3), _row(table, "AAA", 4)
    sc_lap, restart, after = (_row(table, "AAA", lap) for lap in (6, 7, 8))

    assert vsc_lap["is_vsc"] and vsc_lap["is_yellow"] and not vsc_lap["is_sc"]
    assert not after_vsc["is_restart_lap"]
    assert sc_lap["is_sc"] and not sc_lap["is_red_flag"]
    assert restart["is_restart_lap"] and not after["is_restart_lap"]
    assert table["is_restart_lap"].sum() == 3


def test_pit_stop_and_tyre_columns():
    table = _table()
    in_lap, out_lap = _row(table, "BBB", 4), _row(table, "BBB", 5)

    assert in_lap["is_pit_in_lap"] and not in_lap["is_pit_out_lap"]
    assert out_lap["is_pit_out_lap"]
    assert in_lap["pit_lane_time_s"] == pytest.approx(22.0)
    assert table["pit_lane_time_s"].notna().sum() == 1
    assert (in_lap["compounds_used_so_far"], out_lap["compounds_used_so_far"]) == (1, 2)
    assert (out_lap["stint"], out_lap["tyre_age"], out_lap["compound"]) == (2, 1, "HARD")


def test_unknown_compound_is_missing():
    row = _row(_table(), "CCC", 2)

    assert pd.isna(row["compound"])
    assert row["compounds_used_so_far"] == 1


def test_weather_is_last_sample_before_lap_end():
    table = _table()

    assert _row(table, "AAA", 4)["track_temp"] == 40.0
    assert _row(table, "AAA", 5)["track_temp"] == 45.0
    assert not _row(table, "AAA", 5)["rainfall"]
    assert np.isnan(_row(table, "CCC", 3)["track_temp"])


def test_clean_laps_and_reference_pace():
    table = _table()
    clean = table[table["is_clean_lap"]].groupby("driver")["lap_number"].apply(list).to_dict()

    assert clean == {"AAA": [2, 4, 8], "BBB": [2, 8], "CCC": [4]}
    assert _row(table, "AAA", 5)["driver_ref_pace_s"] == pytest.approx(90.0)
    assert _row(table, "AAA", 5)["lap_delta_s"] == pytest.approx(9.0)
    assert _row(table, "BBB", 1)["driver_ref_pace_s"] == pytest.approx(89.5)


def test_duplicate_driver_lap_raises():
    laps = _race()

    with pytest.raises(ValueError, match="duplicate"):
        build_lap_table(pd.concat([laps, laps.iloc[[0]]]), _weather(), season=2025, round_number=14, circuit="Budapest")


def test_missing_deleted_flags_raise():
    with pytest.raises(ValueError, match="Deleted"):
        build_lap_table(_race().assign(Deleted=None), _weather(), season=2025, round_number=14, circuit="Budapest")


def test_input_order_does_not_change_output():
    shuffled = _race().sample(frac=1, random_state=0)

    table = build_lap_table(shuffled, _weather(), season=2025, round_number=14, circuit="Budapest")

    pd.testing.assert_frame_equal(table, _table())


def test_lap_after_missing_lap_is_not_a_restart():
    laps = _race()
    laps = laps[~((laps["Driver"] == "AAA") & (laps["LapNumber"] == 7))]

    table = build_lap_table(laps, _weather(), season=2025, round_number=14, circuit="Budapest")

    assert not _row(table, "AAA", 8)["is_restart_lap"]


def test_rainfall_stays_boolean_with_untimed_laps():
    assert _table()["rainfall"].dtype == "boolean"

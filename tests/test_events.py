import numpy as np
import pandas as pd
import pytest

from f1strat.data.events import EVENT_COLUMNS, build_events

GREEN_LAPS = [(100.0, True)] * 10
RED_LAPS = [(100.0, True)] * 3 + [(2000.0, False), (200.0, False)] + [(100.0, True)] * 5


def _laps(laps):
    durations = np.array([duration for duration, _ in laps])
    ends = durations.cumsum()
    leader = pd.DataFrame(
        {
            "Driver": "AAA",
            "LapNumber": np.arange(1, len(laps) + 1, dtype=float),
            "Time": pd.to_timedelta(ends, unit="s"),
            "LapTime": pd.to_timedelta([d if timed else np.nan for d, timed in laps], unit="s"),
        }
    )
    trailer = leader.assign(Driver="BBB", Time=leader["Time"] + pd.Timedelta(seconds=2))
    return pd.concat([trailer, leader], ignore_index=True)


def _status(rows):
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([time for time, _ in rows], unit="s"),
            "Status": [status for _, status in rows],
            "Message": "",
        }
    )


def _messages(rows=()):
    return pd.DataFrame({"Lap": [lap for lap, _ in rows], "Message": [text for _, text in rows]})


def _events(laps, status, messages=()):
    return build_events(_status(status), _messages(messages), _laps(laps), season=2026, round_number=1)


def _row(events, event_type):
    rows = events[events["event_type"] == event_type]
    assert len(rows) == 1
    return rows.iloc[0]


def test_safety_car_period():
    events = _events(GREEN_LAPS, [(0, "1"), (250, "4"), (480, "1")])

    assert list(events.columns) == EVENT_COLUMNS
    sc = _row(events, "SC")
    assert (sc["start_lap"], sc["end_lap"]) == (3, 5)
    assert sc["duration_s"] == pytest.approx(230)
    assert sc["restart_type"] == "rolling"


def test_vsc_ending_does_not_close_the_period():
    events = _events(GREEN_LAPS, [(0, "1"), (150, "6"), (260, "7"), (270, "1")])

    vsc = _row(events, "VSC")
    assert (vsc["start_lap"], vsc["end_lap"]) == (2, 3)
    assert vsc["duration_s"] == pytest.approx(120)
    assert vsc["restart_type"] is None


@pytest.mark.parametrize("all_clear_s", [420, 2400])
def test_red_flag_runs_through_untimed_laps(all_clear_s):
    status = [(0, "1"), (210, "4"), (250, "5"), (all_clear_s, "1"), (2600, "2"), (2610, "1")]
    messages = [(3, "RED FLAG - RACE SUSPENDED"), (4, "STANDING START"), (5, "EXTRA FORMATION LAP"), (5, "STANDING START")]

    events = _events(RED_LAPS, status, messages)

    assert list(events["event_type"]) == ["SC", "RED"]
    sc = _row(events, "SC")
    assert (sc["start_lap"], sc["end_lap"], sc["end_time_s"]) == (3, 3, 250)
    assert sc["restart_type"] is None
    red = _row(events, "RED")
    assert (red["start_lap"], red["end_lap"]) == (3, 5)
    assert (red["start_time_s"], red["end_time_s"]) == (250, 2500)
    assert red["restart_type"] == "standing"


def test_rolling_restart_behind_safety_car_starts_an_sc_period():
    status = [(0, "1"), (250, "5"), (400, "1"), (2350, "4"), (2560, "1")]
    messages = [(3, "RED FLAG"), (5, "RACE WILL RESUME AT 17:14 - ROLLING START PROCEDURE")]

    events = _events(RED_LAPS, status, messages)

    assert list(events["event_type"]) == ["RED", "SC"]
    assert _row(events, "RED")["restart_type"] == "rolling"
    sc = _row(events, "SC")
    assert (sc["start_lap"], sc["end_lap"]) == (6, 6)
    assert (sc["start_time_s"], sc["end_time_s"]) == (2500, 2560)


def test_restart_type_comes_from_a_message_after_the_suspension():
    status = [(0, "1"), (250, "5")]
    messages = [(3, "RED FLAG - RACE SUSPENDED"), (7, "STANDING START"), (9, "ROLLING START")]

    events = _events(RED_LAPS, status, messages)

    red = _row(events, "RED")
    assert red["end_lap"] == 5
    assert red["restart_type"] == "standing"


def test_period_open_at_the_flag_ends_on_the_last_lap():
    events = _events(GREEN_LAPS, [(0, "1"), (850, "4")])

    sc = _row(events, "SC")
    assert (sc["start_lap"], sc["end_lap"], sc["end_time_s"]) == (9, 10, 1000)
    assert sc["restart_type"] is None


def test_red_flag_after_the_last_lap_is_dropped():
    events = _events(GREEN_LAPS, [(0, "1"), (1050, "5")])

    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS
    assert events["start_lap"].dtype == "int64"
    assert events["duration_s"].dtype == "float64"


def test_start_behind_safety_car_is_rolling():
    messages = [(3, "RED FLAG"), (5, "RACE WILL START BEHIND THE SAFETY CAR")]

    events = _events(RED_LAPS, [(0, "1"), (250, "5")], messages)

    assert _row(events, "RED")["restart_type"] == "rolling"


def test_red_flag_ends_on_the_last_leader_lap_with_a_time_of_day():
    laps = _laps(RED_LAPS)
    laps.loc[laps["LapNumber"] == 5, "Time"] = pd.NaT

    events = build_events(_status([(0, "1"), (250, "5")]), _messages([(5, "STANDING START")]), laps, 2026, 1)

    red = _row(events, "RED")
    assert (red["end_lap"], red["end_time_s"]) == (4, 2300)


def test_restart_message_of_a_later_red_flag_is_not_used():
    laps = RED_LAPS[:8] + [(300.0, False)] + [(100.0, True)] * 2
    messages = [(3, "RED FLAG"), (8, "RED FLAG"), (9, "STANDING START")]

    with pytest.raises(ValueError, match="restart"):
        _events(laps, [(0, "1"), (250, "5"), (2750, "5")], messages)


def test_empty_track_status_raises():
    with pytest.raises(ValueError, match="empty"):
        build_events(_status([]), _messages(), _laps(GREEN_LAPS), 2026, 1)


def test_track_status_without_time_raises():
    status = _status([(0, "1"), (250, "4")])
    status.loc[1, "Time"] = pd.NaT

    with pytest.raises(ValueError, match="time"):
        build_events(status, _messages(), _laps(GREEN_LAPS), 2026, 1)


def test_race_ending_under_red_flag_has_no_restart():
    events = _events(GREEN_LAPS[:5], [(0, "1"), (450, "5")], [(5, "RED FLAG")])

    red = _row(events, "RED")
    assert (red["start_lap"], red["end_lap"], red["end_time_s"]) == (5, 5, 500)
    assert red["restart_type"] is None


def test_red_flag_without_restart_message_raises():
    with pytest.raises(ValueError, match="restart"):
        _events(RED_LAPS, [(0, "1"), (250, "5")], [(3, "RED FLAG")])

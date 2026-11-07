import fastf1
import pandas as pd

EVENT_COLUMNS = [
    "season", "round", "event_type", "start_lap", "end_lap",
    "start_time_s", "end_time_s", "duration_s", "restart_type",
]
EVENT_DTYPES = {
    "season": "int64", "round": "int64", "start_lap": "int64", "end_lap": "int64",
    "start_time_s": "float64", "end_time_s": "float64", "duration_s": "float64",
}
PERIOD_TYPES = {"4": "SC", "6": "VSC"}
RESTART_TYPES = {"SC": "rolling", "VSC": None}
GREEN = "1"
RED = "5"
CLOSING = {GREEN, RED}
RESTART_MESSAGE = r"STANDING START|ROLLING START|START BEHIND THE SAFETY CAR"


def _leader_laps(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"])
    leader = timed.loc[timed.groupby("LapNumber")["Time"].idxmin()].sort_values("LapNumber")
    return pd.DataFrame(
        {
            "lap": leader["LapNumber"].astype(int).to_numpy(),
            "end_s": leader["Time"].dt.total_seconds().to_numpy(),
            "timed": leader["LapTime"].notna().to_numpy(),
        }
    )


def _lap_at(leader: pd.DataFrame, time_s: float) -> int | None:
    later = leader.loc[leader["end_s"] > time_s, "lap"]
    return int(later.iloc[0]) if len(later) else None


def _event(kind: str, start_s: float, end_s: float, start_lap: int, end_lap: int, restart: str | None) -> dict:
    return {
        "event_type": kind,
        "start_lap": start_lap,
        "end_lap": end_lap,
        "start_time_s": start_s,
        "end_time_s": end_s,
        "duration_s": end_s - start_s,
        "restart_type": restart,
    }


def _restart_type(messages: pd.DataFrame, start_lap: int, before_lap: int | None) -> str | None:
    window = messages["Lap"] >= start_lap
    if before_lap is not None:
        window &= messages["Lap"] < before_lap
    restarts = messages.loc[window & messages["Message"].str.contains(RESTART_MESSAGE, na=False), "Message"]
    if restarts.empty:
        return None
    return "standing" if "STANDING" in restarts.iloc[0] else "rolling"


def _suspensions(timeline: pd.DataFrame, leader: pd.DataFrame) -> list[dict]:
    reds: list[dict] = []
    for time_s in timeline.loc[timeline["code"] == RED, "time_s"]:
        if reds and time_s < reds[-1]["end_time_s"]:
            continue
        start_lap = _lap_at(leader, time_s)
        if start_lap is None:
            continue
        resumed = leader[(leader["lap"] > start_lap) & leader["timed"]]
        suspended = leader[leader["lap"] < resumed["lap"].iloc[0]] if len(resumed) else leader
        end = suspended.iloc[-1]
        reds.append(_event("RED", time_s, float(end["end_s"]), start_lap, int(end["lap"]), None))
    return reds


def _red_flags(timeline: pd.DataFrame, messages: pd.DataFrame, leader: pd.DataFrame) -> list[dict]:
    reds = _suspensions(timeline, leader)
    last_lap = int(leader["lap"].iloc[-1])
    next_starts = [red["start_lap"] for red in reds[1:]] + [None]
    flagged = []
    for red, next_start in zip(reds, next_starts):
        restart = _restart_type(messages, red["start_lap"], next_start)
        if restart is None and red["end_lap"] < last_lap:
            raise ValueError(f"no restart message for the red flag on lap {red['start_lap']}")
        flagged.append({**red, "restart_type": restart})
    return flagged


def _without_suspensions(timeline: pd.DataFrame, reds: list[dict]) -> pd.DataFrame:
    for red in reds:
        inside = timeline["time_s"].between(red["start_time_s"], red["end_time_s"], inclusive="left")
        last = timeline.loc[inside & (timeline["code"] != RED), "code"]
        resume = last.iloc[-1] if len(last) else GREEN
        bounds = pd.DataFrame({"time_s": [red["start_time_s"], red["end_time_s"]], "code": [RED, resume]})
        timeline = pd.concat([timeline[~inside], bounds], ignore_index=True)
    return timeline.sort_values("time_s", kind="stable")


def _spans(timeline: pd.DataFrame, last_end_s: float) -> list[tuple[str, float, float, str | None]]:
    spans: list[tuple[str, float, float, str | None]] = []
    open_period: tuple[str, float] | None = None
    for time_s, code in timeline[["time_s", "code"]].itertuples(index=False):
        kind = PERIOD_TYPES.get(code)
        if open_period is not None and (code in CLOSING or (kind and kind != open_period[0])):
            spans.append((*open_period, time_s, code))
            open_period = None
        if kind and open_period is None:
            open_period = (kind, time_s)
    if open_period is not None:
        spans.append((*open_period, last_end_s, None))
    return spans


def _neutralised_periods(timeline: pd.DataFrame, leader: pd.DataFrame) -> list[dict]:
    last_end_s = float(leader["end_s"].iloc[-1])
    periods = []
    for kind, start_s, end_s, closed_by in _spans(timeline, last_end_s):
        start_lap = _lap_at(leader, start_s)
        if start_lap is None:
            continue
        end_lap = _lap_at(leader, end_s)
        if end_lap is None:
            end_lap = int(leader["lap"].iloc[-1])
        restarted = closed_by != RED and end_s < last_end_s
        restart = RESTART_TYPES[kind] if restarted else None
        periods.append(_event(kind, start_s, min(end_s, last_end_s), start_lap, end_lap, restart))
    return periods


def build_events(
    track_status: pd.DataFrame,
    messages: pd.DataFrame,
    laps: pd.DataFrame,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    if track_status.empty or laps.empty:
        raise ValueError("track_status or laps is empty; load the session with laps=True")
    if track_status["Time"].isna().any():
        raise ValueError("track_status has rows without a time")
    leader = _leader_laps(laps)
    timeline = pd.DataFrame(
        {"time_s": track_status["Time"].dt.total_seconds(), "code": track_status["Status"].astype(str)}
    ).sort_values("time_s", kind="stable")
    reds = _red_flags(timeline, messages, leader)
    periods = _neutralised_periods(_without_suspensions(timeline, reds), leader)
    events = pd.DataFrame(periods + reds, columns=EVENT_COLUMNS[2:])
    events = events.sort_values("start_time_s", kind="stable").reset_index(drop=True)
    return events.assign(season=season, round=round_number)[EVENT_COLUMNS].astype(EVENT_DTYPES)


def race_events(year: int, event_name: str) -> pd.DataFrame:
    event = fastf1.get_event(year, event_name)
    session = event.get_session("Race")
    session.load(laps=True, telemetry=False, weather=False, messages=True)
    return build_events(
        session.track_status,
        session.race_control_messages,
        session.laps,
        season=year,
        round_number=int(event["RoundNumber"]),
    )

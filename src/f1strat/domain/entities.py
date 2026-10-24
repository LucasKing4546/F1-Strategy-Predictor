from dataclasses import astuple, dataclass
from enum import Enum


class Mark(Enum):
    OVERALL = "overall"
    PERSONAL = "personal"
    NONE = "none"


@dataclass(frozen=True)
class LapTiming:
    lap_s: float | None
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None


def _mark(value: float | None, best: float | None, fastest: float | None) -> Mark:
    if value is None:
        return Mark.NONE
    if value == fastest:
        return Mark.OVERALL
    if value == best:
        return Mark.PERSONAL
    return Mark.NONE


@dataclass(frozen=True)
class CarState:
    driver: str
    team: str
    position: int
    gap_to_leader_s: float
    gap_ahead_s: float | None
    gap_behind_s: float | None
    car_ahead: str | None
    car_behind: str | None
    compound: str | None
    tyre_age: int | None
    stint: int | None
    compounds_used: frozenset[str]
    pit_in: bool
    pit_out: bool
    last: LapTiming
    last_deleted: bool
    best: LapTiming


@dataclass(frozen=True)
class RaceState:
    season: int
    round: int
    circuit: str
    lap: int
    laps_remaining: int
    is_sc: bool
    is_vsc: bool
    is_red_flag: bool
    fastest: LapTiming
    cars: tuple[CarState, ...]

    @property
    def is_neutralised(self) -> bool:
        return self.is_sc or self.is_vsc or self.is_red_flag

    def car(self, driver: str) -> CarState:
        for car in self.cars:
            if car.driver == driver:
                return car
        raise KeyError(f"{driver} has no state on lap {self.lap}")

    def marks(self, driver: str) -> tuple[Mark, Mark, Mark, Mark]:
        car = self.car(driver)
        if car.last_deleted:
            return Mark.NONE, Mark.NONE, Mark.NONE, Mark.NONE
        lap, s1, s2, s3 = (
            _mark(*times) for times in zip(astuple(car.last), astuple(car.best), astuple(self.fastest))
        )
        return lap, s1, s2, s3

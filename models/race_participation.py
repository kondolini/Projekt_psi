from datetime import datetime
from typing import Optional
import pickle


class RaceParticipation:
    def __init__(
        self,
        dog_id: str,
        race_id: str,
        race_datetime: datetime,
        trap_number: Optional[int] = None,
        sp: Optional[str] = None,
        position: Optional[str] = None,
        btn_distance: Optional[str] = None,
        split_time: Optional[float] = None,
        comment: Optional[str] = None,
        run_time: Optional[float] = None,
        adjusted_time: Optional[float] = None,
        weight: Optional[float] = None,
        winner_id: Optional[str] = None,
        track_name: Optional[str] = None,
        race_class: Optional[str] = None,
        distance: Optional[float] = None,
        going: Optional[str] = None,
        win_time: Optional[float] = None,
        meeting_id: Optional[str] = None,
    ):
        self.dog_id = dog_id
        self.race_id = race_id
        self.race_datetime = race_datetime

        self.trap_number = trap_number
        self.sp = sp
        self.position = position
        self.btn_distance = btn_distance
        self.split_time = split_time
        self.comment = comment
        self.run_time = run_time
        self.adjusted_time = adjusted_time
        self.weight = weight
        self.winner_id = winner_id

        self.track_name = track_name
        self.race_class = race_class
        self.distance = distance
        self.going = going
        self.win_time = win_time
        self.meeting_id = meeting_id

    def __lt__(self, other):
        return self.race_datetime < other.race_datetime

    def __repr__(self):
        return f"RaceParticipation({self.dog_id} @ {self.race_datetime.strftime('%Y-%m-%d %H:%M')})"
    
    def print_info(self):
        print({
            "dog_id": self.dog_id,
            "race_id": self.race_id,
            "race_datetime": self.race_datetime,
            "trap_number": self.trap_number,
            "sp": self.sp,
            "position": self.position,
            "btn_distance": self.btn_distance,
            "split_time": self.split_time,
            "comment": self.comment,
            "run_time": self.run_time,
            "adjusted_time": self.adjusted_time,
            "weight": self.weight,
            "winner_id": self.winner_id,
            "track_name": self.track_name,
            "race_class": self.race_class,
            "distance": self.distance,
            "going": self.going,
            "win_time": self.win_time,
            "meeting_id": self.meeting_id,  # <-- Added field
        })

    def to_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(path: str) -> "RaceParticipation":
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_track_metadata(self) -> dict:
        return {
            "name": self.track_name,
            "distance": self.distance,
            "race_class": self.race_class,
            "going": self.going,
        }


def parse_race_participation(row: dict) -> Optional[RaceParticipation]:
    try:
        race_datetime = datetime.strptime(
            f"{row['raceDate']} {row['raceTime']}", "%d/%m/%Y %H:%M:%S"
        )
    except Exception:
        return None  # Skip rows with bad date/time

    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def safe_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    return RaceParticipation(
        dog_id=row["dogId"],
        race_id=row["raceId"],
        race_datetime=race_datetime,
        trap_number=safe_int(row.get("trapNumber")),
        sp=row.get("SP"),
        position=row.get("resultPosition"),
        btn_distance=row.get("resultBtnDistance"),
        split_time=safe_float(row.get("resultSectionalTime")),
        comment=row.get("resultComment"),
        run_time=safe_float(row.get("resultRunTime")),
        adjusted_time=safe_float(row.get("resultAdjustedTime")),
        weight=safe_float(row.get("resultDogWeight")),
        winner_id=row.get("winnerOr2ndId"),
        track_name=row.get("trackName"),
        race_class=row.get("raceClass"),
        distance=safe_float(row.get("raceDistance")),
        going=row.get("raceGoing"),
        win_time=safe_float(row.get("raceWinTime")),
        meeting_id=row.get("meetingId"),  # <-- Added extraction
    )

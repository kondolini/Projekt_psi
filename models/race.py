import os
import pickle
from datetime import date, time
from typing import List, Optional, Dict

from models.track import Track
from models.race_participation import RaceParticipation
from models.dog import Dog
import re

def convert_sp_to_decimal(sp: Optional[str]) -> Optional[float]:
    """
    Convert a British-style starting price (SP) string to decimal odds.
    Examples:
        "5/2f" -> 3.5
        "7/4j" -> 2.75
        "evs" or "evens" -> 2.0
        "3/1" -> 4.0
    """
    if not sp:
        return None

    sp = sp.strip().lower()

    # Handle evens
    if sp in {"evs", "evens"}:
        return 2.0

    # Remove trailing annotations like "f", "j", "p", etc.
    sp_cleaned = re.match(r"(\d+)\s*/\s*(\d+)", sp)
    if sp_cleaned:
        num = int(sp_cleaned.group(1))
        denom = int(sp_cleaned.group(2))
        return round(1 + (num / denom), 2)

    # Try to parse straight decimal or integer
    try:
        return float(re.match(r"^\d+(\.\d+)?", sp).group(0))
    except Exception:
        return None

class Race:
    def __init__(
        self,
        race_id: str,
        race_date: date,
        race_time: time,
        distance: float,
        race_class: str,
        track_name: str,
        dog_ids: List[str],
        odds_vec: Optional[List[float]] = None,
        race_time_vec: Optional[List[float]] = None,
        commentary_tags_vec: Optional[List[List[str]]] = None,
        category: Optional[str] = None,
        rainfall_7d: Optional[List[float]] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
    ):
        self.race_id = race_id
        self.race_date = race_date
        self.race_time = race_time
        self.distance = distance
        self.race_class = race_class
        self.track_name = track_name
        self.dog_ids = dog_ids
        self.odds_vec = odds_vec or []
        self.race_time_vec = race_time_vec or []
        self.commentary_tags_vec = commentary_tags_vec or []
        self.category = category

        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    @classmethod
    def from_participations(cls, participations: List[RaceParticipation]) -> "Race":
        assert participations, "No participations provided"
        assert all(p.race_id == participations[0].race_id for p in participations), "Mismatched race_ids"

        participations.sort(key=lambda p: p.trap_number or 0)
        p0 = participations[0]

        odds_vec = [convert_sp_to_decimal(p.sp) for p in participations]
        odds_vec = None if all(o is None for o in odds_vec) else odds_vec

        return cls(
            race_id=p0.race_id,
            race_date=p0.race_datetime.date(),
            race_time=p0.race_datetime.time(),
            distance=p0.distance,
            race_class=p0.race_class,
            track_name=p0.track_name,
            dog_ids=[p.dog_id for p in participations],
            odds_vec=odds_vec,
            race_time_vec=[p.run_time for p in participations],
            commentary_tags_vec=[[] for _ in participations],  # placeholder
            category=getattr(p0, "category", None),
        )

    @classmethod
    def from_dogs(cls, dogs: List[Dog], race_id: str) -> Optional["Race"]:
        participations = [dog.get_participation_by_race_id(race_id) for dog in dogs]
        participations = [p for p in participations if p]
        if not participations:
            return None
        return cls.from_participations(participations)

    def set_weather(self, rainfall_7d: List[float], temperature: float, humidity: float):
        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    def get_track(self, track_lookup: Dict[str, Track]) -> Optional[Track]:
        return track_lookup.get(self.track_name)

    def get_dogs(self, dog_lookup: Dict[str, Dog]) -> List[Dog]:
        return [dog_lookup[dog_id] for dog_id in self.dog_ids if dog_id in dog_lookup]

    def get_dog_participation(self, dog_id: str, dog_lookup: Dict[str, Dog]) -> Optional[RaceParticipation]:
        dog = dog_lookup.get(dog_id)
        if dog:
            return dog.get_participation_by_race_id(self.race_id)
        return None

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Race":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __lt__(self, other):
        return (self.race_date, self.race_time) < (other.race_date, other.race_time)

    def __repr__(self):
        return f"<Race {self.race_id} @ {self.track_name} on {self.race_date} - {len(self.dog_ids)} dogs>"

    def print_info(self):
        print(f"Race ID: {self.race_id}")
        print(f"Date: {self.race_date}, Time: {self.race_time}")
        print(f"Track: {self.track_name}, Distance: {self.distance}m")
        print(f"Class: {self.race_class}, Category: {self.category}")
        print(f"Dogs: {self.dog_ids}")
        print(f"Odds: {self.odds_vec}")
        print(f"Race Times: {self.race_time_vec}")
        if self.rainfall_7d:
            print(f"Rainfall (7d): {self.rainfall_7d}")
        if self.temperature is not None:
            print(f"Temperature: {self.temperature}")
        if self.humidity is not None:
            print(f"Humidity: {self.humidity}")

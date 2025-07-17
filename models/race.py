import os
import pickle
from datetime import datetime
from typing import List, Optional

from models.track import Track
from models.race_participation import RaceParticipation
from models.dog import Dog


class Race:
    def __init__(
        self,
        race_id: str,
        race_date: datetime,
        race_time: datetime,
        distance: int,
        race_class: str,
        category: str,
        track_name: str,
        odds_vec: List[float],
        race_time_vec: List[float],
        commentary_tags_vec: List[List[str]],
        dog_ids: List[str],
        rainfall_7d: Optional[List[float]] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None
    ):
        self.race_id = race_id
        self.race_date = race_date
        self.race_time = race_time
        self.distance = distance
        self.race_class = race_class
        self.category = category
        self.track_name = track_name
        self.odds_vec = odds_vec
        self.race_time_vec = race_time_vec
        self.commentary_tags_vec = commentary_tags_vec
        self.dog_ids = dog_ids

        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    @classmethod
    def from_participations(cls, participations: List[RaceParticipation]) -> "Race":
        assert all(p.race_id == participations[0].race_id for p in participations), "Mismatched race_ids"
        participations = sorted(participations, key=lambda p: p.trap_number)

        p0 = participations[0]
        return cls(
            race_id=p0.race_id,
            race_date=p0.race_date,
            race_time=p0.race_time,
            distance=p0.distance,
            race_class=p0.race_class,
            category=p0.category,
            track_name=p0.track_name,
            odds_vec=[p.odds for p in participations],
            race_time_vec=[p.finish_time for p in participations],
            commentary_tags_vec=[p.commentary_tags for p in participations],
            dog_ids=[p.dog_id for p in participations],
        )

    def set_weather(self, rainfall_7d: List[float], temperature: float, humidity: float):
        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    def get_dog_participation(self, dog_id: str, dog_lookup: dict) -> Optional[RaceParticipation]:
        dog = dog_lookup.get(dog_id)
        if dog:
            for p in dog.participations:
                if p.race_id == self.race_id:
                    return p
        return None

    def get_dogs(self, dog_lookup: dict) -> List[Dog]:
        return [dog_lookup[dog_id] for dog_id in self.dog_ids if dog_id in dog_lookup]

    def get_track(self, track_lookup: dict) -> Optional[Track]:
        return track_lookup.get(self.track_name)

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
        print(f"Finish Times: {self.race_time_vec}")
        print(f"Tags: {self.commentary_tags_vec}")
        if self.rainfall_7d:
            print(f"Rainfall (7d): {self.rainfall_7d}")
        if self.temperature is not None:
            print(f"Temperature: {self.temperature}")
        if self.humidity is not None:
            print(f"Humidity: {self.humidity}")

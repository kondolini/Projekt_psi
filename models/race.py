import os
import pickle
import re
from datetime import date, time, datetime
from typing import Dict, Optional, List
from models.track import Track
from models.race_participation import RaceParticipation
from models.dog import Dog
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np


def convert_sp_to_decimal(sp: Optional[str]) -> Optional[float]:
    if not sp or str(sp).strip().lower() in {"", "nan", "none", "-", "n/a", " "}:
        return None
    sp = str(sp).strip().lower()
    if sp in {"evs", "evens"}:
        return 2.0
    match = re.match(r"(\d+)\s*/\s*(\d+)", sp)
    if match:
        num, denom = int(match.group(1)), int(match.group(2))
        return round(1 + (num / denom), 2)
    try:
        val = float(re.match(r"^\d+(\.\d+)?", sp).group(0))
        return val if val > 1.0 else None
    except:
        return None


def power_method_devig(decimal_odds: Dict[int, Optional[float]], tol: float = 1e-6, max_iter: int = 100):
    implied_probs = {t: (1 / o) for t, o in decimal_odds.items() if o and o > 1.0}
    if not implied_probs:
        return {}, {}

    def booksum(k):
        return sum(p ** k for p in implied_probs.values())

    lo, hi = 0.01, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        s = booksum(mid)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid

    k_final = mid
    denominator = sum(p ** k_final for p in implied_probs.values())
    fair_probs = {t: round((p ** k_final) / denominator, 6) for t, p in implied_probs.items()}
    fair_odds = {t: round(1 / p, 4) for t, p in fair_probs.items()}
    return fair_odds, fair_probs


class Race:
    def __init__(
        self,
        race_id: str,
        meeting_id: str,
        race_date: date,
        race_time: time,
        distance: float,
        race_class: str,
        track_name: str,
        dog_ids: Dict[int, str],
        odds: Optional[Dict[int, Optional[float]]] = None,
        race_times: Optional[Dict[int, Optional[float]]] = None,
        commentary_tags: Optional[Dict[int, list]] = None,
        weights: Optional[Dict[int, Optional[float]]] = None,
        category: Optional[str] = None,
        rainfall_7d: Optional[list] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
    ):
        self.race_id = race_id
        self.meeting_id = meeting_id
        self.race_date = race_date
        self.race_time = race_time
        self.distance = distance
        self.race_class = race_class
        self.track_name = track_name
        self.dog_ids = dog_ids
        self.odds = odds or {}
        self.implied_probs = {t: (1 / o if o and o > 1.0 else None) for t, o in self.odds.items()} if self.odds else {}
        self.devig_odds, self.fair_probs = power_method_devig(self.odds) if self.odds else ({}, {})
        self.race_times = race_times or {}
        self.commentary_tags = commentary_tags or {}
        self.weights = weights or {}
        self.category = category
        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    @classmethod
    def from_participations(cls, participations: list["RaceParticipation"]) -> "Race":
        assert participations, "No participations provided"
        trap_map = {p.trap_number: p for p in participations if p.trap_number is not None}
        p0 = next(iter(trap_map.values()))
        dog_ids = {trap: p.dog_id for trap, p in trap_map.items()}
        odds = {trap: convert_sp_to_decimal(p.sp) for trap, p in trap_map.items()}
        if all(o is None for o in odds.values()):
            odds = None
        race_times = {trap: p.run_time for trap, p in trap_map.items()}
        
        # Extract commentary tags from race participation comments
        commentary_tags = {}
        for trap, p in trap_map.items():
            if p.comment:
                # Convert comment to string if it's not already
                comment_str = str(p.comment) if p.comment is not None else ""
                if comment_str and comment_str.lower() not in ['nan', 'none', '']:
                    # Parse comment into tags (split by comma and clean)
                    tags = [tag.strip() for tag in comment_str.split(',') if tag.strip()]
                    commentary_tags[trap] = tags
                else:
                    commentary_tags[trap] = []
            else:
                commentary_tags[trap] = []
        
        weights = {trap: p.weight for trap, p in trap_map.items()}

        return cls(
            race_id=p0.race_id,
            meeting_id=p0.meeting_id,
            race_date=p0.race_datetime.date(),
            race_time=p0.race_datetime.time(),
            distance=p0.distance,
            race_class=p0.race_class,
            track_name=p0.track_name,
            dog_ids=dog_ids,
            odds=odds,
            race_times=race_times,
            commentary_tags=commentary_tags,
            weights=weights,
            category=getattr(p0, "category", None),
        )

    @classmethod
    def from_dogs(cls, trap_to_dog: Dict[int, "Dog"], race_id: str, meeting_id: str) -> Optional["Race"]:
        participations = [dog.get_participation_by_race(race_id, meeting_id) for dog in trap_to_dog.values()]
        participations = [p for p in participations if p]
        return cls.from_participations(participations) if participations else None

    def is_trial_race(self) -> bool:
        return not self.odds or all(v is None for v in self.odds.values())

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def set_weather(self, rainfall_7d, temperature, humidity):
        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity

    @staticmethod
    def load(path: str) -> "Race":
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_track(self, track_lookup: Dict[str, Track]) -> Optional[Track]:
        return track_lookup.get(self.track_name)

    def get_dogs(self, dog_lookup: Dict[str, Dog]) -> Dict[int, Dog]:
        return {trap: dog_lookup[dog_id] for trap, dog_id in self.dog_ids.items() if dog_id in dog_lookup}

    def get_dog_participation(self, trap: int, dog_lookup: Dict[str, Dog]) -> Optional["RaceParticipation"]:
        dog_id = self.dog_ids.get(trap)
        if dog_id and dog_id in dog_lookup:
            return dog_lookup[dog_id].get_participation_by_race(self.race_id, self.meeting_id)
        return None
    
    @staticmethod
    def get_races_chronologically(race_lookup: Dict[str, "Race"]) -> List["Race"]:
        """Get all races sorted chronologically for training"""
        return sorted(race_lookup.values(), key=lambda r: (r.race_date, r.race_time))

    @staticmethod
    def get_races_before_date(race_lookup: Dict[str, "Race"], cutoff_date: date) -> List["Race"]:
        """Get all races before a specific date for training/validation split"""
        return [race for race in race_lookup.values() 
                if race.race_date < cutoff_date]
    
    @staticmethod
    def get_races_after_date(race_lookup: Dict[str, "Race"], cutoff_date: date) -> List["Race"]:
        """Get all races after a specific date for training/validation split"""
        return [race for race in race_lookup.values()
                if race.race_date >= cutoff_date]

    def get_race_datetime(self) -> datetime:
        """Get combined datetime for easier sorting"""
        return datetime.combine(self.race_date, self.race_time)

    def __lt__(self, other):
        return (self.race_date, self.race_time) < (other.race_date, other.race_time)

    def __repr__(self):
        return f"<Race {self.race_id} @ {self.track_name} on {self.race_date} (meeting {self.meeting_id}) - {len(self.dog_ids)} dogs>"

    def print_info(self):
        print(f"Race ID: {self.race_id}")
        print(f"Meeting ID: {self.meeting_id}")
        print(f"Date: {self.race_date}, Time: {self.race_time}")
        print(f"Track: {self.track_name}, Distance: {self.distance}m")
        print(f"Class: {self.race_class}, Category: {self.category}")
        print(f"Dogs: {self.dog_ids}")
        print(f"Odds (Decimal): {self.odds}")
        print(f"Implied Probs: {self.implied_probs}")
        print(f"De-vig Odds (Power Method): {self.devig_odds}")
        print(f"Fair Probs (Power Method): {self.fair_probs}")
        print(f"Race Times: {self.race_times}")
        print(f"Weights: {self.weights}")
        if self.rainfall_7d:
            print(f"Rainfall (7d): {self.rainfall_7d}")
        if self.temperature is not None:
            print(f"Temperature: {self.temperature}")
        if self.humidity is not None:
            print(f"Humidity: {self.humidity}")
        print(f"Commentary Tags: {self.commentary_tags}")

    @staticmethod
    def load_race_by_key(race_key, race_index: dict):
        """Load a race by its key using the race index"""
        if race_key not in race_index:
            return None
        
        race_info = race_index[race_key]
        bucket_path = race_info['path']
        storage_key = race_info['key']
        
        try:
            with open(bucket_path, 'rb') as f:
                races_bucket = pickle.load(f)
            return races_bucket.get(storage_key)
        except Exception as e:
            print(f"Error loading race {race_key}: {e}")
            return None

    @staticmethod
    def load_all_races_from_buckets(race_output_dir: str) -> Dict[str, "Race"]:
        """Load all races from bucket files"""
        all_races = {}
        
        for bucket_file in os.listdir(race_output_dir):
            if bucket_file.startswith('races_bucket_') and bucket_file.endswith('.pkl'):
                bucket_path = os.path.join(race_output_dir, bucket_file)
                try:
                    with open(bucket_path, 'rb') as f:
                        races_bucket = pickle.load(f)
                    
                    for storage_key, race in races_bucket.items():
                        all_races[storage_key] = race
                        
                except Exception as e:
                    print(f"Error loading race bucket {bucket_file}: {e}")
        
        return all_races


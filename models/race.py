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
        race_date: datetime = None,
        race_time: datetime = None,
        distance: int = None,
        race_class: str = None,
        category: str = None,
        track_name: str = None,
        odds_vec: List[float] = None,
        race_time_vec: List[float] = None,
        commentary_tags_vec: List[List[str]] = None,
        dog_ids: List[str] = None,
        rainfall_7d: Optional[List[float]] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        track: Optional[Track] = None
    ):
        self.race_id = race_id
        self.race_date = race_date
        self.race_time = race_time
        self.distance = distance
        self.race_class = race_class
        self.category = category
        self.track_name = track_name
        self.odds_vec = odds_vec or []
        self.race_time_vec = race_time_vec or []
        self.commentary_tags_vec = commentary_tags_vec or []
        self.dog_ids = dog_ids or []

        self.rainfall_7d = rainfall_7d
        self.temperature = temperature
        self.humidity = humidity
        
        # New attributes for enhanced functionality
        self.track = track
        self.participations: List[RaceParticipation] = []
        self.race_datetime = race_date  # Alias for consistency
        self.race_distance = distance   # Alias for consistency
        self.going = None

    @classmethod
    def from_participations(cls, race_id: str, participations: List[RaceParticipation], track: Optional[Track] = None) -> "Race":
        """Create a Race from a list of participations - enhanced version"""
        if not participations:
            raise ValueError("No participations provided to construct Race")
        
        # Sort participations by trap number
        participations = sorted(participations, key=lambda p: p.trap_number if p.trap_number else 99)
        
        # Extract race metadata from first participation
        p0 = participations[0]
        
        # Create race with all available data
        race = cls(
            race_id=race_id,
            race_date=p0.race_datetime.date() if p0.race_datetime else None,
            race_time=p0.race_datetime.time() if p0.race_datetime else None,
            distance=int(p0.distance) if p0.distance else None,
            race_class=p0.race_class,
            category=getattr(p0, 'category', None),
            track_name=p0.track_name,
            odds_vec=[getattr(p, 'sp', None) for p in participations],
            race_time_vec=[p.run_time for p in participations],
            commentary_tags_vec=[getattr(p, 'comment', '').split() for p in participations],
            dog_ids=[p.dog_id for p in participations],
            track=track
        )
        
        # Set additional attributes
        race.participations = participations
        race.race_datetime = p0.race_datetime
        race.race_distance = p0.distance
        race.going = p0.going
        
        return race

    def add_participation(self, participation: RaceParticipation):
        """Add a participation to this race"""
        self.participations.append(participation)
        
        # Update dog_ids list
        if participation.dog_id not in self.dog_ids:
            self.dog_ids.append(participation.dog_id)
        
        # Set race metadata from first participation if not set
        if not self.race_datetime and participation.race_datetime:
            self.race_datetime = participation.race_datetime
            self.race_date = participation.race_datetime.date()
            self.race_time = participation.race_datetime.time()
        if not self.race_distance and participation.distance:
            self.race_distance = participation.distance
            self.distance = int(participation.distance)
        if not self.race_class and participation.race_class:
            self.race_class = participation.race_class
        if not self.going and participation.going:
            self.going = participation.going
        if not self.track_name and participation.track_name:
            self.track_name = participation.track_name

    def get_winner(self) -> Optional[RaceParticipation]:
        """Get the winning participation (position 1)"""
        for participation in self.participations:
            # Handle both string and numeric positions
            pos = participation.position
            if pos == "1" or pos == 1 or pos == 1.0:
                return participation
        return None

    def get_participants_count(self) -> int:
        """Get number of participants in this race"""
        return len(self.participations)

    def __repr__(self):
        track_name = self.track.name if self.track else (self.track_name or "Unknown Track")
        date_str = self.race_datetime.strftime("%Y-%m-%d") if self.race_datetime else "Unknown Date"
        return f"Race({self.race_id}, {track_name}, {date_str}, {len(self.participations)} dogs)"

    def print_info(self):
        """Print detailed race information with all participation details"""
        winner = self.get_winner()
        
        print({
            "race_id": self.race_id,
            "race_datetime": self.race_datetime,
            "track": self.track.name if self.track else self.track_name,
            "distance": self.race_distance or self.distance,
            "race_class": self.race_class,
            "going": self.going,
            "participants": len(self.participations),
            "winner": winner.dog_id if winner else None
        })
        
        # Print detailed participation information
        print("\nDetailed Participants:")
        print("-" * 80)
        
        # Sort participations by position for display
        sorted_participations = sorted(self.participations, 
                                     key=lambda p: float(p.position) if p.position and str(p.position).replace('.','').isdigit() else 99.0)
        
        for i, participation in enumerate(sorted_participations):
            print(f"  {i+1}. Trap {participation.trap_number}: Dog {participation.dog_id}")
            print(f"     Position: {participation.position}")
            print(f"     Run Time: {participation.run_time}s" if participation.run_time else "     Run Time: N/A")
            print(f"     Split Time: {participation.split_time}s" if participation.split_time else "     Split Time: N/A")
            print(f"     Weight: {participation.weight}kg" if participation.weight else "     Weight: N/A")
            print(f"     Starting Price: {participation.sp}" if participation.sp else "     Starting Price: N/A")
            print(f"     Btn Distance: {participation.btn_distance}" if participation.btn_distance else "     Btn Distance: N/A")
            print(f"     Comments: {participation.comment}" if participation.comment else "     Comments: None")
            print(f"     Adjusted Time: {participation.adjusted_time}s" if participation.adjusted_time else "     Adjusted Time: N/A")
            
            # Show winner information if this dog won
            if participation.winner_id:
                print(f"     Winner ID: {participation.winner_id}")
            
            print()  # Empty line between dogs

    def get_race_summary(self):
        """Get a comprehensive race summary with all details"""
        summary = {
            'race_metadata': {
                'race_id': self.race_id,
                'race_datetime': self.race_datetime,
                'track': self.track.name if self.track else self.track_name,
                'distance': self.race_distance or self.distance,
                'race_class': self.race_class,
                'going': self.going,
                'participants_count': len(self.participations)
            },
            'participants': [],
            'race_statistics': {}
        }
        
        # Add detailed participant information
        for participation in self.participations:
            participant_data = {
                'dog_id': participation.dog_id,
                'trap_number': participation.trap_number,
                'position': participation.position,
                'run_time': participation.run_time,
                'split_time': participation.split_time,
                'weight': participation.weight,
                'starting_price': participation.sp,
                'btn_distance': participation.btn_distance,
                'comments': participation.comment,
                'adjusted_time': participation.adjusted_time,
                'winner_id': participation.winner_id
            }
            summary['participants'].append(participant_data)
        
        # Calculate race statistics
        run_times = [p.run_time for p in self.participations if p.run_time]
        if run_times:
            summary['race_statistics'] = {
                'fastest_time': min(run_times),
                'slowest_time': max(run_times),
                'average_time': sum(run_times) / len(run_times),
                'winning_time': self.get_winner().run_time if self.get_winner() and self.get_winner().run_time else None
            }
        
        return summary

    @staticmethod
    def save(race: "Race", path: str):
        with open(path, "wb") as f:
            pickle.dump(race, f)

    @staticmethod
    def load(path: str) -> "Race":
        with open(path, "rb") as f:
            return pickle.load(f)

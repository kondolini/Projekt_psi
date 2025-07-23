import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog

class ChronologicalTrainer:
    """Manages chronological training data generation for AI models"""
    
    def __init__(self, races: Dict[str, Race], dogs: Dict[str, Dog]):
        self.races = races
        self.dogs = dogs
        self.chronological_races = Race.get_races_chronologically(races)
        
    def generate_training_sequence(self, start_date: datetime = None, end_date: datetime = None) -> List[Tuple[Race, Dict[str, dict]]]:
        """
        Generate chronologically ordered training data.
        Returns: List of (race, dog_memory_states) tuples
        
        Each dog_memory_states contains the feature vector for each dog AT THE TIME OF THE RACE
        (ensuring no data leakage from future races)
        """
        training_sequence = []
        dog_memory_vectors = defaultdict(dict)  # Per-dog memory state
        
        filtered_races = self.chronological_races
        if start_date:
            filtered_races = [r for r in filtered_races if r.get_race_datetime() >= start_date]
        if end_date:
            filtered_races = [r for r in filtered_races if r.get_race_datetime() <= end_date]
        
        print(f"Processing {len(filtered_races)} races chronologically...")
        
        for i, race in enumerate(filtered_races):
            race_datetime = race.get_race_datetime()
            race_dog_states = {}
            
            # For each dog in this race, get their state BEFORE this race
            for trap, dog_id in race.dog_ids.items():
                if dog_id in self.dogs:
                    dog = self.dogs[dog_id]
                    
                    # Get dog's form at this point in time (no data leakage)
                    form_vector = dog.get_form_vector_at_time(race_datetime)
                    
                    # Include previous memory vector if exists
                    if dog_id in dog_memory_vectors:
                        form_vector['memory_vector'] = dog_memory_vectors[dog_id]
                    
                    race_dog_states[dog_id] = form_vector
            
            training_sequence.append((race, race_dog_states))
            
            # Update memory vectors AFTER processing the race
            # (this simulates the model learning from each race result)
            for trap, dog_id in race.dog_ids.items():
                if dog_id in self.dogs:
                    # This is where you'd update the dog's memory vector
                    # based on this race's outcome
                    dog_memory_vectors[dog_id] = self.update_memory_vector(
                        dog_memory_vectors.get(dog_id, {}), 
                        race, 
                        trap
                    )
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(filtered_races)} races...")
        
        return training_sequence
    
    def update_memory_vector(self, current_memory: dict, race: Race, trap: int) -> dict:
        """
        Update a dog's memory vector based on race outcome.
        This is where you'd implement the memory update logic.
        """
        # Example simple memory update
        new_memory = current_memory.copy()
        
        # Update based on race performance
        actual_time = race.race_times.get(trap)
        if actual_time:
            new_memory['last_race_time'] = actual_time
            new_memory['races_count'] = new_memory.get('races_count', 0) + 1
        
        return new_memory
    
    def train_test_split_by_date(self, split_date: datetime) -> Tuple[List, List]:
        """Split data chronologically by date"""
        train_races = [r for r in self.chronological_races if r.get_race_datetime() < split_date]
        test_races = [r for r in self.chronological_races if r.get_race_datetime() >= split_date]
        
        return train_races, test_races

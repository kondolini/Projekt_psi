"""
Data processing utilities for converting Race objects to ML-ready tensors
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import pickle
import os

from models.race import Race
from models.dog import Dog
from models.race_participation import RaceParticipation


class CommentaryProcessor:
    """Processes commentary tags into embeddings"""
    
    def __init__(self, max_vocab_size: int = 1000):
        self.max_vocab_size = max_vocab_size
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.vocab_size = 0
        
    def build_vocabulary(self, all_participations: List[RaceParticipation]):
        """Build vocabulary from all commentary tags"""
        tag_counter = Counter()
        
        for participation in all_participations:
            if participation.comment:
                tags = participation.commentary_tags
                for tag in tags:
                    if tag.strip():  # Only non-empty tags
                        tag_counter[tag.strip().lower()] += 1
        
        # Keep most common tags
        most_common = tag_counter.most_common(self.max_vocab_size - 2)  # -2 for UNK and PAD
        
        # Build mappings
        self.tag_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_tag = {0: "<PAD>", 1: "<UNK>"}
        
        for i, (tag, count) in enumerate(most_common, 2):
            self.tag_to_idx[tag] = i
            self.idx_to_tag[i] = tag
            
        self.vocab_size = len(self.tag_to_idx)
        print(f"Built commentary vocabulary: {self.vocab_size} tags")
        
    def encode_tags(self, tags: List[str], max_length: int = 10) -> List[int]:
        """Convert tags to indices, pad/truncate to max_length"""
        if not tags:
            return [0] * max_length  # All padding
            
        indices = []
        for tag in tags[:max_length]:  # Truncate if too long
            clean_tag = tag.strip().lower()
            idx = self.tag_to_idx.get(clean_tag, 1)  # 1 = UNK
            indices.append(idx)
            
        # Pad if too short
        while len(indices) < max_length:
            indices.append(0)  # 0 = PAD
            
        return indices


class RaceDataProcessor:
    """Converts Race objects to ML-ready features"""
    
    def __init__(self):
        self.track_encoder = {}
        self.class_encoder = {}
        self.category_encoder = {}
        self.trainer_encoder = {}
        self.going_encoder = {}
        self.commentary_processor = CommentaryProcessor()
        
        # Statistics for normalization
        self.distance_stats = {"mean": 0, "std": 1}
        self.weight_stats = {"mean": 0, "std": 1}
        self.time_stats = {"mean": 0, "std": 1}
        
    def fit_encoders(self, races: List[Race], dogs: Dict[str, Dog], cache_path: str = None, force_rebuild: bool = False):
        """
        Fit all encoders on the training data with caching support
        
        Args:
            races: Training races
            dogs: Dictionary of all dogs  
            cache_path: Path to save/load encoders cache
            force_rebuild: Force rebuild even if cache exists
        """
        
        # Try to load cached encoders first (unless forced rebuild)
        if cache_path and os.path.exists(cache_path) and not force_rebuild:
            try:
                print(f"Loading cached encoders from: {cache_path}")
                self.load_encoders(cache_path)
                print("✅ Loaded cached encoders successfully!")
                return
            except Exception as e:
                print(f"⚠️  Failed to load cached encoders: {e}")
                print("Rebuilding encoders from scratch...")
        
        if force_rebuild and cache_path and os.path.exists(cache_path):
            print("🔄 Force rebuilding encoders cache...")
        
        print("Fitting encoders on training data...")
        
        # Collect all unique values
        tracks = set()
        classes = set()
        categories = set()
        trainers = set()
        goings = set()
        distances = []
        weights = []
        times = []
        all_participations = []
        
        for race in races:
            tracks.add(race.track_name)
            classes.add(race.race_class)
            categories.add(race.category if race.category is not None else "unknown")
            
            for trap, dog_id in race.dog_ids.items():
                dog = dogs.get(dog_id)
                if dog:
                    if dog.trainer:
                        trainers.add(dog.trainer)
                    if dog.weight:
                        weights.append(dog.weight)
                    
                    # Collect historical participations for this dog
                    historical = dog.get_participations_up_to(race.get_race_datetime())
                    all_participations.extend(historical)
                    
                    for participation in historical:
                        if participation.distance:
                            distances.append(participation.distance)
                        if participation.run_time:
                            times.append(participation.run_time)
                        if participation.going:
                            goings.add(participation.going)
        
        # Build encoders with safe handling of mixed types
        def safe_sort(items):
            """Sort items safely, handling mixed types by converting to strings"""
            return sorted([str(item) for item in items])
        
        self.track_encoder = {str(track): i for i, track in enumerate(safe_sort(tracks))}
        self.class_encoder = {str(cls): i for i, cls in enumerate(safe_sort(classes))}
        self.category_encoder = {str(cat): i for i, cat in enumerate(safe_sort(categories))}
        self.trainer_encoder = {str(trainer): i for i, trainer in enumerate(safe_sort(trainers))}
        self.going_encoder = {str(going): i for i, going in enumerate(safe_sort(goings))}
        
        # Calculate normalization statistics
        if distances:
            self.distance_stats = {"mean": np.mean(distances), "std": np.std(distances)}
        if weights:
            self.weight_stats = {"mean": np.mean(weights), "std": np.std(weights)}
        if times:
            self.time_stats = {"mean": np.mean(times), "std": np.std(times)}
            
        # Build commentary vocabulary
        self.commentary_processor.build_vocabulary(all_participations)
        
        print(f"Encoder sizes - Tracks: {len(self.track_encoder)}, Classes: {len(self.class_encoder)}, "
              f"Categories: {len(self.category_encoder)}, Trainers: {len(self.trainer_encoder)}")
        
        # Save encoders to cache if path provided
        if cache_path:
            self.save_encoders(cache_path)
            print(f"💾 Saved encoders to cache: {cache_path}")
    
    def process_race(self, race: Race, dogs: Dict[str, Dog], max_history: int = 10) -> Dict:
        """Convert a single race to ML features"""
        
        # Race-level features
        race_features = self._extract_race_features(race)
        
        # Per-dog features
        dog_features = []
        target_labels = []
        
        for trap in range(1, 7):  # Assuming max 6 traps
            dog_id = race.dog_ids.get(trap)
            if dog_id and dog_id in dogs:
                dog = dogs[dog_id]
                
                # Extract features for this dog
                features = self._extract_dog_features(dog, race, max_history)
                dog_features.append(features)
                
                # Extract target (whether this dog won)
                won = self._extract_target(race, trap)
                target_labels.append(won)
            else:
                # Empty trap or missing dog
                dog_features.append(self._get_empty_dog_features(max_history))
                target_labels.append(0)
        
        # Extract market odds for each trap (0 if no odds - test race)
        market_odds = []
        for trap in range(1, 7):
            if race.odds and trap in race.odds and race.odds[trap] is not None:
                market_odds.append(float(race.odds[trap]))
            else:
                market_odds.append(0.0)  # No odds available (test race)
        
        return {
            "race_features": race_features,
            "dog_features": dog_features,  # List of 6 dog feature dicts
            "targets": target_labels,      # List of 6 binary labels
            "market_odds": market_odds,    # List of 6 market odds (0 for test races)
            "race_id": race.race_id,
            "meeting_id": race.meeting_id,
            "race_datetime": race.get_race_datetime(),
            "is_test_race": race.is_test_race(),  # Flag for races without odds
            "has_complete_field": race.has_complete_field()  # Flag for races with complete participant data
        }
    
    def _extract_race_features(self, race: Race) -> Dict:
        """Extract race-level features"""
        # Date/time features
        race_dt = race.get_race_datetime()
        day_of_week = race_dt.weekday()
        month = race_dt.month
        hour = race_dt.hour
        minute = race_dt.minute
        
        # Categorical features (one-hot encoded indices) - use string keys consistently
        track_idx = self.track_encoder.get(str(race.track_name), 0)
        class_idx = self.class_encoder.get(str(race.race_class), 0)
        category_idx = self.category_encoder.get(
            str(race.category if race.category is not None else "unknown"), 0
        )
        
        # Normalized distance
        distance_norm = (race.distance - self.distance_stats["mean"]) / self.distance_stats["std"]
        
        return {
            "day_of_week": day_of_week,
            "month": month,
            "hour": hour,
            "minute": minute,
            "track_idx": track_idx,
            "class_idx": class_idx,
            "category_idx": category_idx,
            "distance_norm": distance_norm
        }
    
    def _extract_dog_features(self, dog: Dog, race: Race, max_history: int) -> Dict:
        """Extract features for a single dog"""
        # Static features
        trainer_idx = self.trainer_encoder.get(str(dog.trainer), 0) if dog.trainer else 0
        weight_norm = ((dog.weight or self.weight_stats["mean"]) - self.weight_stats["mean"]) / self.weight_stats["std"]
        
        # Dog ID hash (simple embedding)
        dog_id_hash = hash(dog.id) % 10000  # Arbitrary large number for embeddings
        
        # Historical participations (for RNN input)
        historical = dog.get_participations_up_to(race.get_race_datetime())
        history_features = self._extract_history_features(historical[-max_history:], max_history)
        
        return {
            "trainer_idx": trainer_idx,
            "weight_norm": weight_norm,
            "dog_id_hash": dog_id_hash,
            "history": history_features  # Dict with sequences for RNN
        }
    
    def _extract_history_features(self, participations: List[RaceParticipation], max_length: int) -> Dict:
        """Extract sequential features from race history"""
        sequences = {
            "positions": [],
            "distances": [],
            "times": [],
            "going_indices": [],
            "commentary_indices": []
        }
        
        for participation in participations:
            # Position (1-based, 0 for unknown)
            try:
                position = int(participation.position) if participation.position else 0
            except:
                position = 0
            sequences["positions"].append(position)
            
            # Distance (normalized)
            distance = participation.distance or self.distance_stats["mean"]
            distance_norm = (distance - self.distance_stats["mean"]) / self.distance_stats["std"]
            sequences["distances"].append(distance_norm)
            
            # Run time (normalized)
            time = participation.run_time or self.time_stats["mean"]
            time_norm = (time - self.time_stats["mean"]) / self.time_stats["std"]
            sequences["times"].append(time_norm)
            
            # Going condition
            going_idx = self.going_encoder.get(str(participation.going), 0) if participation.going else 0
            sequences["going_indices"].append(going_idx)
            
            # Commentary tags
            tags = participation.commentary_tags if participation.comment else []
            tag_indices = self.commentary_processor.encode_tags(tags, max_length=5)
            sequences["commentary_indices"].append(tag_indices)
        
        # Pad sequences to max_length
        for key, seq in sequences.items():
            if key == "commentary_indices":
                # Special handling for nested list
                while len(seq) < max_length:
                    seq.append([0] * 5)  # Padding with zeros
                sequences[key] = seq[:max_length]  # Truncate if too long
            else:
                while len(seq) < max_length:
                    seq.append(0)
                sequences[key] = seq[:max_length]
        
        return sequences
    
    def _get_empty_dog_features(self, max_history: int) -> Dict:
        """Get features for empty trap"""
        return {
            "trainer_idx": 0,
            "weight_norm": 0.0,
            "dog_id_hash": 0,
            "history": {
                "positions": [0] * max_history,
                "distances": [0.0] * max_history,
                "times": [0.0] * max_history,
                "going_indices": [0] * max_history,
                "commentary_indices": [[0] * 5] * max_history
            }
        }
    
    def _extract_target(self, race: Race, trap: int) -> int:
        """Extract target label (1 if won, 0 otherwise)"""
        # Check if this trap won by looking at race times
        if not race.race_times:
            return 0
            
        trap_time = race.race_times.get(trap)
        if not trap_time:
            return 0
            
        # Check if this is the minimum time (winner)
        all_times = [t for t in race.race_times.values() if t is not None]
        if not all_times:
            return 0
            
        min_time = min(all_times)
        return 1 if trap_time == min_time else 0
    
    def save_encoders(self, path: str):
        """Save all encoders and statistics"""
        encoder_data = {
            "track_encoder": self.track_encoder,
            "class_encoder": self.class_encoder,
            "category_encoder": self.category_encoder,
            "trainer_encoder": self.trainer_encoder,
            "going_encoder": self.going_encoder,
            "distance_stats": self.distance_stats,
            "weight_stats": self.weight_stats,
            "time_stats": self.time_stats,
            "commentary_processor": self.commentary_processor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(encoder_data, f)
            
        print(f"Saved encoders to {path}")
    
    def load_encoders(self, path: str):
        """Load all encoders and statistics"""
        with open(path, 'rb') as f:
            encoder_data = pickle.load(f)
            
        self.track_encoder = encoder_data["track_encoder"]
        self.class_encoder = encoder_data["class_encoder"]
        self.category_encoder = encoder_data["category_encoder"]
        self.trainer_encoder = encoder_data["trainer_encoder"]
        self.going_encoder = encoder_data["going_encoder"]
        self.distance_stats = encoder_data["distance_stats"]
        self.weight_stats = encoder_data["weight_stats"]
        self.time_stats = encoder_data["time_stats"]
        self.commentary_processor = encoder_data["commentary_processor"]
        
        print(f"Loaded encoders from {path}")


def create_dataset(races: List[Race], dogs: Dict[str, Dog], processor: RaceDataProcessor, cache_path: str = None, force_rebuild: bool = False) -> List[Dict]:
    """
    Create ML dataset from races and dogs with caching support
    
    Args:
        races: List of race objects
        dogs: Dictionary of dog objects
        processor: Fitted data processor
        cache_path: Path to save/load dataset cache
        force_rebuild: Force rebuild even if cache exists
        
    Returns:
        List of processed race samples
    """
    
    # Try to load cached dataset first (unless forced rebuild)
    if cache_path and os.path.exists(cache_path) and not force_rebuild:
        try:
            print(f"Loading cached dataset from: {cache_path}")
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"✅ Loaded {len(dataset)} cached samples!")
            return dataset
        except Exception as e:
            print(f"⚠️  Failed to load cached dataset: {e}")
            print("Processing dataset from scratch...")
    
    if force_rebuild and cache_path and os.path.exists(cache_path):
        print("🔄 Force rebuilding dataset cache...")
    
    # Process dataset from scratch
    dataset = []
    
    print(f"Processing {len(races)} races...")
    for i, race in enumerate(races):
        try:
            features = processor.process_race(race, dogs)
            dataset.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(races)} races")
                
        except Exception as e:
            print(f"Error processing race {race.race_id}: {e}")
            continue
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Save dataset to cache if path provided
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"💾 Saved dataset to cache: {cache_path}")
        except Exception as e:
            print(f"⚠️  Failed to save dataset cache: {e}")
    
    return dataset

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date
from collections import defaultdict, Counter
import re
from sklearn.preprocessing import LabelEncoder
import logging

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from models.track import Track

logger = logging.getLogger(__name__)


class GreyhoundDataset(Dataset):
    """
    PyTorch Dataset for greyhound racing data
    
    Handles loading races, dogs, and creating training examples with proper
    temporal ordering to prevent data leakage.
    """
    
    def __init__(self,
                 races: List[Race],
                 dog_lookup: Dict[str, Dog],
                 track_lookup: Dict[str, Track],
                 max_dogs_per_race: int = 8,
                 max_history_length: int = 10,
                 max_commentary_length: int = 5,
                 min_dogs_per_race: int = 3,
                 exclude_trial_races: bool = True):
        """
        Initialize dataset
        
        Args:
            races: List of Race objects sorted chronologically
            dog_lookup: Dictionary mapping dog_id to Dog objects
            track_lookup: Dictionary mapping track_name to Track objects
            max_dogs_per_race: Maximum number of dogs per race (for padding)
            max_history_length: Maximum number of historical races per dog
            max_commentary_length: Maximum number of commentary tags per race
            min_dogs_per_race: Minimum dogs required for valid race
            exclude_trial_races: Whether to exclude races without odds
        """
        self.dog_lookup = dog_lookup
        self.track_lookup = track_lookup
        self.max_dogs_per_race = max_dogs_per_race
        self.max_history_length = max_history_length
        self.max_commentary_length = max_commentary_length
        self.min_dogs_per_race = min_dogs_per_race
        
        # Filter valid races
        self.races = self._filter_valid_races(races, exclude_trial_races)
        
        # Build vocabularies and encoders
        self._build_encoders()
        
        logger.info(f"Dataset initialized with {len(self.races)} valid races")
        
    def _filter_valid_races(self, races: List[Race], exclude_trial_races: bool) -> List[Race]:
        """Filter races that meet criteria for training"""
        valid_races = []
        
        for race in races:
            # Skip trial races if specified
            if exclude_trial_races and race.is_trial_race():
                continue
                
            # Check minimum number of dogs
            if len(race.dog_ids) < self.min_dogs_per_race:
                continue
                
            # Check if race has complete field (for betting races)
            if not exclude_trial_races and not race.has_complete_field():
                continue
                
            # Ensure we have data for all dogs in the race
            dogs_available = sum(1 for dog_id in race.dog_ids.values() 
                               if dog_id in self.dog_lookup)
            
            if dogs_available < self.min_dogs_per_race:
                continue
                
            valid_races.append(race)
            
        return valid_races
    
    def _build_encoders(self):
        """Build label encoders for categorical features"""
        
        # Collect all unique values
        tracks = set()
        classes = set()
        categories = set()
        trainers = set()
        going_conditions = set()
        commentary_vocab = set()
        
        for race in self.races:
            tracks.add(race.track_name)
            classes.add(race.race_class or 'Unknown')
            categories.add(race.category or 'Unknown')
            
            # Get dog information
            for dog_id in race.dog_ids.values():
                if dog_id in self.dog_lookup:
                    dog = self.dog_lookup[dog_id]
                    if dog.trainer:
                        trainers.add(dog.trainer)
                    
                    # Get historical race information
                    race_datetime = datetime.combine(race.race_date, race.race_time)
                    for participation in dog.get_participations_up_to(race_datetime):
                        if participation.going:
                            going_conditions.add(participation.going)
                        
                        # Extract commentary vocabulary
                        for tag in participation.commentary_tags:
                            # Simple tokenization
                            words = re.findall(r'\b\w+\b', tag.lower())
                            commentary_vocab.update(words)
            
            # Add race commentary
            for tags in race.commentary_tags.values():
                for tag in tags:
                    words = re.findall(r'\b\w+\b', tag.lower())
                    commentary_vocab.update(words)
        
        # Build encoders
        self.track_encoder = LabelEncoder()
        self.track_encoder.fit(list(tracks) + ['Unknown'])
        
        self.class_encoder = LabelEncoder()
        self.class_encoder.fit(list(classes) + ['Unknown'])
        
        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(list(categories) + ['Unknown'])
        
        self.trainer_encoder = LabelEncoder()
        self.trainer_encoder.fit(list(trainers) + ['Unknown'])
        
        self.going_encoder = LabelEncoder()
        self.going_encoder.fit(list(going_conditions) + ['Unknown'])
        
        # Commentary vocabulary (add special tokens)
        commentary_list = list(commentary_vocab) + ['<PAD>', '<UNK>']
        self.commentary_vocab = {word: idx for idx, word in enumerate(commentary_list)}
        
        # Store sizes for model initialization
        self.vocab_sizes = {
            'num_tracks': len(self.track_encoder.classes_),
            'num_classes': len(self.class_encoder.classes_),
            'num_categories': len(self.category_encoder.classes_),
            'num_trainers': len(self.trainer_encoder.classes_),
            'num_going_conditions': len(self.going_encoder.classes_),
            'commentary_vocab_size': len(self.commentary_vocab)
        }
        
        logger.info(f"Vocabulary sizes: {self.vocab_sizes}")
    
    def __len__(self):
        return len(self.races)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        race = self.races[idx]
        race_datetime = datetime.combine(race.race_date, race.race_time)
        
        # === Race-level features ===
        race_features = self._extract_race_features(race)
        track_id = self._encode_track(race.track_name)
        class_id = self._encode_class(race.race_class)
        category_id = self._encode_category(race.category)
        
        # === Dog-level features ===
        dogs_data = self._extract_dogs_data(race, race_datetime)
        
        # Pad or truncate to max_dogs_per_race
        dogs_data = self._pad_dogs_data(dogs_data)
        
        return {
            # Race features
            'race_features': torch.FloatTensor(race_features),
            'track_ids': track_id,
            'class_ids': class_id,
            'category_ids': category_id,
            
            # Dog features
            'dog_features': torch.FloatTensor(dogs_data['dog_features']),
            'trainer_ids': torch.LongTensor(dogs_data['trainer_ids']),
            'dog_ids': torch.LongTensor(dogs_data['dog_ids']),
            'dog_mask': torch.BoolTensor(dogs_data['dog_mask']),
            
            # History features
            'history_features': torch.FloatTensor(dogs_data['history_features']),
            'going_ids': torch.LongTensor(dogs_data['going_ids']),
            'commentary_ids': torch.LongTensor(dogs_data['commentary_ids']),
            'history_mask': torch.BoolTensor(dogs_data['history_mask']),
            
            # Targets and odds
            'win_labels': torch.FloatTensor(dogs_data['win_labels']),
            'market_odds': torch.FloatTensor(dogs_data['market_odds']),
            
            # Metadata
            'race_id': race.race_id,
            'race_date': race.race_date.isoformat()
        }
    
    def _extract_race_features(self, race: Race) -> List[float]:
        """Extract race-level numerical features"""
        race_datetime = datetime.combine(race.race_date, race.race_time)
        
        # Temporal features
        day_of_week = race_datetime.weekday() / 6.0  # 0-1 normalized
        month = (race_datetime.month - 1) / 11.0  # 0-1 normalized
        hour = race_datetime.hour / 23.0  # 0-1 normalized
        minute = race_datetime.minute / 59.0  # 0-1 normalized
        
        # Distance normalization (typical range 200-800m)
        distance_norm = min(max((race.distance - 200) / 600, 0.0), 1.0)
        
        # Weather features (with defaults if missing)
        temp_norm = (race.temperature - 0) / 40.0 if race.temperature else 0.375  # 15°C default
        humidity_norm = race.humidity / 100.0 if race.humidity else 0.5  # 50% default
        rainfall_sum = sum(race.rainfall_7d) if race.rainfall_7d else 0.0  # Total 7-day rainfall
        
        return [day_of_week, month, hour, minute, distance_norm, temp_norm, humidity_norm, rainfall_sum]
    
    def _extract_dogs_data(self, race: Race, race_datetime: datetime) -> Dict[str, List]:
        """Extract all dog-related data for a race"""
        dogs_data = {
            'dog_features': [],
            'trainer_ids': [],
            'dog_ids': [],
            'dog_mask': [],
            'history_features': [],
            'going_ids': [],
            'commentary_ids': [],
            'history_mask': [],
            'win_labels': [],
            'market_odds': []
        }
        
        # Sort dogs by trap number for consistency
        sorted_dogs = sorted(race.dog_ids.items())
        
        for trap_num, dog_id in sorted_dogs:
            if dog_id not in self.dog_lookup:
                continue
                
            dog = self.dog_lookup[dog_id]
            
            # === Static dog features ===
            dog_features = self._extract_dog_static_features(dog, race, trap_num, race_datetime)
            dogs_data['dog_features'].append(dog_features)
            
            # === Dog encodings ===
            trainer_id = self._encode_trainer(dog.trainer)
            dog_hash_id = hash(dog_id) % 100000  # Hash to fixed range
            dogs_data['trainer_ids'].append(trainer_id)
            dogs_data['dog_ids'].append(dog_hash_id)
            dogs_data['dog_mask'].append(True)
            
            # === Historical features ===
            history_data = self._extract_dog_history(dog, race_datetime)
            dogs_data['history_features'].append(history_data['features'])
            dogs_data['going_ids'].append(history_data['going_ids'])
            dogs_data['commentary_ids'].append(history_data['commentary_ids'])
            dogs_data['history_mask'].append(history_data['mask'])
            
            # === Labels and odds ===
            # Win label (1 if winner, 0 otherwise)
            win_label = 1.0 if race.race_times.get(trap_num) and self._is_winner(race, trap_num) else 0.0
            dogs_data['win_labels'].append(win_label)
            
            # Market odds (use 1.0 if missing - will be masked anyway)
            market_odds = race.odds.get(trap_num, 1.0) if race.odds else 1.0
            dogs_data['market_odds'].append(market_odds)
        
        return dogs_data
    
    def _extract_dog_static_features(self, dog: Dog, race: Race, trap_num: int, race_datetime: datetime) -> List[float]:
        """Extract static features for a dog in a race"""
        
        # Weight normalization (typical range 25-40kg)
        weight = race.weights.get(trap_num) or dog.weight or 32.5  # Use race weight if available, else dog weight, else default
        weight_norm = min(max((weight - 25) / 15, 0.0), 1.0)
        
        # Age calculation (if birth date available)
        if dog.birth_date:
            age_days = (race_datetime.date() - dog.birth_date.date()).days
            age_days_norm = min(age_days / 2190, 1.0)  # 6 years max
        else:
            age_days_norm = 0.5  # Default to middle age
            
        # Trap position normalization (typically 1-8)
        trap_norm = (trap_num - 1) / 7.0 if trap_num else 0.0
        
        return [weight_norm, age_days_norm, trap_norm]
    
    def _extract_dog_history(self, dog: Dog, race_datetime: datetime) -> Dict[str, List]:
        """Extract historical race data for a dog"""
        
        # Get recent participations before this race
        historical_participations = dog.get_last_n_races_before(race_datetime, self.max_history_length)
        
        features = []
        going_ids = []
        commentary_ids = []
        mask = []
        
        for participation in historical_participations:
            # Position feature (normalize by typical field size)
            position = self._parse_position(participation.position)
            position_norm = min(position / 8.0, 1.0) if position else 0.5
            
            # Distance normalization
            distance_norm = min(max((participation.distance - 200) / 600, 0.0), 1.0) if participation.distance else 0.5
            
            # Time normalization (typical range 20-40 seconds)
            time_norm = min(max((participation.run_time - 20) / 20, 0.0), 1.0) if participation.run_time else 0.5
            
            features.append([position_norm, distance_norm, time_norm])
            
            # Going condition
            going_id = self._encode_going(participation.going)
            going_ids.append(going_id)
            
            # Commentary
            commentary_encoded = self._encode_commentary(participation.commentary_tags)
            commentary_ids.append(commentary_encoded)
            
            mask.append(True)
        
        # Pad to max_history_length
        while len(features) < self.max_history_length:
            features.append([0.0, 0.0, 0.0])  # Zero padding
            going_ids.append(self._encode_going('Unknown'))
            commentary_ids.append([self.commentary_vocab['<PAD>']] * self.max_commentary_length)
            mask.append(False)
        
        return {
            'features': features,
            'going_ids': going_ids,
            'commentary_ids': commentary_ids,
            'mask': mask
        }
    
    def _pad_dogs_data(self, dogs_data: Dict[str, List]) -> Dict[str, List]:
        """Pad dog data to max_dogs_per_race"""
        
        current_dogs = len(dogs_data['dog_features'])
        
        # Pad to max_dogs_per_race
        while len(dogs_data['dog_features']) < self.max_dogs_per_race:
            dogs_data['dog_features'].append([0.0, 0.0, 0.0])  # Padding features
            dogs_data['trainer_ids'].append(self._encode_trainer('Unknown'))
            dogs_data['dog_ids'].append(0)  # Padding dog ID
            dogs_data['dog_mask'].append(False)  # Mark as padding
            
            # Pad history
            padded_history = [[0.0, 0.0, 0.0]] * self.max_history_length
            dogs_data['history_features'].append(padded_history)
            dogs_data['going_ids'].append([self._encode_going('Unknown')] * self.max_history_length)
            dogs_data['commentary_ids'].append([[self.commentary_vocab['<PAD>']] * self.max_commentary_length] * self.max_history_length)
            dogs_data['history_mask'].append([False] * self.max_history_length)
            
            # Pad labels and odds
            dogs_data['win_labels'].append(0.0)
            dogs_data['market_odds'].append(1.0)
        
        # Truncate if too many dogs
        for key in dogs_data:
            dogs_data[key] = dogs_data[key][:self.max_dogs_per_race]
            
        return dogs_data
    
    def _is_winner(self, race: Race, trap_num: int) -> bool:
        """Determine if dog in trap won the race"""
        if not race.race_times:
            return False
            
        # Find the fastest time
        valid_times = {t: time for t, time in race.race_times.items() if time and time > 0}
        if not valid_times:
            return False
            
        fastest_time = min(valid_times.values())
        fastest_trap = [t for t, time in valid_times.items() if time == fastest_time][0]
        
        return trap_num == fastest_trap
    
    def _parse_position(self, position_str: Optional[str]) -> Optional[int]:
        """Parse position string to integer"""
        if not position_str:
            return None
        try:
            # Extract first number from position string
            import re
            match = re.search(r'\d+', str(position_str))
            return int(match.group()) if match else None
        except:
            return None
    
    def _encode_track(self, track_name: Optional[str]) -> int:
        """Encode track name to integer"""
        name = track_name or 'Unknown'
        try:
            return self.track_encoder.transform([name])[0]
        except:
            return self.track_encoder.transform(['Unknown'])[0]
    
    def _encode_class(self, race_class: Optional[str]) -> int:
        """Encode race class to integer"""
        cls = race_class or 'Unknown'
        try:
            return self.class_encoder.transform([cls])[0]
        except:
            return self.class_encoder.transform(['Unknown'])[0]
    
    def _encode_category(self, category: Optional[str]) -> int:
        """Encode category to integer"""
        cat = category or 'Unknown'
        try:
            return self.category_encoder.transform([cat])[0]
        except:
            return self.category_encoder.transform(['Unknown'])[0]
    
    def _encode_trainer(self, trainer: Optional[str]) -> int:
        """Encode trainer to integer"""
        tr = trainer or 'Unknown'
        try:
            return self.trainer_encoder.transform([tr])[0]
        except:
            return self.trainer_encoder.transform(['Unknown'])[0]
    
    def _encode_going(self, going: Optional[str]) -> int:
        """Encode going condition to integer"""
        go = going or 'Unknown'
        try:
            return self.going_encoder.transform([go])[0]
        except:
            return self.going_encoder.transform(['Unknown'])[0]
    
    def _encode_commentary(self, tags: List[str]) -> List[int]:
        """Encode commentary tags to integers"""
        encoded = []
        
        for tag in tags[:self.max_commentary_length]:
            words = re.findall(r'\b\w+\b', tag.lower())
            for word in words:
                encoded.append(self.commentary_vocab.get(word, self.commentary_vocab['<UNK>']))
                if len(encoded) >= self.max_commentary_length:
                    break
            if len(encoded) >= self.max_commentary_length:
                break
        
        # Pad to max_commentary_length
        while len(encoded) < self.max_commentary_length:
            encoded.append(self.commentary_vocab['<PAD>'])
            
        return encoded[:self.max_commentary_length]


def load_data_from_buckets(dogs_enhanced_dir: str, races_dir: str, unified_dir: str) -> Tuple[Dict[str, Dog], List[Race]]:
    """
    Load dogs and races from bucket files
    
    Returns:
        Tuple of (dog_lookup, races_list)
    """
    logger.info("Loading dogs from enhanced directory...")
    
    # Load all dogs
    dog_lookup = {}
    for bucket_file in os.listdir(dogs_enhanced_dir):
        if bucket_file.startswith('dogs_bucket_') and bucket_file.endswith('.pkl'):
            bucket_path = os.path.join(dogs_enhanced_dir, bucket_file)
            try:
                with open(bucket_path, 'rb') as f:
                    bucket_dogs = pickle.load(f)
                    dog_lookup.update(bucket_dogs)
            except Exception as e:
                logger.error(f"Error loading dog bucket {bucket_file}: {e}")
    
    logger.info(f"Loaded {len(dog_lookup)} dogs")
    
    # Load race index
    race_index_path = os.path.join(unified_dir, 'race_index.pkl')
    with open(race_index_path, 'rb') as f:
        race_index = pickle.load(f)
    
    logger.info("Loading races from buckets...")
    
    # Load all races
    races = []
    loaded_buckets = {}
    
    for race_key, race_info in race_index.items():
        bucket_path = race_info['path']
        storage_key = race_info['key']
        
        # Load bucket if not already loaded
        if bucket_path not in loaded_buckets:
            try:
                with open(bucket_path, 'rb') as f:
                    loaded_buckets[bucket_path] = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading race bucket {bucket_path}: {e}")
                continue
        
        # Get race from bucket
        races_bucket = loaded_buckets[bucket_path]
        if storage_key in races_bucket:
            races.append(races_bucket[storage_key])
    
    # Sort races chronologically
    races.sort(key=lambda r: (r.race_date, r.race_time))
    
    logger.info(f"Loaded {len(races)} races")
    
    return dog_lookup, races


def create_train_val_split(races: List[Race], 
                          val_start_date: date,
                          val_end_date: Optional[date] = None) -> Tuple[List[Race], List[Race]]:
    """
    Create temporal train/validation split
    
    Args:
        races: List of races sorted chronologically
        val_start_date: Start date for validation set
        val_end_date: End date for validation set (if None, use all races after start)
        
    Returns:
        Tuple of (train_races, val_races)
    """
    train_races = []
    val_races = []
    
    for race in races:
        if race.race_date < val_start_date:
            train_races.append(race)
        elif val_end_date is None or race.race_date <= val_end_date:
            val_races.append(race)
        # Races after val_end_date are ignored
    
    logger.info(f"Train/Val split: {len(train_races)} train, {len(val_races)} validation races")
    
    return train_races, val_races


def create_dataloaders(train_dataset: GreyhoundDataset,
                      val_dataset: GreyhoundDataset,
                      batch_size: int = 32,
                      num_workers: int = 0,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

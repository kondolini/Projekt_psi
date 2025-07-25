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
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add parent directory to path for imports - make it more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from models.track import Track
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


def process_race_batch_for_vocab(args) -> Dict[str, set]:
    """
    Process a batch of races to extract vocabulary items (for multiprocessing)
    
    Args:
        args: Tuple of (races_batch, dog_lookup)
        
    Returns:
        Dictionary with vocabulary sets
    """
    races_batch, dog_lookup = args
    
    # Local vocabulary sets
    tracks = set()
    classes = set()
    categories = set()
    trainers = set()
    going_conditions = set()
    commentary_vocab = set()
    
    for race in races_batch:
        tracks.add(race.track_name)
        classes.add(race.race_class or 'Unknown')
        categories.add(race.category or 'Unknown')
        
        # Add race commentary
        for tags in race.commentary_tags.values():
            for tag in tags:
                words = re.findall(r'\b\w+\b', tag.lower())
                commentary_vocab.update(words)
        
        # Get dog information for this race
        for dog_id in race.dog_ids.values():
            if dog_id in dog_lookup:
                dog = dog_lookup[dog_id]
                if dog.trainer:
                    trainers.add(dog.trainer)
                
                # Process historical races for vocabulary
                race_datetime = datetime.combine(race.race_date, race.race_time)
                historical_participations = dog.get_last_n_races_before(race_datetime, 20)
                
                for participation in historical_participations:
                    if participation.going:
                        going_conditions.add(participation.going)
                    
                    # Extract commentary vocabulary
                    for tag in participation.commentary_tags:
                        words = re.findall(r'\b\w+\b', tag.lower())
                        commentary_vocab.update(words)
    
    return {
        'tracks': tracks,
        'classes': classes,
        'categories': categories,
        'trainers': trainers,
        'going_conditions': going_conditions,
        'commentary_vocab': commentary_vocab
    }


def process_race_for_dataset(args) -> Optional[Dict]:
    """
    Process a single race to create dataset sample (for multiprocessing)
    
    Args:
        args: Tuple of (race_index, race, dog_lookup, encoders, max_dogs_per_race, max_history_length, max_commentary_length)
        
    Returns:
        Processed race data or None if invalid
    """
    try:
        race_index, race, dog_lookup, encoders, max_dogs_per_race, max_history_length, max_commentary_length = args
        
        # Check if race has minimum required dogs
        dogs_available = sum(1 for dog_id in race.dog_ids.values() if dog_id in dog_lookup)
        if dogs_available < 3:  # min_dogs_per_race
            return None
        
        # Process race features
        race_features = []
        
        # Distance (normalized)
        distance_normalized = (race.distance - 300) / 500 if race.distance else 0.0
        race_features.append(distance_normalized)
        
        # Weather features
        if race.rainfall_7d and len(race.rainfall_7d) >= 7:
            race_features.extend(race.rainfall_7d[:7])
        else:
            race_features.extend([0.0] * 7)
        
        race_features = np.array(race_features[:8], dtype=np.float32)
        
        # Encode categorical features
        track_id = encoders['track_encoder'].transform([race.track_name or 'Unknown'])[0]
        class_id = encoders['class_encoder'].transform([race.race_class or 'Unknown'])[0]
        category_id = encoders['category_encoder'].transform([race.category or 'Unknown'])[0]
        
        # Process dogs
        dog_features = []
        trainer_ids = []
        dog_ids = []
        win_labels = []
        market_odds = []
        history_features = []
        going_ids = []
        commentary_ids = []
        history_mask = []
        
        # Get race results for win labels
        race_positions = {}
        for trap_num, dog_id in race.dog_ids.items():
            if dog_id in dog_lookup:
                dog = dog_lookup[dog_id]
                race_datetime = datetime.combine(race.race_date, race.race_time)
                participation = dog.get_race_participation(race.race_id, race_datetime)
                if participation and participation.position is not None:
                    race_positions[trap_num] = participation.position
        
        # Process each trap position
        for trap_num in range(1, max_dogs_per_race + 1):
            if trap_num in race.dog_ids:
                dog_id = race.dog_ids[trap_num]
                if dog_id in dog_lookup:
                    dog = dog_lookup[dog_id]
                    
                    # Dog static features
                    dog_age = dog.age_at_race(race.race_date) if hasattr(dog, 'age_at_race') else 0.0
                    dog_weight = dog.weight if hasattr(dog, 'weight') and dog.weight else 30.0
                    dog_feat = [dog_age, dog_weight, 1.0]  # 1.0 indicates valid dog
                    
                    # Trainer
                    trainer_name = dog.trainer or 'Unknown'
                    trainer_id = encoders['trainer_encoder'].transform([trainer_name])[0]
                    
                    # Win label
                    win_label = 1.0 if race_positions.get(trap_num) == 1 else 0.0
                    
                    # Market odds
                    odds = race.odds.get(trap_num, 5.0) if race.odds else 5.0
                    
                    # Historical features
                    race_datetime = datetime.combine(race.race_date, race.race_time)
                    hist_participations = dog.get_last_n_races_before(race_datetime, max_history_length)
                    
                    hist_features = []
                    hist_going = []
                    hist_commentary = []
                    hist_mask_seq = []
                    
                    for i in range(max_history_length):
                        if i < len(hist_participations):
                            participation = hist_participations[i]
                            
                            # Historical race features
                            hist_time = participation.finish_time if participation.finish_time else 30.0
                            hist_pos = participation.position if participation.position else 4.0
                            hist_odds = participation.odds if participation.odds else 5.0
                            hist_features.append([hist_time, hist_pos, hist_odds])
                            
                            # Going condition
                            going_name = participation.going or 'Unknown'
                            going_id = encoders['going_encoder'].transform([going_name])[0]
                            hist_going.append(going_id)
                            
                            # Commentary
                            commentary_encoded = []
                            for j in range(max_commentary_length):
                                if j < len(participation.commentary_tags):
                                    tag = participation.commentary_tags[j].lower()
                                    words = re.findall(r'\b\w+\b', tag)
                                    if words:
                                        word = words[0]  # Take first word
                                        word_id = encoders['commentary_encoder'].get(word, encoders['commentary_encoder']['<UNK>'])
                                    else:
                                        word_id = encoders['commentary_encoder']['<PAD>']
                                else:
                                    word_id = encoders['commentary_encoder']['<PAD>']
                                commentary_encoded.append(word_id)
                            hist_commentary.append(commentary_encoded)
                            
                            hist_mask_seq.append(True)
                        else:
                            # Padding
                            hist_features.append([0.0, 0.0, 0.0])
                            hist_going.append(0)  # Padding ID
                            hist_commentary.append([0] * max_commentary_length)  # Padding IDs
                            hist_mask_seq.append(False)
                    
                    dog_features.append(dog_feat)
                    trainer_ids.append(trainer_id)
                    dog_ids.append(int(dog_id) if dog_id.isdigit() else 0)
                    win_labels.append(win_label)
                    market_odds.append(odds)
                    history_features.append(hist_features)
                    going_ids.append(hist_going)
                    commentary_ids.append(hist_commentary)
                    history_mask.append(hist_mask_seq)
                else:
                    # Invalid dog - padding
                    dog_features.append([0.0, 0.0, 0.0])
                    trainer_ids.append(0)
                    dog_ids.append(0)
                    win_labels.append(0.0)
                    market_odds.append(5.0)
                    history_features.append([[0.0, 0.0, 0.0]] * max_history_length)
                    going_ids.append([0] * max_history_length)
                    commentary_ids.append([[0] * max_commentary_length] * max_history_length)
                    history_mask.append([False] * max_history_length)
            else:
                # Empty trap - padding
                dog_features.append([0.0, 0.0, 0.0])
                trainer_ids.append(0)
                dog_ids.append(0)
                win_labels.append(0.0)
                market_odds.append(5.0)
                history_features.append([[0.0, 0.0, 0.0]] * max_history_length)
                going_ids.append([0] * max_history_length)
                commentary_ids.append([[0] * max_commentary_length] * max_history_length)
                history_mask.append([False] * max_history_length)
        
        # Create dog mask (which positions have valid dogs)
        dog_mask = []
        for trap_num in range(1, max_dogs_per_race + 1):
            has_valid_dog = (trap_num in race.dog_ids and 
                           race.dog_ids[trap_num] in dog_lookup)
            dog_mask.append(has_valid_dog)
        
        return {
            'race_features': np.array(race_features, dtype=np.float32),
            'track_ids': track_id,
            'class_ids': class_id,
            'category_ids': category_id,
            'dog_features': np.array(dog_features, dtype=np.float32),
            'trainer_ids': np.array(trainer_ids, dtype=np.int32),
            'dog_ids': np.array(dog_ids, dtype=np.int32),
            'dog_mask': np.array(dog_mask, dtype=bool),
            'win_labels': np.array(win_labels, dtype=np.float32),
            'market_odds': np.array(market_odds, dtype=np.float32),
            'history_features': np.array(history_features, dtype=np.float32),
            'going_ids': np.array(going_ids, dtype=np.int32),
            'commentary_ids': np.array(commentary_ids, dtype=np.int32),
            'history_mask': np.array(history_mask, dtype=bool)
        }
        
    except Exception as e:
        logger.warning(f"Error processing race {race_index}: {e}")
        return None


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
                 exclude_trial_races: bool = True,
                 max_races: Optional[int] = None,
                 encoders: Optional[Dict] = None,
                 vocab_sizes: Optional[Dict] = None):
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
            max_races: Optional limit on number of races to process (for testing)
            encoders: Pre-built encoders (if None, will build from data)
            vocab_sizes: Pre-computed vocabulary sizes (if None, will compute from encoders)
        """
        self.dog_lookup = dog_lookup
        self.track_lookup = track_lookup
        self.max_dogs_per_race = max_dogs_per_race
        self.max_history_length = max_history_length
        self.max_commentary_length = max_commentary_length
        self.min_dogs_per_race = min_dogs_per_race
        
        # Limit races if specified (useful for testing/debugging)
        if max_races is not None:
            races = races[:max_races]
            print(f"Limited to first {max_races} races for processing")
        
        # Filter valid races
        print(f"Filtering valid races from {len(races)} input races...")
        self.races = self._filter_valid_races(races, exclude_trial_races)
        
        # Use pre-built encoders if provided, otherwise build them
        if encoders is not None and vocab_sizes is not None:
            print("Using pre-built encoders...")
            self._load_encoders(encoders, vocab_sizes)
        else:
            print("Building vocabularies and encoders...")
            self._build_encoders()
        
        logger.info(f"Dataset initialized with {len(self.races)} valid races")
        
    def _filter_valid_races(self, races: List[Race], exclude_trial_races: bool) -> List[Race]:
        """Filter races that meet criteria for training"""
        valid_races = []
        
        for race in races:
            # Skip trial races if specified (they typically don't have odds)
            if exclude_trial_races and race.is_trial_race():
                continue
                
            # Check minimum number of dogs
            if len(race.dog_ids) < self.min_dogs_per_race:
                continue
                
            # Check if race has complete field (for betting races)
            if not exclude_trial_races and not race.has_complete_field():
                continue
            
            # CRITICAL: Ensure race has odds for all dogs (required for training)
            if not race.odds or len(race.odds) < self.min_dogs_per_race:
                continue
            
            # Verify ALL dogs in the race have valid odds (more comprehensive check)
            has_all_valid_odds = True
            valid_odds = []
            for trap_num in race.dog_ids.keys():
                if trap_num not in race.odds:
                    has_all_valid_odds = False
                    break  # Missing odds for this dog
                odds_value = race.odds[trap_num]
                if odds_value is None or odds_value <= 0:
                    has_all_valid_odds = False
                    break  # Invalid odds
                valid_odds.append(odds_value)
            
            if not has_all_valid_odds:
                continue  # Skip race if any dog has missing/invalid odds
                
            # CRITICAL: Check implied probability sum to ensure complete betting market
            # Sum should be > 1.0 (typically 1.05-1.20) due to bookmaker overround
            # If ≤ 1.0, it indicates incomplete market or missing runners
            implied_prob_sum = sum(1.0 / odds for odds in valid_odds)
            if implied_prob_sum <= 1.0:
                continue  # Skip race - incomplete betting market (sum of implied probabilities ≤ 1.0)
            
            # Ensure we have data for all dogs in the race
            dogs_available = sum(1 for dog_id in race.dog_ids.values() 
                               if dog_id in self.dog_lookup)
            
            if dogs_available < self.min_dogs_per_race:
                continue
                
            valid_races.append(race)
            
        return valid_races
    
    def _load_encoders(self, encoders: Dict, vocab_sizes: Dict):
        """Load pre-built encoders and vocabulary sizes"""
        
        # Load encoders
        self.track_encoder = encoders['track_encoder']
        self.class_encoder = encoders['class_encoder'] 
        self.category_encoder = encoders['category_encoder']
        self.trainer_encoder = encoders['trainer_encoder']
        self.going_encoder = encoders['going_encoder']
        
        # Load commentary vocabulary mapping (this is the key fix!)
        self.commentary_vocab = encoders['commentary_encoder']
        
        # Store vocabulary sizes
        self.vocab_sizes = vocab_sizes.copy()
        
        logger.info("Loaded pre-built encoders:")
        for name, size in vocab_sizes.items():
            logger.info(f"  - {name}: {size}")
        
        # Verify commentary_vocab was loaded correctly
        if hasattr(self, 'commentary_vocab') and self.commentary_vocab:
            logger.info(f"Commentary vocabulary loaded: {len(self.commentary_vocab)} words")
    
    def _build_encoders(self):
        """Build label encoders for categorical features using multithreading"""
        
        print(f"Building encoders from {len(self.races)} races using multithreading...")
        
        # Split races into batches for parallel processing
        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 cores max
        batch_size = max(1, len(self.races) // (num_workers * 4))  # Create more batches than workers
        
        race_batches = []
        for i in range(0, len(self.races), batch_size):
            batch = self.races[i:i + batch_size]
            race_batches.append((batch, self.dog_lookup))
        
        print(f"Processing {len(race_batches)} batches with {num_workers} workers...")
        
        # Initialize combined vocabulary sets
        all_tracks = set()
        all_classes = set()
        all_categories = set()
        all_trainers = set()
        all_going_conditions = set()
        all_commentary_vocab = set()
        
        # Process batches in parallel using ThreadPoolExecutor (better for I/O bound tasks)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_race_batch_for_vocab, batch) for batch in race_batches]
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing race batches"):
                try:
                    result = future.result()
                    
                    # Merge vocabulary sets
                    all_tracks.update(result['tracks'])
                    all_classes.update(result['classes'])
                    all_categories.update(result['categories'])
                    all_trainers.update(result['trainers'])
                    all_going_conditions.update(result['going_conditions'])
                    all_commentary_vocab.update(result['commentary_vocab'])
                    
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
        
        logger.info(f"Vocabulary sizes:")
        logger.info(f"  - Tracks: {len(all_tracks)}")
        logger.info(f"  - Classes: {len(all_classes)}")
        logger.info(f"  - Categories: {len(all_categories)}")
        logger.info(f"  - Trainers: {len(all_trainers)}")
        logger.info(f"  - Going conditions: {len(all_going_conditions)}")
        logger.info(f"  - Commentary vocabulary: {len(all_commentary_vocab)}")
        
        # Build encoders
        print("Building label encoders...")
        
        self.track_encoder = LabelEncoder()
        self.track_encoder.fit(list(all_tracks) + ['Unknown'])
        
        self.class_encoder = LabelEncoder()
        self.class_encoder.fit(list(all_classes) + ['Unknown'])
        
        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(list(all_categories) + ['Unknown'])
        
        self.trainer_encoder = LabelEncoder()
        self.trainer_encoder.fit(list(all_trainers) + ['Unknown'])
        
        self.going_encoder = LabelEncoder()
        self.going_encoder.fit(list(all_going_conditions) + ['Unknown'])
        
        # Build commentary vocabulary mapping
        commentary_vocab_list = ['<PAD>', '<UNK>'] + sorted(list(all_commentary_vocab))
        self.commentary_vocab = {word: idx for idx, word in enumerate(commentary_vocab_list)}
        
        # Store vocabulary sizes
        self.vocab_sizes = {
            'num_tracks': len(self.track_encoder.classes_),
            'num_classes': len(self.class_encoder.classes_),
            'num_categories': len(self.category_encoder.classes_),
            'num_trainers': len(self.trainer_encoder.classes_),
            'num_going_conditions': len(self.going_encoder.classes_),
            'commentary_vocab_size': len(commentary_vocab_list)
        }
        
        logger.info("✅ Encoders built successfully with multithreading")
    
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
        
        # Handle case where race should be skipped due to missing/invalid odds
        if dogs_data is None:
            # Try the next race instead of crashing (with safeguard to prevent infinite loops)
            next_idx = (idx + 1) % len(self.races)
            if next_idx != idx:  # Prevent infinite recursion
                return self.__getitem__(next_idx)
            else:
                # If we only have one race and it's invalid, create dummy data
                raise ValueError(f"Race {race.race_id} has invalid odds and is the only race available")
        
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
            
            # Market odds (ensure we have valid odds - should be guaranteed by race filtering)
            market_odds = race.odds.get(trap_num) if race.odds else None
            if market_odds is None or market_odds <= 0:
                # Skip this race entirely if any dog has missing/invalid odds
                return None
            dogs_data['market_odds'].append(float(market_odds))
        
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
            dogs_data['market_odds'].append(5.0)  # Default odds for padding
        
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


def build_encoders_on_full_dataset(all_races: List[Race], dog_lookup: Dict[str, Dog]) -> Tuple[Dict, Dict]:
    """
    Build encoders on the full dataset (training + validation combined) using multiprocessing
    This ensures consistent encoding across all data splits.
    
    Args:
        all_races: Complete list of races (train + val combined)
        dog_lookup: Dictionary of all dogs
        
    Returns:
        Tuple of (encoders_dict, vocab_sizes_dict)
    """
    from multiprocessing import Pool, cpu_count
    import math
    
    logger.info(f"Building encoders on full dataset ({len(all_races)} races) using multiprocessing...")
    
    # Determine optimal number of processes
    num_processes = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    batch_size = math.ceil(len(all_races) / num_processes)
    
    logger.info(f"Using {num_processes} processes with batch size {batch_size}")
    
    # Prepare batches for multiprocessing
    race_batches = []
    for i in range(0, len(all_races), batch_size):
        batch = all_races[i:i + batch_size]
        race_batches.append((batch, dog_lookup))
    
    # Process batches in parallel
    logger.info("Processing vocabulary extraction in parallel...")
    
    with Pool(processes=num_processes) as pool:
        # Use map to process all batches
        results = list(tqdm(
            pool.imap(process_race_batch_for_vocab, race_batches),
            total=len(race_batches),
            desc="Processing vocabulary batches",
            unit="batches"
        ))
    
    # Combine results from all processes
    logger.info("Combining vocabulary results...")
    tracks = set()
    classes = set()
    categories = set()
    trainers = set()
    going_conditions = set()
    commentary_vocab = set()
    
    for result in results:
        tracks.update(result['tracks'])
        classes.update(result['classes'])
        categories.update(result['categories'])
        trainers.update(result['trainers'])
        going_conditions.update(result['going_conditions'])
        commentary_vocab.update(result['commentary_vocab'])
    
    logger.info(f"Vocabulary sizes:")
    logger.info(f"  - Tracks: {len(tracks)}")
    logger.info(f"  - Classes: {len(classes)}")
    logger.info(f"  - Categories: {len(categories)}")
    logger.info(f"  - Trainers: {len(trainers)}")
    logger.info(f"  - Going conditions: {len(going_conditions)}")
    logger.info(f"  - Commentary vocabulary: {len(commentary_vocab)}")
    
    # Build encoders
    logger.info("Building label encoders...")
    
    track_encoder = LabelEncoder()
    track_encoder.fit(list(tracks) + ['Unknown'])
    
    class_encoder = LabelEncoder()
    class_encoder.fit(list(classes) + ['Unknown'])
    
    category_encoder = LabelEncoder()
    category_encoder.fit(list(categories) + ['Unknown'])
    
    trainer_encoder = LabelEncoder()
    trainer_encoder.fit(list(trainers) + ['Unknown'])
    
    going_encoder = LabelEncoder()
    going_encoder.fit(list(going_conditions) + ['Unknown'])
    
    # Build commentary vocabulary mapping
    commentary_vocab_list = ['<PAD>', '<UNK>'] + sorted(list(commentary_vocab))
    commentary_word_to_id = {word: idx for idx, word in enumerate(commentary_vocab_list)}
    
    # Store encoders
    encoders = {
        'track_encoder': track_encoder,
        'class_encoder': class_encoder,
        'category_encoder': category_encoder,
        'trainer_encoder': trainer_encoder,
        'going_encoder': going_encoder,
        'commentary_encoder': commentary_word_to_id
    }
    
    # Vocabulary sizes
    vocab_sizes = {
        'num_tracks': len(track_encoder.classes_),
        'num_classes': len(class_encoder.classes_),
        'num_categories': len(category_encoder.classes_),
        'num_trainers': len(trainer_encoder.classes_),
        'num_going_conditions': len(going_encoder.classes_),
        'commentary_vocab_size': len(commentary_vocab_list)
    }
    
    logger.info("Encoders built successfully with multiprocessing")
    
    return encoders, vocab_sizes


def load_data_from_buckets(dogs_enhanced_dir: str, races_dir: str, unified_dir: str) -> Tuple[Dict[str, Dog], List[Race]]:
    """
    Load dogs and races from bucket files
    
    Returns:
        Tuple of (dog_lookup, races_list)
    """
    logger.info("Loading dogs from enhanced directory...")
    
    # Get list of dog bucket files
    dog_bucket_files = [f for f in os.listdir(dogs_enhanced_dir) 
                       if f.startswith('dogs_bucket_') and f.endswith('.pkl')]
    
    # Load all dogs with progress bar
    dog_lookup = {}
    for bucket_file in tqdm(dog_bucket_files, desc="Loading dog buckets", unit="buckets"):
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
    
    # Load all races with progress bar
    races = []
    loaded_buckets = {}
    
    for race_key, race_info in tqdm(race_index.items(), desc="Loading races", unit="races"):
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

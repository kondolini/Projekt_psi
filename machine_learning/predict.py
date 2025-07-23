"""
Prediction script for trained Greyhound Racing Model
"""

import os
import sys
import argparse
import torch
import pickle
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from machine_learning.data_processor import RaceDataProcessor, create_dataset
from machine_learning.model import GreyhoundRacingModel, collate_race_batch


def load_trained_model(model_path: str, device: str = 'auto') -> tuple:
    """
    Load a trained model and data processor
    
    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        model: Loaded GreyhoundRacingModel
        processor: Loaded RaceDataProcessor
        metadata: Model training metadata
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load model
    model, _, _, metadata = GreyhoundRacingModel.load_model(model_path, device)
    model.eval()  # Set to evaluation mode
    
    # Load data processor (should be in same directory)
    processor_path = os.path.join(os.path.dirname(model_path), 'data_processor.pkl')
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Data processor not found: {processor_path}")
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    print("Model and processor loaded successfully!")
    return model, processor, metadata


def predict_single_race(
    model: GreyhoundRacingModel, 
    processor: RaceDataProcessor,
    race: Race, 
    dogs: Dict[str, Dog]
) -> Dict[int, float]:
    """
    Predict win probabilities for a single race
    
    Args:
        model: Trained model
        processor: Data processor
        race: Race object to predict
        dogs: Dictionary of all dogs
        
    Returns:
        Dictionary mapping trap number to win probability
    """
    device = model.get_device()
    
    # Process race into model format
    race_data = processor.process_race(race, dogs)
    
    # Convert to batch format
    batch = collate_race_batch([race_data])
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(batch)  # [1, 6]
    
    # Convert to dictionary
    win_probs = {}
    for trap in range(1, 7):  # Traps 1-6
        trap_idx = trap - 1
        if trap in race.dog_ids:
            win_probs[trap] = float(predictions[0, trap_idx].cpu())
        else:
            win_probs[trap] = 0.0
    
    return win_probs


def predict_race_batch(
    model: GreyhoundRacingModel,
    processor: RaceDataProcessor, 
    races: List[Race],
    dogs: Dict[str, Dog],
    batch_size: int = 32
) -> List[Dict[int, float]]:
    """
    Predict win probabilities for multiple races efficiently
    
    Args:
        model: Trained model
        processor: Data processor
        races: List of Race objects
        dogs: Dictionary of all dogs
        batch_size: Batch size for processing
        
    Returns:
        List of dictionaries mapping trap number to win probability
    """
    device = model.get_device()
    model.eval()
    
    all_predictions = []
    
    # Process races in batches
    for i in range(0, len(races), batch_size):
        batch_races = races[i:i + batch_size]
        
        # Process races
        race_data_list = []
        for race in batch_races:
            try:
                race_data = processor.process_race(race, dogs)
                race_data_list.append(race_data)
            except Exception as e:
                print(f"Error processing race {race.race_id}: {e}")
                # Add dummy data for failed races
                dummy_data = {
                    "race_features": {},
                    "dog_features": [{}] * 6,
                    "targets": [0] * 6,
                    "race_id": race.race_id,
                    "meeting_id": race.meeting_id,
                    "race_datetime": race.get_race_datetime()
                }
                race_data_list.append(dummy_data)
        
        if not race_data_list:
            continue
            
        # Convert to batch tensors
        batch = collate_race_batch(race_data_list)
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(batch)  # [batch_size, 6]
        
        # Convert predictions to list of dictionaries
        for j, race in enumerate(batch_races):
            win_probs = {}
            for trap in range(1, 7):
                trap_idx = trap - 1
                if trap in race.dog_ids:
                    win_probs[trap] = float(predictions[j, trap_idx].cpu())
                else:
                    win_probs[trap] = 0.0
            all_predictions.append(win_probs)
    
    return all_predictions


def load_test_data(data_dir: str, test_split_date: str = "2023-01-01"):
    """Load test data for prediction"""
    print("Loading dogs...")
    dogs = {}
    dogs_dir = os.path.join(data_dir, "dogs_enhanced")
    
    if not os.path.exists(dogs_dir):
        raise FileNotFoundError(f"Dogs directory not found: {dogs_dir}")
    
    for fname in os.listdir(dogs_dir):
        if fname.endswith('.pkl'):
            try:
                with open(os.path.join(dogs_dir, fname), 'rb') as f:
                    bucket = pickle.load(f)
                    for dog_id, dog_obj in bucket.items():
                        if isinstance(dog_obj, Dog):
                            dogs[dog_id] = dog_obj
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    print(f"Loaded {len(dogs)} dogs")
    
    # Load pre-built race objects from buckets
    print("Loading pre-built race objects...")
    races_dir = os.path.join(data_dir, "races")
    all_races = []
    
    if not os.path.exists(races_dir):
        raise FileNotFoundError(f"Races directory not found: {races_dir}")
    
    # Load all race buckets
    for fname in os.listdir(races_dir):
        if fname.startswith('races_bucket_') and fname.endswith('.pkl'):
            try:
                bucket_path = os.path.join(races_dir, fname)
                with open(bucket_path, 'rb') as f:
                    races_bucket = pickle.load(f)
                
                for storage_key, race in races_bucket.items():
                    if isinstance(race, Race):
                        all_races.append(race)
                        
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    print(f"Loaded {len(all_races)} race objects from buckets")
    
    # Sort races chronologically
    all_races.sort(key=lambda r: r.get_race_datetime())
    
    # Split by date for test data
    split_date = datetime.strptime(test_split_date, "%Y-%m-%d").date()
    test_races = [r for r in all_races if r.race_date >= split_date]
    
    print(f"Test races: {len(test_races)}")
    return dogs, test_races


def main():
    parser = argparse.ArgumentParser(description='Predict with trained Greyhound Racing Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--test_split', type=str, default='2023-01-01', help='Test split date')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--num_races', type=int, default=10, help='Number of test races to predict (0 = all)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    try:
        # Load model and processor
        model, processor, metadata = load_trained_model(args.model_path, args.device)
        
        print(f"\nModel Training Info:")
        if metadata['epoch'] is not None:
            print(f"  Trained for {metadata['epoch'] + 1} epochs")
        if metadata['metrics']:
            for key, value in metadata['metrics'].items():
                print(f"  {key}: {value:.4f}")
        
        # Load test data
        print(f"\nLoading test data...")
        dogs, test_races = load_test_data(args.data_dir, args.test_split)
        
        # Limit number of races if specified
        if args.num_races > 0:
            test_races = test_races[:args.num_races]
            print(f"Using first {len(test_races)} test races")
        
        # Make predictions
        print(f"\nMaking predictions for {len(test_races)} races...")
        predictions = predict_race_batch(model, processor, test_races, dogs, args.batch_size)
        
        # Display results for first few races
        print(f"\nPrediction Results (first 5 races):")
        for i, (race, pred) in enumerate(zip(test_races[:5], predictions[:5])):
            print(f"\nRace {i+1}: {race.race_id} on {race.race_date}")
            print(f"  Track: {race.track_name}, Distance: {race.distance}m")
            print("  Win Probabilities:")
            
            for trap in sorted(pred.keys()):
                if trap in race.dog_ids:
                    dog_id = race.dog_ids[trap]
                    prob = pred[trap]
                    implied_odds = 1 / prob if prob > 0 else float('inf')
                    print(f"    Trap {trap} (Dog {dog_id}): {prob:.3f} (odds: {implied_odds:.2f})")
        
        print(f"\nCompleted predictions for {len(predictions)} races!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

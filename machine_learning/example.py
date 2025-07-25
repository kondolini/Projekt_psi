#!/usr/bin/env python3
"""
Example script showing how to train the greyhound racing model

This script demonstrates the basic usage of the training system.
For production use, use the full train.py script with command line arguments.
"""

import os
import sys
import torch
from datetime import date

# Add parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from machine_learning.dataset import load_data_from_buckets, create_train_val_split, GreyhoundDataset, create_dataloaders
from machine_learning.model import GreyhoundRacingModel
from machine_learning.trainer import create_trainer
from machine_learning.utils import setup_logging, print_gpu_info, print_model_info, print_data_summary


def main():
    """Simple training example"""
    
    # Setup logging
    setup_logging('INFO')
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_gpu_info()
    
    print("🐕 Greyhound Racing Model Training Example")
    print("=" * 50)
    
    # Data paths (adjust these to your actual data locations)
    data_dir = 'data'
    dogs_enhanced_dir = os.path.join(data_dir, 'dogs_enhanced')
    races_dir = os.path.join(data_dir, 'races')
    unified_dir = os.path.join(data_dir, 'unified')
    
    # Check if data directories exist
    for dir_path, name in [(dogs_enhanced_dir, 'dogs_enhanced'), 
                          (races_dir, 'races'), 
                          (unified_dir, 'unified')]:
        if not os.path.exists(dir_path):
            print(f"❌ Data directory not found: {dir_path}")
            print(f"Please ensure you have run the data construction scripts first.")
            return
    
    print("📊 Loading data...")
    
    # Load data
    dog_lookup, races = load_data_from_buckets(dogs_enhanced_dir, races_dir, unified_dir)
    
    print(f"Loaded {len(dog_lookup)} dogs and {len(races)} races")
    
    # Create train/validation split (use 2023 as validation year)
    val_start_date = date(2023, 1, 1)
    train_races, val_races = create_train_val_split(races, val_start_date)
    
    # Create datasets
    track_lookup = {}  # Empty for this example
    
    train_dataset = GreyhoundDataset(
        races=train_races[:1000],  # Limit to first 1000 races for quick example
        dog_lookup=dog_lookup,
        track_lookup=track_lookup,
        max_dogs_per_race=8,
        max_history_length=10,
        max_commentary_length=5,
        exclude_trial_races=True
    )
    
    val_dataset = GreyhoundDataset(
        races=val_races[:200],  # Limit to first 200 validation races
        dog_lookup=dog_lookup,
        track_lookup=track_lookup,
        max_dogs_per_race=8,
        max_history_length=10,
        max_commentary_length=5,
        exclude_trial_races=True
    )
    
    # Use same encoders for validation
    val_dataset.track_encoder = train_dataset.track_encoder
    val_dataset.class_encoder = train_dataset.class_encoder
    val_dataset.category_encoder = train_dataset.category_encoder
    val_dataset.trainer_encoder = train_dataset.trainer_encoder
    val_dataset.going_encoder = train_dataset.going_encoder
    val_dataset.commentary_vocab = train_dataset.commentary_vocab
    val_dataset.vocab_sizes = train_dataset.vocab_sizes
    
    print_data_summary(train_dataset, val_dataset)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, 
        val_dataset, 
        batch_size=16,  # Small batch size for example
        num_workers=0
    )
    
    print("🧠 Creating model...")
    
    # Create model
    model = GreyhoundRacingModel(
        num_tracks=train_dataset.vocab_sizes['num_tracks'],
        num_classes=train_dataset.vocab_sizes['num_classes'],
        num_categories=train_dataset.vocab_sizes['num_categories'],
        num_trainers=train_dataset.vocab_sizes['num_trainers'],
        num_going_conditions=train_dataset.vocab_sizes['num_going_conditions'],
        commentary_vocab_size=train_dataset.vocab_sizes['commentary_vocab_size'],
        embedding_dim=32,
        hidden_dim=64,
        rnn_hidden_dim=32,
        max_history_length=10,
        max_commentary_length=5,
        dropout_rate=0.2,
        max_dogs_per_race=8
    )
    
    print_model_info(model)
    
    print("🚀 Starting training...")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        alpha=1.0,              # Confidence multiplier
        temperature=1.0,        # Initial temperature
        commission=0.05,        # 5% commission
        profit_weight=0.7,      # Focus more on profit
        accuracy_weight=0.3,    # But also consider accuracy
        device=device
    )
    
    # Train for just a few epochs as example
    trainer.train(
        num_epochs=5,
        temperature_annealing=True
    )
    
    print("✅ Training completed!")
    print("\nFor full training with all features, use:")
    print("python machine_learning/train.py --epochs 100 --batch_size 32")
    print("\nFor resuming training:")
    print("python machine_learning/train.py --resume checkpoints/interrupted_checkpoint.pth")
    

if __name__ == '__main__':
    main()

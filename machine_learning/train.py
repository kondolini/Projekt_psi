#!/usr/bin/env python3
"""
Greyhound Racing Model Training Script

This script trains a neural network model for greyhound racing predictions
with a focus on betting profitability.

Features:
- Graceful stopping and resuming from checkpoints
- GPU support with automatic detection
- Progress tracking and comprehensive logging
- PnL monitoring and betting simulation
- Temperature annealing for improved convergence
- Data quality checks and validation
- Configurable hyperparameters via command line

Usage:
    python train.py --epochs 100 --batch_size 32 --learning_rate 0.001
    python train.py --resume checkpoints/interrupted_checkpoint.pth
    python train.py --from_checkpoint checkpoints/best_model.pth --epochs 50
"""

import os
import sys
import argparse
import logging
import torch
from datetime import date, datetime
from typing import Optional, Tuple

# Add parent directory for imports - make it more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import our modules
from machine_learning.model import GreyhoundRacingModel
from machine_learning.dataset import (
    load_data_from_buckets, 
    create_train_val_split, 
    GreyhoundDataset, 
    create_dataloaders,
    build_encoders_on_full_dataset
)
from machine_learning.trainer import create_trainer
from machine_learning.cache_manager import CacheManager
from machine_learning.utils import (
    setup_logging, 
    print_gpu_info, 
    print_model_info, 
    print_data_summary,
    set_random_seeds, 
    create_run_name, 
    save_run_config,
    save_encoders,
    create_directory_structure,
    cleanup_old_checkpoints
)

logger = logging.getLogger(__name__)


def resolve_data_paths(base_data_dir: str) -> Tuple[str, str, str]:
    """
    Resolve data directory paths relative to the script location if they're not absolute
    
    Args:
        base_data_dir: Base data directory (can be relative or absolute)
        
    Returns:
        Tuple of (dogs_enhanced_dir, races_dir, unified_dir)
    """
    # If not absolute, make it relative to the project root (parent of machine_learning)
    if not os.path.isabs(base_data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_data_dir = os.path.join(project_root, base_data_dir)
    
    dogs_enhanced_dir = os.path.join(base_data_dir, 'dogs_enhanced')
    races_dir = os.path.join(base_data_dir, 'races')
    unified_dir = os.path.join(base_data_dir, 'unified')
    
    return dogs_enhanced_dir, races_dir, unified_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Greyhound Racing Prediction Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory containing data')
    parser.add_argument('--dogs_enhanced_dir', type=str, default=None,
                       help='Directory containing enhanced dog buckets')
    parser.add_argument('--races_dir', type=str, default=None,
                       help='Directory containing race buckets')
    parser.add_argument('--unified_dir', type=str, default=None,
                       help='Directory containing unified indices')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    
    # Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=32,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--rnn_hidden_dim', type=int, default=32,
                       help='RNN hidden dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--max_history_length', type=int, default=10,
                       help='Maximum number of historical races per dog')
    parser.add_argument('--max_commentary_length', type=int, default=5,
                       help='Maximum number of commentary tags per race')
    parser.add_argument('--max_dogs_per_race', type=int, default=8,
                       help='Maximum number of dogs per race')
    
    # Loss function parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Confidence multiplier for betting')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Initial temperature for soft betting selection')
    parser.add_argument('--commission', type=float, default=0.05,
                       help='Betting commission rate')
    parser.add_argument('--profit_weight', type=float, default=0.7,
                       help='Weight for profit loss component')
    parser.add_argument('--accuracy_weight', type=float, default=0.3,
                       help='Weight for accuracy loss component')
    parser.add_argument('--min_expected_profit', type=float, default=0.0,
                       help='Minimum expected profit threshold for betting')
    
    # Data split arguments
    parser.add_argument('--val_start_date', type=str, default='2023-01-01',
                       help='Start date for validation set (YYYY-MM-DD)')
    parser.add_argument('--val_end_date', type=str, default=None,
                       help='End date for validation set (YYYY-MM-DD)')
    parser.add_argument('--exclude_trial_races', action='store_true',
                       help='Exclude races without odds (trial races)')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint file')
    parser.add_argument('--from_checkpoint', type=str, default=None,
                       help='Load model from checkpoint but start fresh training')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate on validation set every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience (epochs)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='machine_learning/outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name for this training run')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Optimization flags
    parser.add_argument('--no_temperature_annealing', action='store_true',
                       help='Disable temperature annealing during training')
    parser.add_argument('--cleanup_checkpoints', action='store_true',
                       help='Clean up old checkpoint files')
    parser.add_argument('--max_races', type=int, default=None,
                       help='Limit number of races for testing (default: use all)')
    
    # Cache management
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Directory for caching processed data and encoders')
    parser.add_argument('--rebuild_cache', action='store_true',
                       help='Force rebuild of cached data and encoders')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear all cached data before training')
    parser.add_argument('--cache_stats', action='store_true',
                       help='Show cache statistics and exit')
    
    return parser.parse_args()


def setup_directories_and_logging(args):
    """Setup output directories and logging"""
    
    # Create run name if not provided
    if args.run_name is None:
        args.run_name = create_run_name()
    
    # Create output directory structure
    run_dir = os.path.join(args.output_dir, args.run_name)
    directories = create_directory_structure(run_dir)
    
    # Setup logging
    log_file = os.path.join(directories['logs'], 'training.log')
    setup_logging(args.log_level, log_file)
    
    logger.info(f"Starting training run: {args.run_name}")
    logger.info(f"Output directory: {run_dir}")
    
    return directories


def setup_device(args):
    """Setup and validate device"""
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Print GPU info if using CUDA
    if device.type == 'cuda':
        gpu_available = print_gpu_info()
        if not gpu_available:
            logger.warning("GPU requested but not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        logger.info("Using CPU for training")
    
    return device


def load_and_prepare_data(args):
    """Load and prepare data for training with caching support"""
    
    # Initialize cache manager
    cache_manager = CacheManager(args.cache_dir)
    
    if args.cache_stats:
        cache_manager.print_cache_stats()
        return None, None, None, None
    
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("Cache cleared")
    
    logger.info("Loading data from buckets...")
    
    # Resolve data directory paths (handles relative/absolute paths properly)
    dogs_enhanced_dir, races_dir, unified_dir = resolve_data_paths(args.data_dir)
    
    # Allow override of individual directories if specified
    if args.dogs_enhanced_dir:
        dogs_enhanced_dir = args.dogs_enhanced_dir
    if args.races_dir:
        races_dir = args.races_dir  
    if args.unified_dir:
        unified_dir = args.unified_dir
    
    # Validate directories exist
    for dir_path, name in [(dogs_enhanced_dir, 'dogs_enhanced'), 
                          (races_dir, 'races'), 
                          (unified_dir, 'unified')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path} ({name})")
    
    logger.info(f"Data directories:")
    logger.info(f"  - Dogs: {dogs_enhanced_dir}")
    logger.info(f"  - Races: {races_dir}")
    logger.info(f"  - Unified: {unified_dir}")
    
    # Configuration for cache key generation
    data_dirs = {
        'dogs_enhanced': dogs_enhanced_dir,
        'races': races_dir,
        'unified': unified_dir
    }
    
    # Configuration that affects data processing
    config = {
        'max_races': args.max_races,
        'exclude_trial_races': args.exclude_trial_races,
        'val_start_date': args.val_start_date,
        'val_end_date': args.val_end_date,
        'max_dogs_per_race': args.max_dogs_per_race,
        'max_history_length': args.max_history_length,
        'max_commentary_length': args.max_commentary_length
    }
    
    # Try to load cached encoders first (needed for dataset creation)
    encoders = None
    vocab_sizes = None
    
    if not args.rebuild_cache:
        cached_encoders = cache_manager.get_cached_encoders(data_dirs, config)
        if cached_encoders is not None:
            encoders, vocab_sizes = cached_encoders
    
    # Load raw data (always need this)
    dog_lookup, races = load_data_from_buckets(dogs_enhanced_dir, races_dir, unified_dir)
    
    # If no cached encoders, build them on the full dataset
    if encoders is None:
        logger.info("Building encoders on full dataset...")
        encoders, vocab_sizes = build_encoders_on_full_dataset(races, dog_lookup)
        
        # Cache the encoders
        cache_manager.save_encoders(encoders, vocab_sizes, data_dirs, config)
    
    # Create train/validation split
    logger.info("Creating train/validation split...")
    val_start_date = datetime.strptime(args.val_start_date, '%Y-%m-%d').date()
    val_end_date = None
    if args.val_end_date:
        val_end_date = datetime.strptime(args.val_end_date, '%Y-%m-%d').date()
    
    train_races, val_races = create_train_val_split(races, val_start_date, val_end_date)
    
    logger.info(f"Train/Val split: {len(train_races)} train, {len(val_races)} validation races")
    
    # Try to load cached datasets
    cached_data = None
    if not args.rebuild_cache:
        cached_data = cache_manager.get_cached_datasets(data_dirs, config)
    
    if cached_data is not None:
        train_dataset, val_dataset, metadata = cached_data
        logger.info(f"Loaded cached datasets with {metadata['train_size']} train and {metadata['val_size']} validation samples")
    else:
        logger.info("Creating datasets...")
        
        # We'll use empty track_lookup for now - could be enhanced later
        track_lookup = {}
        
        # Create datasets with pre-built encoders
        train_dataset = GreyhoundDataset(
            races=train_races,
            dog_lookup=dog_lookup,
            track_lookup=track_lookup,
            max_dogs_per_race=args.max_dogs_per_race,
            max_history_length=args.max_history_length,
            max_commentary_length=args.max_commentary_length,
            exclude_trial_races=args.exclude_trial_races,
            max_races=args.max_races,
            encoders=encoders,
            vocab_sizes=vocab_sizes
        )
        
        val_dataset = GreyhoundDataset(
            races=val_races,
            dog_lookup=dog_lookup,
            track_lookup=track_lookup,
            max_dogs_per_race=args.max_dogs_per_race,
            max_history_length=args.max_history_length,
            max_commentary_length=args.max_commentary_length,
            exclude_trial_races=args.exclude_trial_races,
            encoders=encoders,
            vocab_sizes=vocab_sizes
        )
        
        # Cache the processed datasets
        metadata = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_dogs': len(dog_lookup),
            'num_races': len(races)
        }
        
        cache_manager.save_datasets(train_dataset, val_dataset, data_dirs, config, metadata)
    
    # Print data summary
    print_data_summary(train_dataset, val_dataset)
    
    return train_dataset, val_dataset, vocab_sizes, dog_lookup


def create_model(args, vocab_sizes):
    """Create and initialize model"""
    
    logger.info("Creating model...")
    
    model = GreyhoundRacingModel(
        num_tracks=vocab_sizes['num_tracks'],
        num_classes=vocab_sizes['num_classes'],
        num_categories=vocab_sizes['num_categories'],
        num_trainers=vocab_sizes['num_trainers'],
        num_going_conditions=vocab_sizes['num_going_conditions'],
        commentary_vocab_size=vocab_sizes['commentary_vocab_size'],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        max_history_length=args.max_history_length,
        max_commentary_length=args.max_commentary_length,
        dropout_rate=args.dropout_rate,
        max_dogs_per_race=args.max_dogs_per_race
    )
    
    # Count and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    print(f"🏗️  MODEL INFO:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: {total_params * 4 / 1024 / 1024:.1f} MB (32-bit)")
    
    # Print detailed model information
    print_model_info(model)
    
    return model


def load_from_checkpoint_if_specified(args, model, device):
    """Load model from checkpoint if specified"""
    
    if args.from_checkpoint and os.path.exists(args.from_checkpoint):
        logger.info(f"Loading model from checkpoint: {args.from_checkpoint}")
        
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Model loaded successfully")
    
    return model


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories and logging
    directories = setup_directories_and_logging(args)
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Setup device
    device = setup_device(args)
    
    try:
        # Load and prepare data
        result = load_and_prepare_data(args)
        
        # Handle cache stats case
        if result[0] is None:  # cache_stats was requested
            return
        
        train_dataset, val_dataset, vocab_sizes, dog_lookup = result
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            train_dataset, 
            val_dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create model
        model = create_model(args, vocab_sizes)
        
        # Load from checkpoint if specified
        model = load_from_checkpoint_if_specified(args, model, device)
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            alpha=args.alpha,
            temperature=args.temperature,
            commission=args.commission,
            profit_weight=args.profit_weight,
            accuracy_weight=args.accuracy_weight,
            device=device
        )
        
        # Update trainer settings
        trainer.checkpoint_dir = directories['checkpoints']
        trainer.log_dir = directories['logs']
        trainer.save_every = args.save_every
        trainer.eval_every = args.eval_every
        trainer.early_stopping.patience = args.early_stopping_patience
        
        # Save run configuration
        config = vars(args).copy()
        config['vocab_sizes'] = train_dataset.vocab_sizes
        config['device'] = str(device)
        config['model_info'] = model.get_model_info()
        save_run_config(config, directories['outputs'])
        
        # Clean up old checkpoints if requested
        if args.cleanup_checkpoints:
            cleanup_old_checkpoints(directories['checkpoints'])
        
        # Start training
        logger.info(f"Starting training for {args.epochs} epochs")
        logger.info(f"Loss function - Alpha: {args.alpha}, Temperature: {args.temperature}, Commission: {args.commission}")
        logger.info(f"Early stopping patience: {args.early_stopping_patience} epochs")
        
        training_history = trainer.train(
            num_epochs=args.epochs,
            resume_from_checkpoint=args.resume,
            temperature_annealing=not args.no_temperature_annealing
        )
        
        # Plot training history
        plot_path = os.path.join(directories['plots'], 'training_history.png')
        trainer.plot_training_history(plot_path)
        
        # Final summary
        logger.info("🎉 Training completed successfully!")
        
        if training_history['val_ppb']:
            best_val_ppb = max(training_history['val_ppb'])
            final_val_ppb = training_history['val_ppb'][-1]
            logger.info(f"Best validation PPB: ${best_val_ppb:.4f}")
            logger.info(f"Final validation PPB: ${final_val_ppb:.4f}")
        
        logger.info(f"Checkpoints saved in: {directories['checkpoints']}")
        logger.info(f"Logs saved in: {directories['logs']}")
        logger.info("Encoders cached for reuse")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

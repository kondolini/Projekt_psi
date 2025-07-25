"""
Greyhound Racing Prediction Model

A comprehensive machine learning system for predicting greyhound racing outcomes
with a focus on betting profitability.

Components:
- model.py: Neural network architecture
- dataset.py: Data loading and preprocessing
- loss.py: Differentiable betting loss function
- trainer.py: Training loop with checkpointing
- train.py: Main training script
- utils.py: Utility functions
"""

from .model import GreyhoundRacingModel
from .dataset import GreyhoundDataset, load_data_from_buckets, create_train_val_split
from .loss import GreyhoundBettingLoss, hard_betting_evaluation
from .trainer import GreyhoundTrainer, create_trainer
from .utils import (
    setup_logging, 
    print_gpu_info, 
    print_model_info,
    save_encoders,
    load_encoders
)

__version__ = "1.0.0"
__author__ = "Greyhound Racing Prediction Team"

__all__ = [
    'GreyhoundRacingModel',
    'GreyhoundDataset', 
    'GreyhoundBettingLoss',
    'GreyhoundTrainer',
    'load_data_from_buckets',
    'create_train_val_split',
    'create_trainer',
    'hard_betting_evaluation',
    'setup_logging',
    'print_gpu_info',
    'print_model_info',
    'save_encoders',
    'load_encoders'
]
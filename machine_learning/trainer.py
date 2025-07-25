import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, date
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from .model import GreyhoundRacingModel
from .loss import GreyhoundBettingLoss, hard_betting_evaluation
from .dataset import GreyhoundDataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MetricsTracker:
    """Track and compute rolling averages of training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, metrics_dict: Dict[str, float]):
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
            
    def get_averages(self) -> Dict[str, float]:
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def reset(self):
        self.metrics.clear()


class GreyhoundTrainer:
    """
    Comprehensive trainer for greyhound racing model with:
    - Graceful stopping and resuming
    - Checkpoint saving/loading
    - Progress tracking
    - PnL monitoring
    - GPU support
    - Temperature annealing
    """
    
    def __init__(self,
                 model: GreyhoundRacingModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: GreyhoundBettingLoss,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs",
                 save_every: int = 5,
                 eval_every: int = 1,
                 early_stopping_patience: int = 20):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        # Mixed precision training for better GPU utilization
        # Temporarily disable mixed precision to ensure GPU utilization
        self.scaler = None
        self.use_amp = False  # Disable mixed precision for now to debug GPU usage
        
        # Directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training settings
        self.save_every = save_every
        self.eval_every = eval_every
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')  # Maximize profit per bet
        
        # State tracking
        self.current_epoch = 0
        self.best_val_ppb = -float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppb': [],
            'val_ppb': [],
            'train_hit_rate': [],
            'val_hit_rate': [],
            'train_betting_freq': [],
            'val_betting_freq': []
        }
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Temperature annealing schedule
        self.initial_temperature = loss_fn.temperature
        self.final_temperature = 0.1
        
        logger.info(f"Trainer initialized. Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # GPU debugging info
        if device.type == 'cuda':
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            logger.info(f"Mixed Precision Training: {self.use_amp}")
            
            # Force GPU memory allocation with larger test
            logger.info("Forcing GPU memory allocation...")
            test_tensor = torch.randn(5000, 5000).to(device)  # Much larger allocation
            test_result = torch.mm(test_tensor, test_tensor.T)  # Force computation
            logger.info(f"GPU Memory after forced allocation: {torch.cuda.memory_allocated() / 1024**3:.3f}GB")
            del test_tensor, test_result
            torch.cuda.empty_cache()
            
            # Ensure model is actually on GPU
            logger.info("Verifying model on GPU...")
            for name, param in self.model.named_parameters():
                if not param.is_cuda:
                    logger.warning(f"Parameter {name} is not on GPU!")
                break  # Just check first parameter
            else:
                logger.info("✅ Model confirmed on GPU")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device and verify
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Debug: Check if data is actually on GPU (only first batch)
            if batch_idx == 0 and self.device.type == 'cuda':
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if not value.is_cuda:
                            logger.warning(f"Batch data '{key}' is not on GPU!")
                        else:
                            logger.info(f"✅ Batch data '{key}' confirmed on GPU")
                            break  # Just check one tensor
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Regular forward pass (mixed precision disabled for debugging)
            model_probs = self.model(batch)
            loss_dict = self.loss_fn(
                model_probs=model_probs,
                market_odds=batch['market_odds'],
                actual_winners=batch['win_labels'],
                dog_mask=batch['dog_mask']
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Collect metrics
            batch_metrics = {
                'total_loss': loss_dict['total_loss'].item(),
                'profit_loss': loss_dict['profit_loss'].item(),
                'accuracy_loss': loss_dict['accuracy_loss'].item(),
                'hit_rate': loss_dict['hit_rate'].mean().item(),
                'expected_ppb': loss_dict['expected_profit_per_bet'].mean().item(),
                'actual_ppb': loss_dict['actual_profit_per_bet'].mean().item(),
                'betting_frequency': loss_dict['betting_frequency'].mean().item()
            }
            
            for key, value in batch_metrics.items():
                epoch_metrics[key].append(value)
            
            self.train_metrics.update(batch_metrics)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_metrics = self.train_metrics.get_averages()
                
                # Get GPU memory info
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_utilization = f"{gpu_memory_allocated:.1f}/{gpu_memory_total:.1f}GB"
                else:
                    gpu_utilization = "N/A"
                
                progress_bar.set_postfix({
                    'Loss': f"{avg_metrics.get('total_loss', 0):.4f}",
                    'PPB': f"{avg_metrics.get('actual_ppb', 0):.4f}",
                    'Hit%': f"{avg_metrics.get('hit_rate', 0)*100:.1f}",
                    'Bet%': f"{avg_metrics.get('betting_frequency', 0)*100:.1f}",
                    'GPU': gpu_utilization
                })
        
        # Calculate epoch averages
        epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return epoch_avg
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        all_profits = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                model_probs = self.model(batch)
                
                # Soft loss (differentiable)
                loss_dict = self.loss_fn(
                    model_probs=model_probs,
                    market_odds=batch['market_odds'],
                    actual_winners=batch['win_labels'],
                    dog_mask=batch['dog_mask']
                )
                
                # Hard evaluation (actual betting simulation)
                hard_eval = hard_betting_evaluation(
                    model_probs=model_probs,
                    market_odds=batch['market_odds'],
                    actual_winners=batch['win_labels'],
                    dog_mask=batch['dog_mask'],
                    alpha=self.loss_fn.alpha,
                    commission=self.loss_fn.commission,
                    min_expected_profit=self.loss_fn.min_expected_profit
                )
                
                # Collect metrics
                batch_metrics = {
                    'total_loss': loss_dict['total_loss'].item(),
                    'profit_loss': loss_dict['profit_loss'].item(),
                    'accuracy_loss': loss_dict['accuracy_loss'].item(),
                    'soft_hit_rate': loss_dict['hit_rate'].mean().item(),
                    'soft_ppb': loss_dict['actual_profit_per_bet'].mean().item(),
                    'soft_betting_freq': loss_dict['betting_frequency'].mean().item(),
                    'hard_hit_rate': hard_eval['hit_rate'],
                    'hard_ppb': hard_eval['profit_per_bet'],
                    'hard_betting_freq': hard_eval['betting_frequency'],
                    'total_profit': hard_eval['total_profit']
                }
                
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)
                
                # Collect individual profits for PnL tracking
                all_profits.extend(hard_eval['individual_profits'])
        
        # Calculate epoch averages
        epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
        epoch_avg['cumulative_profit'] = sum(all_profits)
        epoch_avg['profit_std'] = np.std(all_profits) if all_profits else 0.0
        
        return epoch_avg
    
    def train(self, 
              num_epochs: int,
              resume_from_checkpoint: Optional[str] = None,
              temperature_annealing: bool = True) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            temperature_annealing: Whether to anneal temperature during training
            
        Returns:
            Training history dictionary
        """
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            self.load_checkpoint(resume_from_checkpoint)
            logger.info(f"Resumed training from epoch {self.current_epoch}")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Initial temperature: {self.loss_fn.temperature}")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Temperature annealing
                if temperature_annealing and num_epochs > 1:
                    progress = epoch / (num_epochs - 1)
                    current_temp = self.initial_temperature * (1 - progress) + self.final_temperature * progress
                    self.loss_fn.update_temperature(current_temp)
                
                # Training
                train_metrics = self.train_epoch()
                self.training_history['train_loss'].append(train_metrics['total_loss'])
                self.training_history['train_ppb'].append(train_metrics['actual_ppb'])
                self.training_history['train_hit_rate'].append(train_metrics['hit_rate'])
                self.training_history['train_betting_freq'].append(train_metrics['betting_frequency'])
                
                # Validation
                if epoch % self.eval_every == 0:
                    val_metrics = self.validate_epoch()
                    self.training_history['val_loss'].append(val_metrics['total_loss'])
                    self.training_history['val_ppb'].append(val_metrics['hard_ppb'])
                    self.training_history['val_hit_rate'].append(val_metrics['hard_hit_rate'])
                    self.training_history['val_betting_freq'].append(val_metrics['hard_betting_freq'])
                    
                    # Log metrics
                    logger.info(f"Epoch {epoch}")
                    logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, PPB: {train_metrics['actual_ppb']:.4f}, Hit: {train_metrics['hit_rate']*100:.1f}%")
                    logger.info(f"Val   - Loss: {val_metrics['total_loss']:.4f}, PPB: {val_metrics['hard_ppb']:.4f}, Hit: {val_metrics['hard_hit_rate']*100:.1f}%")
                    logger.info(f"Val Profit: {val_metrics['cumulative_profit']:.2f}, Temp: {self.loss_fn.temperature:.3f}")
                    
                    # Early stopping check
                    current_val_ppb = val_metrics['hard_ppb']
                    if current_val_ppb > self.best_val_ppb:
                        self.best_val_ppb = current_val_ppb
                        self.save_checkpoint('best_model.pth', is_best=True)
                        logger.info(f"New best validation PPB: {current_val_ppb:.4f}")
                    
                    if self.early_stopping(current_val_ppb, epoch):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        logger.info(f"Best epoch was {self.early_stopping.best_epoch} with PPB {self.best_val_ppb:.4f}")
                        break
                
                # Save checkpoint
                if epoch % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                # Reset metrics trackers
                self.train_metrics.reset()
                self.val_metrics.reset()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted_checkpoint.pth')
            logger.info("Checkpoint saved. Training can be resumed later.")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.save_checkpoint('error_checkpoint.pth')
            raise
        
        # Save final checkpoint
        self.save_checkpoint('final_checkpoint.pth')
        
        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            serializable_history = {}
            for key, values in self.training_history.items():
                serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            json.dump(serializable_history, f, indent=2)
        
        logger.info("Training completed")
        return self.training_history
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ppb': self.best_val_ppb,
            'training_history': self.training_history,
            'loss_fn_state': {
                'alpha': self.loss_fn.alpha,
                'temperature': self.loss_fn.temperature,
                'commission': self.loss_fn.commission,
                'profit_weight': self.loss_fn.profit_weight,
                'accuracy_weight': self.loss_fn.accuracy_weight,
                'min_expected_profit': self.loss_fn.min_expected_profit
            },
            'model_config': self.model.get_model_info()
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_ppb = checkpoint['best_val_ppb']
        self.training_history = checkpoint['training_history']
        
        # Restore loss function state
        if 'loss_fn_state' in checkpoint:
            loss_state = checkpoint['loss_fn_state']
            self.loss_fn.alpha = loss_state['alpha']
            self.loss_fn.temperature = loss_state['temperature']
            self.loss_fn.commission = loss_state['commission']
            self.loss_fn.profit_weight = loss_state['profit_weight']
            self.loss_fn.accuracy_weight = loss_state['accuracy_weight']
            self.loss_fn.min_expected_profit = loss_state['min_expected_profit']
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Profit per bet
        axes[0, 1].plot(self.training_history['train_ppb'], label='Train')
        if self.training_history['val_ppb']:
            axes[0, 1].plot(self.training_history['val_ppb'], label='Validation')
        axes[0, 1].set_title('Profit per Bet')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Hit rate
        axes[1, 0].plot([h*100 for h in self.training_history['train_hit_rate']], label='Train')
        if self.training_history['val_hit_rate']:
            axes[1, 0].plot([h*100 for h in self.training_history['val_hit_rate']], label='Validation')
        axes[1, 0].set_title('Hit Rate (%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Betting frequency
        axes[1, 1].plot([f*100 for f in self.training_history['train_betting_freq']], label='Train')
        if self.training_history['val_betting_freq']:
            axes[1, 1].plot([f*100 for f in self.training_history['val_betting_freq']], label='Validation')
        axes[1, 1].set_title('Betting Frequency (%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_trainer(model: GreyhoundRacingModel,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  learning_rate: float = 1e-3,
                  weight_decay: float = 1e-4,
                  alpha: float = 1.0,
                  temperature: float = 1.0,
                  commission: float = 0.05,
                  profit_weight: float = 0.7,
                  accuracy_weight: float = 0.3,
                  device: torch.device = None) -> GreyhoundTrainer:
    """
    Factory function to create a trainer with reasonable defaults
    
    Args:
        model: The greyhound racing model
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization weight
        alpha: Confidence multiplier for betting
        temperature: Softmax temperature for betting selection
        commission: Betting commission rate
        profit_weight: Weight for profit loss component
        accuracy_weight: Weight for accuracy loss component
        device: Device to train on
        
    Returns:
        Configured trainer instance
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Force GPU memory allocation to ensure proper utilization
    if device.type == 'cuda':
        logger.info("Initializing GPU memory...")
        torch.cuda.empty_cache()
        # Warm up GPU with a dummy forward pass
        try:
            dummy_input = {
                'dog_features': torch.randn(2, 6, 100).to(device),
                'race_features': torch.randn(2, 20).to(device),
                'dog_mask': torch.ones(2, 6).to(device)
            }
            with torch.no_grad():
                _ = model(dummy_input)
            logger.info("GPU warm-up successful")
            del dummy_input
        except Exception as e:
            logger.warning(f"GPU warm-up failed: {e}")
        torch.cuda.empty_cache()
    
    # Create loss function
    loss_fn = GreyhoundBettingLoss(
        alpha=alpha,
        temperature=temperature,
        commission=commission,
        profit_weight=profit_weight,
        accuracy_weight=accuracy_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create trainer
    trainer = GreyhoundTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    
    return trainer

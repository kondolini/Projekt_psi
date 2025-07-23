"""
Training script for Greyhound Racing Model V1
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from machine_learning.data_processor import RaceDataProcessor, create_dataset
from machine_learning.model import GreyhoundRacingModel, BettingLoss, collate_race_batch


class RaceDataset(Dataset):
    """PyTorch Dataset for race data"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_data(data_dir: str, test_split_date: str = "2023-01-01") -> Tuple[Dict[str, Dog], List[Race], List[Race]]:
    """
    Load dogs and races, split into train/test by date
    """
    print("Loading dogs...")
    dogs = {}
    dogs_dir = os.path.join(data_dir, "dogs_enhanced")
    
    if not os.path.exists(dogs_dir):
        raise FileNotFoundError(f"Dogs directory not found: {dogs_dir}")
    
    for fname in os.listdir(dogs_dir):
        if fname.endswith('.pkl'):
            with open(os.path.join(dogs_dir, fname), 'rb') as f:
                bucket = pickle.load(f)
                for dog_id, dog_obj in bucket.items():
                    if isinstance(dog_obj, Dog):
                        dogs[dog_id] = dog_obj
    
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
            bucket_path = os.path.join(races_dir, fname)
            try:
                with open(bucket_path, 'rb') as f:
                    races_bucket = pickle.load(f)
                
                for storage_key, race in races_bucket.items():
                    if isinstance(race, Race) and len(race.dog_ids) >= 3:  # At least 3 dogs
                        all_races.append(race)
                        
            except Exception as e:
                print(f"Error loading race bucket {fname}: {e}")
                continue
    
    print(f"Loaded {len(all_races)} race objects from buckets")
    
    # Sort races chronologically
    all_races.sort(key=lambda r: r.get_race_datetime())
    
    # Split by date
    split_date = datetime.strptime(test_split_date, "%Y-%m-%d").date()
    train_races = [r for r in all_races if r.race_date < split_date]
    test_races = [r for r in all_races if r.race_date >= split_date]
    
    print(f"Split: {len(train_races)} train races, {len(test_races)} test races")
    
    return dogs, train_races, test_races


def load_checkpoint_for_resume(checkpoint_path: str, model: GreyhoundRacingModel, 
                              optimizer: torch.optim.Optimizer, 
                              scheduler: torch.optim.lr_scheduler._LRScheduler,
                              device: torch.device) -> Tuple[int, float]:
    """
    Load a checkpoint to resume training
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into  
        device: Device to load on
        
    Returns:
        Tuple of (start_epoch, best_val_profit)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load using the model's load_model method
    _, optimizer_state, scheduler_state, metadata = model.load_model(
        checkpoint_path, 
        device=str(device), 
        load_optimizer=True, 
        load_scheduler=True
    )
    
    # Restore optimizer and scheduler states
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state restored")
    
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        print("Scheduler state restored")
    
    start_epoch = metadata.get('epoch', 0) + 1  # Start from next epoch
    best_val_profit = metadata.get('metrics', {}).get('val_profit', float('-inf'))
    
    print(f"Resuming from epoch {start_epoch}, best val profit: {best_val_profit:.4f}")
    
    return start_epoch, best_val_profit


def train_model(
    model: GreyhoundRacingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: BettingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    start_epoch: int = 0,
    best_val_profit: float = float('-inf')
):
    """Training loop with GPU memory monitoring and progress bars"""
    
    train_losses = []
    val_losses = []
    
    # GPU memory tracking
    if device.type == 'cuda':
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    # Main training loop with progress bar
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Progress", initial=start_epoch, total=num_epochs):
        # Reset balance tracking for this epoch
        criterion.reset_balance()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = defaultdict(float)
        num_train_batches = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch)
            
            # Extract targets and generate realistic odds
            targets = batch["targets"]  # [batch_size, 6]
            
            # Generate realistic market odds based on uniform probability with noise
            # In practice, extract from race.odds when available
            batch_size = targets.shape[0]
            with torch.no_grad():
                # Start with inverse of uniform probability (6.0 for 6 dogs)
                base_odds = torch.full_like(targets, 6.0)
                
                # Add realistic variation: shorter odds for favorites, longer for outsiders
                noise = torch.randn_like(targets) * 1.5  # Random variation
                market_odds = base_odds + noise
                
                # Ensure realistic range: 1.2 to 20.0
                market_odds = torch.clamp(market_odds, min=1.2, max=20.0)
                
                # Make sure odds sum reasonably (market margin)
                odds_sum = 1.0 / market_odds.sum(dim=1, keepdim=True)
                target_margin = 1.1  # 10% overround
                market_odds = market_odds * (odds_sum / target_margin)
                market_odds = torch.clamp(market_odds, min=1.1, max=50.0)
            
            # Calculate loss
            loss, metrics = criterion(predictions, targets, market_odds)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            for key, value in metrics.items():
                train_metrics[key] += value
            num_train_batches += 1
            
            # Update progress bar with PPB
            train_pbar.set_postfix({
                'PPB': f'{metrics.get("ppb", 0):.4f}',
                'Loss': f'{loss.item():.4f}',
                'Balance': f'{metrics.get("total_balance", 1000):.0f}'
            })
            
            # GPU memory management
            if device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Average training metrics
        train_loss /= num_train_batches
        for key in train_metrics:
            train_metrics[key] /= num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(float)
        num_val_batches = 0
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)
                
                # Forward pass
                predictions = model(batch)
                targets = batch["targets"]
                
                # Generate realistic market odds for validation too
                batch_size = targets.shape[0]
                base_odds = torch.full_like(targets, 6.0)
                noise = torch.randn_like(targets) * 1.5
                market_odds = torch.clamp(base_odds + noise, min=1.2, max=20.0)
                odds_sum = 1.0 / market_odds.sum(dim=1, keepdim=True)
                market_odds = market_odds * (odds_sum / 1.1)
                market_odds = torch.clamp(market_odds, min=1.1, max=50.0)
                
                loss, metrics = criterion(predictions, targets, market_odds)
                
                val_loss += loss.item()
                for key, value in metrics.items():
                    val_metrics[key] += value
                num_val_batches += 1
                
                # Update progress bar
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Average validation metrics
        if num_val_batches > 0:
            val_loss /= num_val_batches
            for key in val_metrics:
                val_metrics[key] /= num_val_batches
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging with PPB as main metric
        tqdm.write(f"\n📊 Epoch {epoch + 1}/{num_epochs} Results:")
        tqdm.write(f"   💰 Train PPB: {train_metrics.get('ppb', 0):.6f}, Val PPB: {val_metrics.get('ppb', 0):.6f}")
        tqdm.write(f"   💵 Train Balance: {train_metrics.get('total_balance', 1000):.2f}, Val Balance: {val_metrics.get('total_balance', 1000):.2f}")
        tqdm.write(f"   📈 Train ROI: {train_metrics['roi']:.4f}, Val ROI: {val_metrics['roi']:.4f}")
        tqdm.write(f"   🎯 Train Hit Rate: {train_metrics['hit_rate']:.4f}, Val Hit Rate: {val_metrics['hit_rate']:.4f}")
        tqdm.write(f"   🎰 Train Bets: {train_metrics.get('num_bets', 0)}, Val Bets: {val_metrics.get('num_bets', 0)}")
        
        # Save model after every epoch
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_ppb': train_metrics.get('ppb', 0),
            'val_ppb': val_metrics.get('ppb', 0),
            'train_balance': train_metrics.get('total_balance', 1000),
            'val_balance': val_metrics.get('total_balance', 1000),
            'val_profit': val_metrics['actual_profit'],
            'train_profit': train_metrics['actual_profit'],
            'val_roi': val_metrics['roi'],
            'train_roi': train_metrics['roi'],
            'val_hit_rate': val_metrics['hit_rate'],
            'train_hit_rate': train_metrics['hit_rate']
        }
        
        # Save current epoch checkpoint
        model.save_model(
            save_path=os.path.join(save_dir, f'epoch_{epoch+1}.pth'),
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics=epoch_metrics
        )
        
        # Save best model
        if val_metrics['actual_profit'] > best_val_profit:
            best_val_profit = val_metrics['actual_profit']
            
            model.save_model(
                save_path=os.path.join(save_dir, 'best_model.pth'),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                metrics=epoch_metrics
            )
            tqdm.write(f"   💾 Saved new best model with val profit: {best_val_profit:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # Save final model
    final_metrics = {
        'train_loss': train_losses[-1] if train_losses else 0,
        'val_loss': val_losses[-1] if val_losses else 0,
        'final_epoch': num_epochs - 1
    }
    
    model.save_model(
        save_path=os.path.join(save_dir, 'final_model.pth'),
        epoch=num_epochs - 1,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        metrics=final_metrics
    )
    
    tqdm.write(f"\n🎯 Training completed! Final model saved.")
    
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train Greyhound Racing Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='machine_learning/outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_split', type=str, default='2023-01-01', help='Test split date')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g., outputs/epoch_10.pth)')
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild of cached datasets')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device with detailed GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Set optimal GPU settings
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Enable mixed precision if supported
        if hasattr(torch.cuda, 'amp'):
            print("Mixed precision training available")
        
    else:
        print("⚠️  WARNING: CUDA not available. Training will be much slower on CPU.")
        print("   Make sure you have:")
        print("   1. NVIDIA GPU with CUDA support")
        print("   2. PyTorch with CUDA support installed")
        print("   3. Compatible CUDA drivers")
    
    print("\n" + "="*60)
    print("🏁 GREYHOUND RACING MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("📁 Loading data...")
    dogs, train_races, test_races = load_data(args.data_dir, args.test_split)
    
    # Create data processor
    processor = RaceDataProcessor()
    processor_path = os.path.join(args.output_dir, 'encoders.pkl')
    
    # Fit encoders with caching support (respects force_rebuild flag)
    print("� Setting up data encoders...")
    if args.force_rebuild:
        print("� Force rebuilding encoders...")
    
    processor.fit_encoders(train_races, dogs, cache_path=processor_path, force_rebuild=args.force_rebuild)
    
    # Create datasets with caching
    print("📊 Creating datasets...")
    
    # Cache paths for datasets
    train_cache_path = os.path.join(args.output_dir, 'train_dataset.pkl')
    test_cache_path = os.path.join(args.output_dir, 'test_dataset.pkl')
    
    # Force rebuild datasets if flag is set
    if args.force_rebuild:
        print("🔄 Force rebuilding dataset cache...")
    
    train_dataset = create_dataset(train_races, dogs, processor, cache_path=train_cache_path, force_rebuild=args.force_rebuild)
    test_dataset = create_dataset(test_races, dogs, processor, cache_path=test_cache_path, force_rebuild=args.force_rebuild)
    
    # Split train into train/val
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    train_samples = train_dataset[:train_size]
    val_samples = train_dataset[train_size:]
    
    print(f"📈 Dataset sizes - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_dataset)}")
    
    # Create data loaders with GPU optimizations
    use_cuda = device.type == 'cuda'
    
    # Note: Using num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        RaceDataset(train_samples), 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_race_batch,
        pin_memory=use_cuda,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    
    val_loader = DataLoader(
        RaceDataset(val_samples), 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_race_batch,
        pin_memory=use_cuda,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    
    # Create model
    model = GreyhoundRacingModel(
        num_tracks=len(processor.track_encoder),
        num_classes=len(processor.class_encoder),
        num_categories=len(processor.category_encoder),
        num_trainers=len(processor.trainer_encoder),
        num_going_conditions=len(processor.going_encoder),
        commentary_vocab_size=processor.commentary_processor.vocab_size
    ).to(device)
    
    print(f"🧠 Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create loss function with optimistic alpha and fixed percentage betting
    criterion = BettingLoss(alpha=1.1, commission=0.05, bet_percentage=0.02)
    
    # Handle resume training
    start_epoch = 0
    best_val_profit = float('-inf')
    
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ Resume checkpoint not found: {args.resume}")
            print("💡 To resume training, use: --resume path/to/epoch_N.pth")
            print("💡 Available checkpoints:")
            for f in os.listdir(args.output_dir):
                if f.startswith('epoch_') and f.endswith('.pth'):
                    print(f"   {os.path.join(args.output_dir, f)}")
            return
        else:
            start_epoch, best_val_profit = load_checkpoint_for_resume(
                args.resume, model, optimizer, scheduler, device
            )
    
    print(f"\n🚀 Starting training from epoch {start_epoch+1} to {args.epochs}...")
    if args.resume:
        print(f"📁 Resuming from: {args.resume}")
        print(f"🎯 Current best validation profit: {best_val_profit:.4f}")
    
    print("\n💡 Training Tips:")
    print("   • Models are saved after every epoch as 'epoch_N.pth'")
    print("   • Best model is saved as 'best_model.pth'")
    print("   • To resume training: --resume outputs/epoch_N.pth")
    print("   • Use Ctrl+C to stop training gracefully")
    
    # Train model
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, device, args.output_dir, start_epoch, best_val_profit
        )
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'start_epoch': start_epoch,
            'total_epochs': args.epochs
        }
        
        with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"🏆 Best model: {os.path.join(args.output_dir, 'best_model.pth')}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print(f"📁 Latest models saved in: {args.output_dir}")
        print(f"💡 Resume with: --resume {os.path.join(args.output_dir, 'epoch_*.pth')}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

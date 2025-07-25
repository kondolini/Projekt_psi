import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, date


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def save_encoders(dataset, save_dir: str):
    """Save label encoders and vocabularies for inference"""
    os.makedirs(save_dir, exist_ok=True)
    
    encoders = {
        'track_encoder': dataset.track_encoder,
        'class_encoder': dataset.class_encoder,
        'category_encoder': dataset.category_encoder,
        'trainer_encoder': dataset.trainer_encoder,
        'going_encoder': dataset.going_encoder,
        'commentary_vocab': dataset.commentary_vocab,
        'vocab_sizes': dataset.vocab_sizes
    }
    
    encoder_path = os.path.join(save_dir, 'encoders.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    logging.info(f"Encoders saved to {encoder_path}")
    return encoder_path


def load_encoders(encoder_path: str) -> Dict:
    """Load saved encoders for inference"""
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    
    logging.info(f"Encoders loaded from {encoder_path}")
    return encoders


def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        
        print(f"🚀 GPU INFO:")
        print(f"  - Available GPUs: {gpu_count}")
        print(f"  - Current device: {current_device} ({device_name})")
        print(f"  - Memory allocated: {memory_allocated:.2f} GB")
        print(f"  - Memory reserved: {memory_reserved:.2f} GB")
        
        return True
    else:
        print("❌ No GPU available. Using CPU.")
        return False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate approximate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def print_model_info(model: torch.nn.Module):
    """Print comprehensive model information"""
    total_params, trainable_params = count_parameters(model)
    model_size = calculate_model_size_mb(model)
    
    print(f"📊 MODEL INFO:")
    print(f"  - Total parameters: {format_number(total_params)}")
    print(f"  - Trainable parameters: {format_number(trainable_params)}")
    print(f"  - Model size: {model_size:.2f} MB")
    print(f"  - Non-trainable parameters: {format_number(total_params - trainable_params)}")


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seeds set to {seed}")


def create_run_name() -> str:
    """Create a unique run name based on timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"greyhound_run_{timestamp}"


def save_run_config(config: Dict, save_dir: str):
    """Save run configuration for reproducibility"""
    config_path = os.path.join(save_dir, 'run_config.json')
    import json
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (date, datetime)):
            serializable_config[key] = value.isoformat()
        elif hasattr(value, '__dict__'):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logging.info(f"Run configuration saved to {config_path}")


def estimate_training_time(num_epochs: int, 
                          time_per_epoch_seconds: float, 
                          eval_every: int = 1) -> str:
    """Estimate total training time"""
    
    eval_epochs = num_epochs // eval_every
    total_seconds = num_epochs * time_per_epoch_seconds + eval_epochs * time_per_epoch_seconds * 0.2  # Eval overhead
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def check_data_quality(train_dataset, val_dataset) -> Dict[str, any]:
    """Perform data quality checks"""
    
    checks = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'vocab_sizes': train_dataset.vocab_sizes,
        'warnings': []
    }
    
    # Check dataset sizes
    if len(train_dataset) < 1000:
        checks['warnings'].append(f"Small training set: {len(train_dataset)} samples")
    
    if len(val_dataset) < 100:
        checks['warnings'].append(f"Small validation set: {len(val_dataset)} samples")
    
    # Check vocabulary sizes
    vocab_sizes = train_dataset.vocab_sizes
    if vocab_sizes['num_tracks'] < 5:
        checks['warnings'].append(f"Few tracks: {vocab_sizes['num_tracks']}")
    
    if vocab_sizes['commentary_vocab_size'] < 100:
        checks['warnings'].append(f"Small commentary vocabulary: {vocab_sizes['commentary_vocab_size']}")
    
    # Sample a few examples to check data
    try:
        sample = train_dataset[0]
        
        # Check for NaN values
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                checks['warnings'].append(f"NaN values found in {key}")
        
        # Check data shapes
        expected_shapes = {
            'race_features': 8,  # Should have 8 race features
            'dog_features': (train_dataset.max_dogs_per_race, 3),  # Should be (max_dogs, 3)
            'win_labels': train_dataset.max_dogs_per_race,  # Should sum to 1
        }
        
        for key, expected_shape in expected_shapes.items():
            if key in sample:
                actual_shape = sample[key].shape
                if isinstance(expected_shape, tuple):
                    if actual_shape != expected_shape:
                        checks['warnings'].append(f"Unexpected shape for {key}: {actual_shape} vs {expected_shape}")
                else:
                    if len(actual_shape) == 1 and actual_shape[0] != expected_shape:
                        checks['warnings'].append(f"Unexpected shape for {key}: {actual_shape} vs ({expected_shape},)")
        
        # Check if win labels sum to 1 (one winner per race)
        win_sum = sample['win_labels'].sum().item()
        if abs(win_sum - 1.0) > 0.01:
            checks['warnings'].append(f"Win labels don't sum to 1: {win_sum}")
            
    except Exception as e:
        checks['warnings'].append(f"Error sampling data: {e}")
    
    return checks


def print_data_summary(train_dataset, val_dataset):
    """Print comprehensive data summary"""
    
    quality_checks = check_data_quality(train_dataset, val_dataset)
    
    print(f"📈 DATA SUMMARY:")
    print(f"  - Training samples: {format_number(quality_checks['train_size'])}")
    print(f"  - Validation samples: {format_number(quality_checks['val_size'])}")
    print(f"  - Split ratio: {quality_checks['train_size']/(quality_checks['train_size']+quality_checks['val_size']):.1%} train")
    
    print(f"\n🗂️ VOCABULARY SIZES:")
    vocab = quality_checks['vocab_sizes']
    for key, size in vocab.items():
        print(f"  - {key}: {format_number(size)}")
    
    if quality_checks['warnings']:
        print(f"\n⚠️ DATA WARNINGS:")
        for warning in quality_checks['warnings']:
            print(f"  - {warning}")
    else:
        print(f"\n✅ Data quality checks passed")


def calculate_betting_stats(profits: List[float]) -> Dict[str, float]:
    """Calculate comprehensive betting statistics"""
    
    if not profits:
        return {
            'total_profit': 0.0,
            'profit_per_bet': 0.0,
            'win_rate': 0.0,
            'total_bets': 0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'profit_std': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    profits = np.array(profits)
    
    # Basic stats
    total_profit = profits.sum()
    profit_per_bet = profits.mean()
    win_rate = (profits > 0).mean()
    total_bets = len(profits)
    max_win = profits.max()
    max_loss = profits.min()
    profit_std = profits.std()
    
    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = profit_per_bet / profit_std if profit_std > 0 else 0.0
    
    # Maximum drawdown
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()
    
    return {
        'total_profit': float(total_profit),
        'profit_per_bet': float(profit_per_bet),
        'win_rate': float(win_rate),
        'total_bets': int(total_bets),
        'max_win': float(max_win),
        'max_loss': float(max_loss),
        'profit_std': float(profit_std),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown)
    }


def print_betting_stats(profits: List[float], title: str = "BETTING STATS"):
    """Print formatted betting statistics"""
    
    stats = calculate_betting_stats(profits)
    
    print(f"\n💰 {title}:")
    print(f"  - Total profit: ${stats['total_profit']:.2f}")
    print(f"  - Profit per bet: ${stats['profit_per_bet']:.4f}")
    print(f"  - Win rate: {stats['win_rate']:.1%}")
    print(f"  - Total bets: {stats['total_bets']:,}")
    print(f"  - Best win: ${stats['max_win']:.2f}")
    print(f"  - Worst loss: ${stats['max_loss']:.2f}")
    print(f"  - Profit std: ${stats['profit_std']:.4f}")
    print(f"  - Sharpe ratio: {stats['sharpe_ratio']:.3f}")
    print(f"  - Max drawdown: ${stats['max_drawdown']:.2f}")


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """Create directory structure for training run"""
    
    directories = {
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'plots': os.path.join(base_dir, 'plots'),
        'encoders': os.path.join(base_dir, 'encoders'),
        'outputs': os.path.join(base_dir, 'outputs')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def cleanup_old_checkpoints(checkpoint_dir: str, keep_latest: int = 5):
    """Clean up old checkpoint files, keeping only the latest ones"""
    
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append((filepath, os.path.getmtime(filepath)))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old checkpoints (keep best_model.pth and final_checkpoint.pth)
    files_to_keep = {'best_model.pth', 'final_checkpoint.pth', 'interrupted_checkpoint.pth'}
    removed_count = 0
    
    for i, (filepath, _) in enumerate(checkpoint_files):
        filename = os.path.basename(filepath)
        
        if i >= keep_latest and filename not in files_to_keep:
            try:
                os.remove(filepath)
                removed_count += 1
            except OSError:
                pass
    
    if removed_count > 0:
        logging.info(f"Cleaned up {removed_count} old checkpoint files")

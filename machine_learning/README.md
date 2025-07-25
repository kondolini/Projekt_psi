# Greyhound Racing Prediction Model

A comprehensive neural network system for predicting greyhound racing outcomes with a focus on betting profitability.

## 🎯 Overview

This model uses a differentiable betting loss function to optimize for actual betting profitability rather than just prediction accuracy. It combines:

- **Race-level features**: Track, weather, distance, timing
- **Dog-level features**: Static info (weight, age, trainer) + historical performance RNN
- **Commentary analysis**: NLP processing of race commentary
- **Economic optimization**: Soft betting selection with temperature annealing

## 🏗️ Architecture

### Model Components

1. **Race Encoder**: Processes race metadata (track, class, weather)
2. **Dog Encoder**: Static dog features + trainer embeddings
3. **History RNN**: LSTM over historical race performances
4. **Commentary Embeddings**: NLP processing of race comments
5. **Final Predictor**: Combines all features → win probabilities

### Loss Function

The key innovation is the **differentiable betting loss** that:
- Uses soft selection (temperature-scaled softmax) instead of hard argmax
- Optimizes expected betting returns while maintaining gradient flow
- Combines profitability loss with accuracy loss
- Includes commission and minimum profit thresholds

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r machine_learning/requirements.txt

# Ensure you have the data constructed
python data_construction/build_and_save_races.py
```

### Basic Training

```bash
# Simple training run
python machine_learning/train.py --epochs 50 --batch_size 32

# Training with custom parameters
python machine_learning/train.py \
    --epochs 100 \
    --learning_rate 0.001 \
    --alpha 1.2 \
    --commission 0.05 \
    --val_start_date 2023-01-01
```

### Resume Training

```bash
# Resume from interruption
python machine_learning/train.py --resume checkpoints/interrupted_checkpoint.pth

# Load model but start fresh training
python machine_learning/train.py --from_checkpoint checkpoints/best_model.pth --epochs 50
```

## 📊 Key Features

### Training Features
- ✅ Graceful stopping and resuming from checkpoints
- ✅ GPU support with automatic detection
- ✅ Progress tracking with tqdm
- ✅ Comprehensive logging
- ✅ Early stopping based on validation PPB (Profit Per Bet)
- ✅ Temperature annealing for improved convergence
- ✅ Automatic checkpoint cleanup

### Monitoring Features
- ✅ Real-time PnL tracking during training
- ✅ Hit rate monitoring
- ✅ Betting frequency tracking
- ✅ Training plots and visualizations
- ✅ Data quality validation

### Betting Simulation
- ✅ Hard betting evaluation (actual strategy simulation)
- ✅ Commission handling
- ✅ Minimum profit thresholds
- ✅ Comprehensive betting statistics

## 📈 Model Parameters

### Core Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 32 | Embedding dimension for categorical features |
| `hidden_dim` | 64 | Hidden layer dimension |
| `rnn_hidden_dim` | 32 | LSTM hidden state size |
| `dropout_rate` | 0.2 | Dropout rate for regularization |
| `max_history_length` | 10 | Historical races per dog |
| `max_commentary_length` | 5 | Commentary tags per race |

### Loss Function Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Confidence multiplier (model vs market) |
| `temperature` | 1.0 | Softmax temperature (annealed during training) |
| `commission` | 0.05 | Betting commission rate |
| `profit_weight` | 0.7 | Weight for profitability vs accuracy |
| `min_expected_profit` | 0.0 | Minimum profit threshold for betting |

## 📁 Output Structure

Training creates the following directory structure:

```
machine_learning/outputs/greyhound_run_20240125_143022/
├── checkpoints/
│   ├── best_model.pth              # Best validation PPB model
│   ├── final_checkpoint.pth        # Final training state
│   ├── checkpoint_epoch_5.pth      # Periodic checkpoints
│   └── interrupted_checkpoint.pth  # Emergency save
├── logs/
│   ├── training.log               # Detailed training log
│   └── training_history.json     # Metrics history
├── plots/
│   └── training_history.png      # Training visualization
├── encoders/
│   └── encoders.pkl              # Saved encoders for inference
└── outputs/
    └── run_config.json           # Run configuration
```

## 🎯 Key Metrics

The model tracks several important metrics:

### During Training (Soft Metrics)
- **Total Loss**: Combined profit + accuracy loss
- **Profit Loss**: Differentiable betting returns
- **Accuracy Loss**: Standard cross-entropy
- **Soft Hit Rate**: Probabilistic winner prediction
- **Soft PPB**: Expected profit per bet (differentiable)

### During Validation (Hard Metrics)
- **Hard Hit Rate**: Actual betting win rate
- **Hard PPB**: Actual profit per bet from simulation
- **Betting Frequency**: Fraction of races with profitable bets
- **Cumulative Profit**: Total profit from validation period

## 🔧 Advanced Usage

### Custom Data Split

```python
from datetime import date
from machine_learning.dataset import create_train_val_split

# Custom validation period
val_start = date(2023, 1, 1)
val_end = date(2023, 6, 30)
train_races, val_races = create_train_val_split(races, val_start, val_end)
```

### Custom Model Architecture

```python
from machine_learning.model import GreyhoundRacingModel

model = GreyhoundRacingModel(
    num_tracks=50,
    num_classes=10,
    # ... other vocab sizes
    embedding_dim=64,      # Larger embeddings
    hidden_dim=128,        # Deeper network
    rnn_hidden_dim=64,     # More RNN capacity
    dropout_rate=0.3       # More regularization
)
```

### Custom Loss Function

```python
from machine_learning.loss import GreyhoundBettingLoss

loss_fn = GreyhoundBettingLoss(
    alpha=1.5,             # More confident than market
    temperature=0.5,       # Sharper selection
    commission=0.03,       # Lower commission
    profit_weight=0.8,     # Focus more on profit
    min_expected_profit=0.05  # Higher profit threshold
)
```

## 📊 Expected Performance

Based on the model architecture and loss function:

- **Hit Rate**: 25-35% (better than random ~16.7% for 6-dog races)
- **Profit Per Bet**: Target 0.05-0.15 units (5-15% ROI)
- **Betting Frequency**: 15-30% of races (selective betting)
- **Sharpe Ratio**: Target >0.5 for consistent profitability

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python machine_learning/train.py --batch_size 16
   ```

2. **Slow Training**
   ```bash
   # Use fewer workers on Windows
   python machine_learning/train.py --num_workers 0
   ```

3. **Data Loading Errors**
   ```bash
   # Ensure data is built first
   python data_construction/build_and_save_races.py
   ```

### Debug Mode

```bash
# Enable debug logging
python machine_learning/train.py --log_level DEBUG

# Quick training test
python machine_learning/example.py
```

## 🔬 Model Validation

The system includes comprehensive validation:

1. **Temporal Split**: Ensures no data leakage (train on past, validate on future)
2. **Hard Betting Simulation**: Tests actual betting strategy
3. **Data Quality Checks**: Validates data integrity
4. **Economic Validation**: Ensures profitable betting thresholds

## 📚 API Reference

### Core Classes

- `GreyhoundRacingModel`: Main neural network
- `GreyhoundDataset`: Data loading and preprocessing
- `GreyhoundBettingLoss`: Differentiable betting loss
- `GreyhoundTrainer`: Training loop with checkpointing

### Utility Functions

- `load_data_from_buckets()`: Load data from storage
- `create_train_val_split()`: Temporal data splitting
- `hard_betting_evaluation()`: Betting simulation
- `print_gpu_info()`: System information

## 🏆 Next Steps

1. **Hyperparameter Tuning**: Grid search over key parameters
2. **Feature Engineering**: Add more sophisticated features
3. **Ensemble Methods**: Combine multiple models
4. **Live Deployment**: Real-time prediction system
5. **Advanced NLP**: Better commentary processing

---

For questions or issues, check the training logs or enable debug mode for detailed information.

# 🤖 Machine Learning Module - Greyhound Racing Prediction

This module implements a deep learning model for predicting greyhound racing outcomes, optimized for profitable betting using the Kelly criterion.

## 🏗️ Architecture Overview

### **Model Components (V1):**

1. **Race-Level Features**: Date/time, track, class, category, distance
2. **Dog Static Features**: Trainer, weight, dog ID embeddings  
3. **Historical Performance**: RNN over past race participations
4. **Commentary Analysis**: Embedding of behavioral tags
5. **Win Probability Prediction**: Softmax over all traps

### **Key Features:**
- ✅ **Chronological Training**: Prevents data leakage
- ✅ **Kelly Criterion**: Optimal bet sizing
- ✅ **Commentary Processing**: Structured behavioral tags
- ✅ **Robust NaN Handling**: Unknown categories for missing data
- ✅ **Betting Simulation**: ROI, hit rate, Sharpe ratio metrics

## 📁 Files Structure

```
machine_learning/
├── data_processor.py    # Data preprocessing and feature engineering
├── model.py            # Neural network architecture
├── train.py            # Training script
├── evaluate.py         # Evaluation and betting simulation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd machine_learning
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py --data_dir ../data --epochs 50 --batch_size 32
```

### 3. Evaluate Model
```bash
python evaluate.py --data_dir ../data --model_dir outputs
```

## 📊 Input Data Format

The model expects the following data structure:

### **Race Object:**
- `race_date`, `race_time`: Temporal features
- `track_name`: Track identifier  
- `race_class`: Competition level
- `category`: Handicap vs regular (can be None)
- `distance`: Race distance in meters
- `dog_ids`: Dictionary mapping trap number to dog ID
- `odds`: Market odds per trap (for evaluation)
- `race_times`: Actual finish times (for target labels)

### **Dog Object:**
- `id`: Unique dog identifier
- `trainer`: Trainer name
- `weight`: Dog weight
- `race_participations`: List of historical races

### **RaceParticipation Object:**
- Historical race performance data
- Commentary tags for behavioral analysis
- Position, times, track conditions

## 🎯 Training Strategy

### **Objective Function:**
The model maximizes **Expected Betting Profit** using Kelly criterion:

```
E[Profit] = Σ (odds * α * p_i - 1) * kelly_bet * (1 - commission)
```

Where:
- `p_i`: Predicted win probability for dog i
- `odds`: Market odds
- `α`: Odds reduction factor (market movement)
- `kelly_bet`: Kelly optimal bet size
- `commission`: Exchange commission rate

### **Data Splits:**
- **Chronological Split**: Train on races before 2023-01-01, test after
- **No Data Leakage**: Only use historical data up to race datetime
- **Validation**: 20% of training data for hyperparameter tuning

## 📈 Evaluation Metrics

### **Classification Metrics:**
- **Accuracy**: Fraction of races where top prediction wins
- **Top-3 Accuracy**: Winner in top 3 predictions

### **Betting Performance:**
- **ROI**: Return on Investment (profit / total_bet)
- **Hit Rate**: Fraction of profitable races
- **Profit Per Bet**: Average profit per race
- **Sharpe Ratio**: Risk-adjusted returns
- **Kelly Sizing**: Optimal bankroll management

## 🔧 Model Configuration

### **Hyperparameters (V1):**
```python
embedding_dim = 32          # Categorical feature embeddings
hidden_dim = 64            # Dense layer size
rnn_hidden_dim = 32        # Historical RNN size
max_history_length = 10    # Number of past races
dropout_rate = 0.2         # Regularization
```

### **Betting Parameters:**
```python
alpha = 0.95              # Odds reduction (market movement)
commission = 0.05         # Exchange commission
max_bet = 0.25            # Maximum 25% of bankroll per bet
```

## 🎪 Commentary Tag Processing

Commentary tags are extracted from race participation comments and processed as:

```python
# Example tags: "SAw,Ld1/2,EvCh" -> ["SAw", "Ld1/2", "EvCh"]
commentary_vocab = ["<PAD>", "<UNK>", "SAw", "Ld1/2", "EvCh", ...]
tag_embeddings = nn.Embedding(vocab_size, embed_dim)
```

Common tags:
- `SAw`: Slowly Away
- `Ld1/2`: Led at halfway
- `EvCh`: Every Chance
- `Bmp`: Bumped

## 🔮 Future Improvements (V2+)

### **Planned Features:**
1. **Weather Integration**: Rainfall, temperature, humidity effects
2. **Pedigree GNN**: Graph Neural Network for sire/dam relationships  
3. **Advanced Commentary**: Transformer-based text processing
4. **Track Bias**: Surface condition and geometry modeling
5. **Ensemble Models**: Multiple architectures for robustness
6. **Online Learning**: Continuous model updates

### **Architecture Enhancements:**
- **Attention Mechanisms**: Focus on relevant historical races
- **Multi-Task Learning**: Predict win/place/show simultaneously
- **Uncertainty Quantification**: Bayesian Neural Networks
- **Advanced Bet Sizing**: Beyond Kelly criterion

## 📋 Known Limitations (V1)

1. **Simplified Odds**: Uses dummy market odds for evaluation
2. **No Weather**: Weather features not yet integrated
3. **Basic Commentary**: Simple tag embedding (no NLP)
4. **Fixed History**: Only last N races (no adaptive selection)
5. **Single Objective**: Only optimizes win probability

## 🎮 Usage Examples

### **Training with Custom Parameters:**
```bash
python train.py \
    --data_dir ../data \
    --output_dir custom_model \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0005 \
    --test_split 2023-06-01
```

### **Evaluation on Different Period:**
```bash
python evaluate.py \
    --data_dir ../data \
    --model_dir custom_model \
    --test_split 2023-06-01
```

## 🏆 Expected Performance

Based on similar racing prediction models:

- **Accuracy**: 15-25% (vs 16.7% random for 6-dog races)
- **ROI**: 5-15% (after commission)
- **Hit Rate**: 30-50% of races show profit
- **Sharpe Ratio**: 0.5-1.5 (reasonable risk-adjusted returns)

**Note**: Performance heavily depends on data quality, market efficiency, and hyperparameter tuning.

---

## 🤝 Contributing

When modifying the model:

1. **Maintain Chronological Validity**: Never use future data
2. **Test Betting Logic**: Verify Kelly calculations
3. **Document Changes**: Update this README
4. **Validate Performance**: Compare against baseline
5. **Handle Edge Cases**: Missing data, empty races, etc.

---

*Built for the Greyhound Racing Prediction project. For questions, check the main project README.*

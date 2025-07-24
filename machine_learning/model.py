"""
Neural Network Model for Greyhound Racing Prediction
V1: Simplified architecture focusing on RNN for race history + Kelly criterion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import json


class GreyhoundRacingModel(nn.Module):
    """
    V1 Model Architecture:
    - Race-level features → Dense layers
    - Per-dog features: Static + RNN for history + Commentary embeddings
    - Output: Win probabilities p_i for each dog
    """
    
    def __init__(
        self,
        # Vocabulary sizes for embeddings
        num_tracks: int,
        num_classes: int,
        num_categories: int,
        num_trainers: int,
        num_going_conditions: int,
        commentary_vocab_size: int,
        
        # Model hyperparameters
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 32,
        commentary_embed_dim: int = 16,
        max_history_length: int = 10,
        max_commentary_length: int = 5,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # Store hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_history_length = max_history_length
        
        # === Race-level Embeddings ===
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Race-level dense features: day_of_week, month, hour, minute, distance_norm
        race_dense_dim = 5
        race_embed_dim = 3 * embedding_dim  # track + class + category
        
        self.race_processor = nn.Sequential(
            nn.Linear(race_dense_dim + race_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # === Dog-level Features ===
        
        # Static dog embeddings
        self.trainer_embedding = nn.Embedding(num_trainers, embedding_dim)
        self.dog_id_embedding = nn.Embedding(10000, embedding_dim)  # Hash space for dog IDs
        
        # Dog static features: weight_norm (1 feature)
        dog_static_dim = 1
        dog_embed_dim = 2 * embedding_dim  # trainer + dog_id
        
        self.dog_static_processor = nn.Sequential(
            nn.Linear(dog_static_dim + dog_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # === Race History RNN ===
        
        # History features per race: position, distance_norm, time_norm, going_idx
        history_dense_dim = 3  # position, distance_norm, time_norm
        self.going_embedding = nn.Embedding(num_going_conditions, embedding_dim // 2)
        
        # Commentary embeddings
        self.commentary_embedding = nn.Embedding(commentary_vocab_size, commentary_embed_dim)
        self.commentary_processor = nn.Sequential(
            nn.Linear(max_commentary_length * commentary_embed_dim, embedding_dim // 2),
            nn.ReLU()
        )
        
        # RNN input: dense features + going embedding + commentary embedding
        rnn_input_dim = history_dense_dim + (embedding_dim // 2) + (embedding_dim // 2)
        
        self.history_rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if rnn_hidden_dim > 1 else 0
        )
        
        # === Final Prediction Layers ===
        
        # Combine all features: race_features + dog_static + rnn_output
        combined_dim = hidden_dim + hidden_dim + rnn_hidden_dim
        
        self.final_processor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)  # Single output: win probability logit
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            batch: Dictionary with keys:
                - race_features: [batch_size, race_feature_dim]
                - dog_features: [batch_size, 6, dog_feature_dim] (6 traps)
                - history_features: [batch_size, 6, max_history, history_feature_dim]
                
        Returns:
            win_probs: [batch_size, 6] - Win probabilities for each trap
        """
        batch_size = batch["race_features"].shape[0]
        
        # === Process Race Features ===
        race_dense = batch["race_features"][:, :5]  # day, month, hour, minute, distance
        track_idx = batch["race_features"][:, 5].long()
        class_idx = batch["race_features"][:, 6].long()
        category_idx = batch["race_features"][:, 7].long()
        
        # Clamp race-level indices to valid ranges
        track_idx = torch.clamp(track_idx, 0, self.track_embedding.num_embeddings - 1)
        class_idx = torch.clamp(class_idx, 0, self.class_embedding.num_embeddings - 1)
        category_idx = torch.clamp(category_idx, 0, self.category_embedding.num_embeddings - 1)
        
        # Race embeddings
        track_emb = self.track_embedding(track_idx)
        class_emb = self.class_embedding(class_idx)
        category_emb = self.category_embedding(category_idx)
        
        # Combine race features
        race_combined = torch.cat([race_dense, track_emb, class_emb, category_emb], dim=1)
        race_features = self.race_processor(race_combined)  # [batch_size, hidden_dim]
        
        # === Process Each Dog ===
        win_logits = []
        
        for trap_idx in range(6):  # Process each trap
            # Static dog features
            dog_dense = batch["dog_features"][:, trap_idx, :1]  # weight_norm
            trainer_idx = batch["dog_features"][:, trap_idx, 1].long()
            dog_id_hash = batch["dog_features"][:, trap_idx, 2].long()
            
            # Clamp dog-level indices to valid ranges
            trainer_idx = torch.clamp(trainer_idx, 0, self.trainer_embedding.num_embeddings - 1)
            dog_id_hash = torch.clamp(dog_id_hash, 0, self.dog_id_embedding.num_embeddings - 1)
            
            # Dog embeddings
            trainer_emb = self.trainer_embedding(trainer_idx)
            dog_id_emb = self.dog_id_embedding(dog_id_hash)
            
            # Combine static dog features
            dog_static_combined = torch.cat([dog_dense, trainer_emb, dog_id_emb], dim=1)
            dog_static_features = self.dog_static_processor(dog_static_combined)
            
            # === Process Race History ===
            history = batch["history_features"][:, trap_idx]  # [batch_size, max_history, feature_dim]
            
            # Split history features
            positions = history[:, :, 0:1]  # [batch_size, max_history, 1]
            distances = history[:, :, 1:2]
            times = history[:, :, 2:3]
            going_indices = history[:, :, 3].long()  # [batch_size, max_history]
            commentary_indices = history[:, :, 4:].long()  # [batch_size, max_history, max_commentary_length]
            
            # Clamp indices to valid ranges to prevent IndexError
            going_indices = torch.clamp(going_indices, 0, self.going_embedding.num_embeddings - 1)
            commentary_indices = torch.clamp(commentary_indices, 0, self.commentary_embedding.num_embeddings - 1)
            
            # Process going conditions
            going_emb = self.going_embedding(going_indices)  # [batch_size, max_history, embed_dim//2]
            
            # Process commentary
            comment_emb = self.commentary_embedding(commentary_indices)  # [batch_size, max_history, max_comment, embed_dim]
            comment_emb_flat = comment_emb.view(batch_size, self.max_history_length, -1)
            comment_features = self.commentary_processor(comment_emb_flat)  # [batch_size, max_history, embed_dim//2]
            
            # Combine history features for RNN
            history_dense = torch.cat([positions, distances, times], dim=2)  # [batch_size, max_history, 3]
            rnn_input = torch.cat([history_dense, going_emb, comment_features], dim=2)
            
            # RNN forward pass
            rnn_output, _ = self.history_rnn(rnn_input)  # [batch_size, max_history, rnn_hidden_dim]
            
            # Use last output (most recent race info)
            last_output = rnn_output[:, -1, :]  # [batch_size, rnn_hidden_dim]
            
            # === Combine All Features ===
            combined_features = torch.cat([
                race_features,           # Race context
                dog_static_features,     # Dog static info
                last_output             # Race history summary
            ], dim=1)
            
            # Final prediction
            trap_logit = self.final_processor(combined_features)  # [batch_size, 1]
            win_logits.append(trap_logit)
        
        # Stack and apply softmax across traps
        win_logits = torch.cat(win_logits, dim=1)  # [batch_size, 6]
        win_probs = F.softmax(win_logits, dim=1)   # [batch_size, 6]
        
        return win_probs
    
    def save_model(self, save_path: str, epoch: int = None, optimizer_state: dict = None, 
                   scheduler_state: dict = None, metrics: dict = None):
        """
        Save model checkpoint with all necessary information
        
        Args:
            save_path: Path to save the model (should end with .pth)
            epoch: Current training epoch
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict  
            metrics: Training metrics (loss, accuracy, etc.)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state and hyperparameters
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_tracks': self.track_embedding.num_embeddings,
                'num_classes': self.class_embedding.num_embeddings,
                'num_categories': self.category_embedding.num_embeddings,
                'num_trainers': self.trainer_embedding.num_embeddings,
                'num_going_conditions': self.going_embedding.num_embeddings,
                'commentary_vocab_size': self.commentary_embedding.num_embeddings,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'rnn_hidden_dim': self.rnn_hidden_dim,
                'max_history_length': self.max_history_length,
            },
            'epoch': epoch,
            'metrics': metrics,
        }
        
        # Add optimizer and scheduler states if provided
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
            
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        # Also save model config as JSON for easy reading
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(checkpoint['model_config'], f, indent=2)
            
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu', load_optimizer: bool = False, 
                   load_scheduler: bool = False) -> Tuple['GreyhoundRacingModel', Optional[dict], Optional[dict], Optional[dict]]:
        """
        Load model from checkpoint
        
        Args:
            load_path: Path to the saved model
            device: Device to load the model on ('cpu', 'cuda', etc.)
            load_optimizer: Whether to return optimizer state
            load_scheduler: Whether to return scheduler state
            
        Returns:
            model: Loaded GreyhoundRacingModel
            optimizer_state: Optimizer state dict (if load_optimizer=True)
            scheduler_state: Scheduler state dict (if load_scheduler=True)
            metadata: Dictionary with epoch, metrics, etc.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=device)
        
        # Extract model configuration
        config = checkpoint['model_config']
        
        # Create model instance with saved configuration
        model = cls(
            num_tracks=config['num_tracks'],
            num_classes=config['num_classes'],
            num_categories=config['num_categories'],
            num_trainers=config['num_trainers'],
            num_going_conditions=config['num_going_conditions'],
            commentary_vocab_size=config['commentary_vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            rnn_hidden_dim=config['rnn_hidden_dim'],
            max_history_length=config['max_history_length'],
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Prepare return values
        optimizer_state = checkpoint.get('optimizer_state_dict') if load_optimizer else None
        scheduler_state = checkpoint.get('scheduler_state_dict') if load_scheduler else None
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics')
        }
        
        print(f"Model loaded from {load_path}")
        if metadata['epoch'] is not None:
            print(f"  Epoch: {metadata['epoch']}")
        if metadata['metrics']:
            print(f"  Metrics: {metadata['metrics']}")
            
        return model, optimizer_state, scheduler_state, metadata
    
    def get_device(self) -> torch.device:
        """Get the device the model is currently on"""
        return next(self.parameters()).device


class BettingLoss(nn.Module):
    """
    Simplified loss function that optimizes for betting profitability
    Uses unit betting (1 unit per race) to avoid balance tracking issues
    """
    
    def __init__(self, alpha: float = 1.1, commission: float = 0.05, 
                 bet_percentage: float = 0.02, dynamic_alpha: bool = True, 
                 min_alpha: float = 0.95, max_alpha: float = 1.2):
        super().__init__()
        self.base_alpha = alpha
        self.alpha = alpha
        self.commission = commission
        self.bet_percentage = bet_percentage  # Not used in unit betting but kept for compatibility
        
        # Dynamic alpha parameters
        self.dynamic_alpha = dynamic_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Tracking variables
        self.total_profit = 0.0
        self.total_races = 0
        
    def reset_balance(self):
        """Reset the cumulative tracking for new epoch"""
        self.total_profit = 0.0
        self.total_races = 0
        
    def _update_dynamic_alpha(self, recent_results: List[int]):
        """Update alpha based on recent performance (within batch only)"""
        if not self.dynamic_alpha or len(recent_results) < 10:
            return
            
        hit_rate = sum(recent_results) / len(recent_results)
        target_hit_rate = 0.25
        
        if hit_rate < target_hit_rate - 0.05:
            self.alpha = min(self.alpha * 1.02, self.max_alpha)
        elif hit_rate > target_hit_rate + 0.05:
            self.alpha = max(self.alpha * 0.98, self.min_alpha)
            
    def _update_dynamic_alpha_by_rate(self, hit_rate: float, ppb: float):
        """Update alpha based on hit rate and profit per bet"""
        if not self.dynamic_alpha:
            return
            
        target_hit_rate = 0.25
        
        # Adjust alpha based on performance
        if hit_rate < target_hit_rate - 0.05 and ppb < 0:
            # Poor performance, increase optimism slightly
            self.alpha = min(self.alpha * 1.01, self.max_alpha)
        elif hit_rate > target_hit_rate + 0.05 and ppb > 0:
            # Good performance, reduce optimism slightly  
            self.alpha = max(self.alpha * 0.99, self.min_alpha)
        
    def forward(
        self, 
        predicted_probs: torch.Tensor,  # [batch_size, 6]
        true_winners: torch.Tensor,     # [batch_size, 6] one-hot
        market_odds: torch.Tensor       # [batch_size, 6] market odds
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate betting loss using simplified unit betting strategy
        
        Strategy: Bet 1 unit on dog with highest expected profit (ALWAYS bet on valid races)
        """
        batch_size, num_traps = predicted_probs.shape
        device = predicted_probs.device
        
        # Check for races without odds (test races) - skip them
        has_valid_odds = torch.any(market_odds > 0, dim=1)  # [batch_size]
        
        # Check for incomplete races (implied probabilities sum < 1.0)
        with torch.no_grad():
            implied_probs = 1.0 / torch.clamp(market_odds, min=1.01)
            implied_probs_sum = implied_probs.sum(dim=1)
            has_complete_field = implied_probs_sum >= 0.90  # Allow tolerance
        
        # Only consider races with valid odds AND complete field
        valid_races = has_valid_odds & has_complete_field
        
        if not valid_races.any():
            # No valid races - return basic prediction loss
            winner_indices = true_winners.argmax(dim=1)
            prediction_loss = F.cross_entropy(predicted_probs, winner_indices)
            
            return prediction_loss, {
                "expected_profit": 0.0,
                "actual_profit": 0.0,
                "ppb": 0.0,
                "roi": 0.0,
                "hit_rate": 0.0,
                "num_bets": 0,
                "num_valid_races": has_valid_odds.sum().item(),
                "num_complete_races": 0,
                "num_incomplete_races": has_valid_odds.sum().item(),
                "prediction_loss": prediction_loss.item(),
                "betting_loss": 0.0,
                "current_alpha": self.alpha
            }
        
        # Calculate expected profits for each dog (MUST BE DIFFERENTIABLE!)
        expected_profits_per_dog = torch.zeros_like(predicted_probs)
        
        # Only calculate for valid races - KEEP GRADIENTS FLOWING
        valid_mask = valid_races.unsqueeze(1).expand_as(predicted_probs)
        # Simplified expected profit: (odds * alpha * prob - 1) for unit betting
        expected_profits_per_dog[valid_mask] = (
            market_odds[valid_mask] * self.alpha * predicted_probs[valid_mask] - 1.0
        ) * (1 - self.commission)
        
        # SIMPLIFIED SOFT SELECTION: Use temperature-scaled softmax (more stable than Gumbel-Softmax)
        temperature = 0.1
        
        # Calculate soft selection weights for all races
        if valid_races.any():
            # Scale expected profits by temperature for soft selection
            scaled_logits = expected_profits_per_dog / temperature
            
            # Apply softmax only to valid races
            soft_selection = torch.zeros_like(predicted_probs)
            
            # For valid races, use softmax selection
            valid_scaled_logits = scaled_logits[valid_races]
            valid_soft_selection = F.softmax(valid_scaled_logits, dim=1)
            soft_selection[valid_races] = valid_soft_selection
        else:
            # No valid races - uniform selection
            soft_selection = torch.ones_like(predicted_probs) / num_traps
        
        # Calculate soft expected profit (differentiable)
        soft_expected_profits = (soft_selection * expected_profits_per_dog).sum(dim=1)  # [batch_size]
        
        # Hard selection for actual betting simulation (non-differentiable path)
        with torch.no_grad():
            best_dog_indices = torch.argmax(expected_profits_per_dog, dim=1)
            
            # Calculate actual outcomes for valid races only
            actual_profits = torch.zeros(batch_size, device=device)
            num_wins = 0
            num_bets = 0
            
            for i in range(batch_size):
                if valid_races[i]:
                    dog_idx = best_dog_indices[i]
                    num_bets += 1
                    
                    # Unit betting: always bet 1 unit per race
                    bet_amount = 1.0
                    
                    # Check if this dog won
                    won = true_winners[i, dog_idx] > 0.5
                    if won:
                        # Profit = (odds * bet_amount) - bet_amount = bet_amount * (odds - 1)
                        profit = bet_amount * (market_odds[i, dog_idx] - 1.0) * (1 - self.commission)
                        actual_profits[i] = profit
                        num_wins += 1
                    else:
                        # Loss = -bet_amount
                        actual_profits[i] = -bet_amount
            
            # Update cumulative tracking (unit betting)
            batch_profit = actual_profits[valid_races].sum().item() if valid_races.any() else 0.0
            batch_bets = valid_races.sum().item()
            
            # Update totals
            self.total_profit += batch_profit
            self.total_races += batch_bets
            
            # Update alpha based on batch results
            hit_rate = num_wins / max(num_bets, 1) if num_bets > 0 else 0.0
            self._update_dynamic_alpha_by_rate(hit_rate, batch_profit / max(num_bets, 1) if num_bets > 0 else 0.0)
        
        # SIMPLIFIED LOSS CALCULATION - FOCUS ON PROFITABILITY
        # 1. Prediction accuracy loss (for general learning)
        if valid_races.any():
            valid_predicted_probs = predicted_probs[valid_races]
            valid_true_winners = true_winners[valid_races]
            valid_winner_indices = valid_true_winners.argmax(dim=1)
            prediction_loss = F.cross_entropy(valid_predicted_probs, valid_winner_indices)
        else:
            prediction_loss = F.cross_entropy(predicted_probs, true_winners.argmax(dim=1))
        
        # 2. PROFIT MAXIMIZATION LOSS - Direct optimization of expected profits
        if valid_races.any() and soft_expected_profits.numel() > 0:
            # Calculate mean expected profit for valid races
            valid_soft_profits = soft_expected_profits[valid_races]
            
            # Clamp to prevent extreme values
            valid_soft_profits = torch.clamp(valid_soft_profits, min=-10.0, max=10.0)
            
            # Direct profit loss: maximize expected profit
            profit_loss = -valid_soft_profits.mean()
            
            # Check for NaN/inf and use fallback
            if torch.isfinite(profit_loss):
                betting_loss = profit_loss
            else:
                betting_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            betting_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 3. AGGRESSIVE WEIGHTING - Focus heavily on profit
        loss = 0.3 * prediction_loss + 0.7 * betting_loss
        
        # Final safety check
        if not torch.isfinite(loss) or torch.isnan(loss):
            print(f"⚠️ Loss is not finite! Using prediction loss only. pred_loss: {prediction_loss.item()}, betting_loss: {betting_loss.item()}")
            loss = prediction_loss
        
        # Calculate metrics
        with torch.no_grad():
            num_valid_bets = valid_races.sum().item()
            total_actual_profit = actual_profits[valid_races].sum().item() if num_valid_bets > 0 else 0.0
            
            # PPB: Profit Per Bet (for valid races only)
            ppb = total_actual_profit / max(num_valid_bets, 1)
            
            # ROI: Return on Investment (profit / amount bet)
            total_bet_amount = num_valid_bets * 1.0  # Unit betting
            roi = total_actual_profit / max(total_bet_amount, 1e-8)
            
            # Hit rate: calculate from actual profits (wins vs losses)
            hit_rate = num_wins / max(num_bets, 1) if num_bets > 0 else 0.0
            
            metrics = {
                "expected_profit": soft_expected_profits[valid_races].sum().item() / max(num_valid_bets, 1) if num_valid_bets > 0 else 0.0,
                "actual_profit": ppb,  # Same as PPB for unit betting
                "ppb": ppb,
                "roi": roi,
                "hit_rate": hit_rate,
                "num_bets": num_valid_bets,
                "num_valid_races": has_valid_odds.sum().item(),
                "num_complete_races": valid_races.sum().item(),
                "num_incomplete_races": has_valid_odds.sum().item() - valid_races.sum().item(),
                "prediction_loss": prediction_loss.item(),
                "betting_loss": betting_loss.item(),
                "current_alpha": self.alpha,
                "total_balance": 1000.0 + self.total_profit,  # Running total balance
                "batch_profit": batch_profit,  # Profit from this batch only
                "total_profit": self.total_profit,  # Total cumulative profit
                "avg_bet_size": 1.0  # Unit betting
            }
        
        return loss, metrics
    

def collate_race_batch(batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to convert list of race samples into batched tensors
    """
    batch_size = len(batch_list)
    
    # Initialize tensors
    race_features = torch.zeros(batch_size, 8)  # 5 numeric + 3 categorical indices
    dog_features = torch.zeros(batch_size, 6, 3)  # 6 traps, 3 static features each
    history_features = torch.zeros(batch_size, 6, 10, 9)  # 6 traps, 10 races, 9 features (4 + 5 commentary)
    targets = torch.zeros(batch_size, 6)
    market_odds = torch.zeros(batch_size, 6)  # Market odds for each trap
    
    for i, sample in enumerate(batch_list):
        # Race features: 5 numerical + 3 categorical
        rf = sample["race_features"]
        race_features[i] = torch.tensor([
            rf["day_of_week"], rf["month"], rf["hour"], rf["minute"], rf["distance_norm"],
            rf["track_idx"], rf["class_idx"], rf["category_idx"]
        ])
        
        # Clamp categorical indices to prevent IndexError
        race_features[i, 5] = torch.clamp(race_features[i, 5], 0, 999)  # track_idx  
        race_features[i, 6] = torch.clamp(race_features[i, 6], 0, 999)  # class_idx
        race_features[i, 7] = torch.clamp(race_features[i, 7], 0, 999)  # category_idx
        
        # Dog features and history
        for trap_idx in range(6):
            if trap_idx < len(sample["dog_features"]):
                df = sample["dog_features"][trap_idx]
                
                # Static dog features
                dog_features[i, trap_idx] = torch.tensor([
                    df["weight_norm"], df["trainer_idx"], df["dog_id_hash"]
                ])
                
                # History features
                hist = df["history"]
                for race_idx in range(10):  # max_history_length
                    if race_idx < len(hist["positions"]):
                        # Flatten commentary indices for this race
                        comment_flat = hist["commentary_indices"][race_idx][:5]  # Take first 5 tags
                        while len(comment_flat) < 5:
                            comment_flat.append(0)
                        
                        history_features[i, trap_idx, race_idx] = torch.tensor([
                            hist["positions"][race_idx],
                            hist["distances"][race_idx], 
                            hist["times"][race_idx],
                            hist["going_indices"][race_idx]
                        ] + comment_flat)
        
        # Targets
        targets[i] = torch.tensor(sample["targets"])
        
        # Market odds (0 for test races)
        market_odds[i] = torch.tensor(sample["market_odds"])
    
    return {
        "race_features": race_features,
        "dog_features": dog_features,
        "history_features": history_features,
        "targets": targets,
        "market_odds": market_odds
    }

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
                 dynamic_alpha: bool = True, min_alpha: float = 0.95, max_alpha: float = 1.2):
        super().__init__()
        self.base_alpha = alpha
        self.alpha = alpha
        self.commission = commission
        
        # Dynamic alpha parameters
        self.dynamic_alpha = dynamic_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def _update_dynamic_alpha(self):
        """Update alpha based on recent performance"""
        if not self.dynamic_alpha or len(self.recent_results) < 50:
            return
            
        # Calculate recent hit rate (last performance_window bets)
        recent_window = self.recent_results[-self.performance_window:]
        hit_rate = sum(recent_window) / len(recent_window) if recent_window else 0.0
        
        # Adjust alpha based on performance
        # If hit rate is low, increase alpha (more optimistic)
        # If hit rate is high, decrease alpha (more conservative)
        target_hit_rate = 0.25  # Target ~25% hit rate for profitable betting
        
        if hit_rate < target_hit_rate - 0.05:  # Performing poorly
            self.alpha = min(self.max_alpha, self.alpha * 1.01)  # Increase optimism
        elif hit_rate > target_hit_rate + 0.05:  # Performing well
            self.alpha = max(self.min_alpha, self.alpha * 0.99)  # Decrease optimism
            
        # Keep alpha in bounds
        self.alpha = max(self.min_alpha, min(self.max_alpha, self.alpha))
        
    def forward(
        self, 
        predicted_probs: torch.Tensor,  # [batch_size, 6]
        true_winners: torch.Tensor,     # [batch_size, 6] one-hot
        market_odds: torch.Tensor       # [batch_size, 6] market odds
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate betting loss using fixed percentage strategy
        
        Strategy: Bet fixed percentage on dog with highest expected profit (ALWAYS bet)
        """
        batch_size, num_traps = predicted_probs.shape
        
        # Update dynamic alpha based on recent performance
        self._update_dynamic_alpha()
        
        # Check for races without odds (test races) - skip them
        has_valid_odds = torch.any(market_odds > 0, dim=1)  # [batch_size]
        
        # Check for incomplete races (implied probabilities sum < 1.0)
        # This indicates missing participants in the dataset
        implied_probs = 1.0 / torch.clamp(market_odds, min=1.01)  # Avoid division by zero
        implied_probs_sum = implied_probs.sum(dim=1)  # [batch_size]
        has_complete_field = implied_probs_sum >= 0.95  # Allow small tolerance for rounding
        
        # Only consider races with valid odds AND complete field
        valid_races = has_valid_odds & has_complete_field
        
        # Calculate expected profits for each dog (only for valid races)
        expected_profits_per_dog = torch.zeros_like(predicted_probs)
        
        # Only calculate for valid races
        valid_mask = valid_races.unsqueeze(1).expand_as(predicted_probs)
        expected_profits_per_dog[valid_mask] = (
            (market_odds[valid_mask] * self.alpha * predicted_probs[valid_mask] - 1) * 
            self.bet_percentage * (1 - self.commission)
        )
        
        # SOFT SELECTION: Use Gumbel-Softmax for differentiable selection
        temperature = 0.1  # Controls sharpness of selection (lower = more selective)
        
        # Create logits for selection (use expected profits as preference scores)
        # Add small noise to avoid identical values
        selection_logits = expected_profits_per_dog / temperature
        
        # Apply Gumbel-Softmax for soft, differentiable selection
        soft_selection = F.gumbel_softmax(selection_logits, hard=False, dim=1, tau=temperature)
        
        # Calculate soft expected profit (weighted average across all dogs)
        soft_expected_profits = (soft_selection * expected_profits_per_dog).sum(dim=1)  # [batch_size]
        
        # For actual betting decisions, still use hard selection (non-differentiable path)
        with torch.no_grad():
            best_dog_indices = torch.argmax(expected_profits_per_dog, dim=1)  # [batch_size]
            best_expected_profits = torch.gather(expected_profits_per_dog, 1, best_dog_indices.unsqueeze(1)).squeeze(1)
        
        # FORCE bet on ALL valid races (removed positive expected profit requirement)
        should_bet = valid_races  # Bet on all races with valid odds and complete field
        
        # Calculate actual outcomes
        actual_profits = torch.zeros_like(best_expected_profits)
        bet_amounts = torch.zeros_like(best_expected_profits)
        
        for i in range(batch_size):
            if should_bet[i]:
                dog_idx = best_dog_indices[i]
                bet_amount = self.bet_percentage  # Fixed percentage
                bet_amounts[i] = bet_amount
                
                # Check if this dog won
                won = true_winners[i, dog_idx] > 0.5  # Winner
                if won:
                    payout = market_odds[i, dog_idx] * self.alpha * bet_amount
                    actual_profits[i] = (payout - bet_amount) * (1 - self.commission)
                else:  # Loser
                    actual_profits[i] = -bet_amount * (1 - self.commission)
                
                # Track result for dynamic alpha adjustment
                self.recent_results.append(1 if won else 0)
                if len(self.recent_results) > self.performance_window:
                    self.recent_results.pop(0)  # Keep only recent results
        
        # Update balance tracking (only for valid races)
        valid_races_count = valid_races.sum().item()
        batch_profit = actual_profits.sum().item()
        self.total_profit += batch_profit
        self.total_races += valid_races_count  # Only count valid races
        
        # Calculate PPB (Profit Per Bet) - only for valid races
        ppb = self.total_profit / max(self.total_races, 1)
        
        # IMPROVED LOSS FUNCTION FOR BETTER BACKPROPAGATION
        # Combination of prediction accuracy and betting profitability
        
        # 1. Cross-entropy loss for prediction accuracy (strong gradients)
        if valid_races.any():
            # Only calculate prediction loss for valid races with complete fields
            valid_predicted_probs = predicted_probs[valid_races]
            valid_true_winners = true_winners[valid_races]
            
            # Convert one-hot to class indices for cross-entropy
            valid_winner_indices = valid_true_winners.argmax(dim=1)
            prediction_loss = F.cross_entropy(valid_predicted_probs, valid_winner_indices)
        else:
            # Fallback if no valid races
            prediction_loss = F.cross_entropy(predicted_probs, true_winners.argmax(dim=1))
        
        # 2. Betting profitability component (economic objective) - USING SOFT SELECTION
        if should_bet.any():
            # Use soft expected profits for differentiable gradients
            total_soft_expected_profit = soft_expected_profits[should_bet].sum()
            betting_loss = -total_soft_expected_profit / batch_size
        else:
            # Encourage confident predictions when no bets are placed
            confidence_penalty = -(predicted_probs.max(dim=1)[0]).mean() * 0.1
            betting_loss = confidence_penalty
        
        # 3. Entropy regularization to prevent overconfident predictions
        epsilon = 1e-8
        entropy_reg = -(predicted_probs * torch.log(predicted_probs + epsilon)).sum(dim=1).mean()
        
        # 4. Combined loss with adaptive weighting
        # Weight betting loss higher as training progresses
        lambda_betting = min(0.5 + (self.total_races / 10000) * 0.5, 1.0)  # 0.5 to 1.0
        lambda_entropy = 0.01
        
        loss = prediction_loss + lambda_betting * betting_loss - lambda_entropy * entropy_reg
        
        # Calculate metrics
        with torch.no_grad():
            num_bets = should_bet.sum().item()
            total_bet_amount = bet_amounts[should_bet].sum().item() if num_bets > 0 else 0.0
            roi = batch_profit / max(total_bet_amount, 1e-8)
            hit_rate = (actual_profits > 0).float().mean().item() if num_bets > 0 else 0.0
            
            # Calculate expected profit for metrics (only from actual bets)
            expected_profit_for_metrics = best_expected_profits[should_bet].sum().item() / batch_size if should_bet.any() else 0.0
            
            # Additional metrics for tracking
            num_valid_odds_races = has_valid_odds.sum().item()
            num_complete_field_races = valid_races.sum().item()
            num_incomplete_races = has_valid_odds.sum().item() - valid_races.sum().item()
            
            # Calculate recent hit rate for alpha tracking
            recent_hit_rate = sum(self.recent_results[-100:]) / max(len(self.recent_results[-100:]), 1) if self.recent_results else 0.0
            
            # Loss component tracking for debugging
            pred_loss_val = prediction_loss.item()
            betting_loss_val = betting_loss.item() if isinstance(betting_loss, torch.Tensor) else 0.0
            entropy_val = entropy_reg.item()
            
            metrics = {
                "expected_profit": expected_profit_for_metrics,
                "actual_profit": batch_profit / batch_size,
                "ppb": ppb,  # Profit Per Bet (main metric)
                "roi": roi,
                "hit_rate": hit_rate,
                "recent_hit_rate": recent_hit_rate,  # Recent 100 bet hit rate
                "num_bets": num_bets,
                "num_valid_races": num_valid_odds_races,  # Track races with odds vs test races
                "num_complete_races": num_complete_field_races,  # Races with complete fields
                "num_incomplete_races": num_incomplete_races,  # Races with missing participants
                "total_balance": self.starting_balance + self.total_profit,
                "avg_bet_size": total_bet_amount / max(num_bets, 1),
                "bet_percentage": self.bet_percentage,
                "current_alpha": self.alpha,
                "base_alpha": self.base_alpha,
                # Loss component debugging
                "prediction_loss": pred_loss_val,
                "betting_loss": betting_loss_val,
                "entropy_reg": entropy_val,
                "lambda_betting": lambda_betting
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

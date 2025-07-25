import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class GreyhoundBettingLoss(nn.Module):
    """
    Differentiable loss function for greyhound betting that combines:
    1. Profitability loss (differentiable betting using soft selection)
    2. Accuracy loss (standard cross-entropy)
    
    The key innovation is using temperature-scaled softmax to make betting 
    selection differentiable while maintaining economic logic.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 temperature: float = 1.0,
                 commission: float = 0.05,
                 profit_weight: float = 0.7,
                 accuracy_weight: float = 0.3,
                 min_expected_profit: float = 0.0):
        """
        Args:
            alpha: Confidence multiplier for model predictions vs market odds
            temperature: Controls softmax sharpness (lower = more selective)
            commission: Betting commission rate
            profit_weight: Weight for profitability loss component
            accuracy_weight: Weight for accuracy loss component  
            min_expected_profit: Minimum expected profit threshold for betting
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.commission = commission
        self.profit_weight = profit_weight
        self.accuracy_weight = accuracy_weight
        self.min_expected_profit = min_expected_profit
        
    def forward(self, 
                model_probs: torch.Tensor, 
                market_odds: torch.Tensor, 
                actual_winners: torch.Tensor,
                dog_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the loss function
        
        Args:
            model_probs: [batch_size, max_dogs] - Model predicted probabilities
            market_odds: [batch_size, max_dogs] - Market odds for each dog
            actual_winners: [batch_size, max_dogs] - One-hot encoded winners
            dog_mask: [batch_size, max_dogs] - Valid dog mask
            
        Returns:
            Dictionary containing loss components and metrics
        """
        batch_size = model_probs.size(0)
        
        # Ensure probabilities are properly normalized for valid dogs only
        masked_probs = model_probs * dog_mask.float()
        prob_sums = masked_probs.sum(dim=1, keepdim=True)
        normalized_probs = masked_probs / (prob_sums + 1e-8)
        
        # === 1. Profitability Loss (Differentiable Betting) ===
        
        # Calculate expected profits: E[profit] = odds * α * p_model - 1
        expected_profits = market_odds * self.alpha * normalized_probs - 1
        
        # Apply commission to expected profits
        expected_profits_after_commission = expected_profits * (1 - self.commission)
        
        # Only consider bets with positive expected profit
        positive_profit_mask = (expected_profits_after_commission > self.min_expected_profit) & dog_mask.bool()
        
        # If no positive expected profits, use uniform distribution over valid dogs
        has_positive_profits = positive_profit_mask.any(dim=1, keepdim=True)
        
        # Soft betting weights using temperature-scaled softmax
        # Only apply softmax to dogs with positive expected profits
        betting_logits = torch.where(
            positive_profit_mask,
            expected_profits_after_commission / self.temperature,
            torch.full_like(expected_profits_after_commission, -1e9)
        )
        
        # For races with no positive profits, fall back to uniform over valid dogs
        uniform_logits = torch.where(
            dog_mask.bool(),
            torch.zeros_like(betting_logits),
            torch.full_like(betting_logits, -1e9)
        )
        
        final_logits = torch.where(has_positive_profits, betting_logits, uniform_logits)
        betting_weights = F.softmax(final_logits, dim=1)
        
        # Calculate actual returns for each dog
        # If we bet on dog i and it wins: return = odds_i - 1, else: return = -1
        actual_returns = torch.where(
            actual_winners.bool(),
            market_odds - 1,  # Profit if this dog wins
            -torch.ones_like(market_odds)  # Loss if this dog doesn't win
        )
        
        # Apply commission to actual returns
        actual_returns_after_commission = torch.where(
            actual_winners.bool(),
            actual_returns * (1 - self.commission),
            actual_returns  # Commission doesn't apply to losses
        )
        
        # Calculate weighted returns (what we actually get based on soft betting)
        weighted_returns = (betting_weights * actual_returns_after_commission).sum(dim=1)
        
        # Profit loss is negative expected return (we want to maximize returns)
        profit_loss = -weighted_returns.mean()
        
        # === 2. Accuracy Loss (Cross-Entropy) ===
        
        # Standard cross-entropy loss for classification accuracy
        log_probs = torch.log(normalized_probs + 1e-8)
        accuracy_loss = -(actual_winners * log_probs).sum(dim=1)
        
        # Only count loss for races with valid winners
        valid_races = actual_winners.sum(dim=1) > 0
        if valid_races.any():
            accuracy_loss = accuracy_loss[valid_races].mean()
        else:
            accuracy_loss = torch.tensor(0.0, device=model_probs.device)
        
        # === 3. Combined Loss ===
        
        total_loss = (self.profit_weight * profit_loss + 
                     self.accuracy_weight * accuracy_loss)
        
        # === 4. Additional Metrics ===
        
        # Calculate metrics for monitoring
        with torch.no_grad():
            # Hit rate (fraction of winners correctly identified as highest probability)
            model_predictions = normalized_probs.argmax(dim=1)
            actual_winner_indices = actual_winners.argmax(dim=1)
            hit_rate = (model_predictions == actual_winner_indices).float()
            
            # Expected profit per bet (based on model predictions)
            expected_profit_per_bet = (betting_weights * expected_profits_after_commission).sum(dim=1)
            
            # Actual profit per bet
            actual_profit_per_bet = weighted_returns
            
            # Betting frequency (fraction of races where we would bet)
            betting_frequency = has_positive_profits.float().squeeze()
            
        return {
            'total_loss': total_loss,
            'profit_loss': profit_loss,
            'accuracy_loss': accuracy_loss,
            'betting_weights': betting_weights,
            'expected_returns': weighted_returns,
            'hit_rate': hit_rate,
            'expected_profit_per_bet': expected_profit_per_bet,
            'actual_profit_per_bet': actual_profit_per_bet,
            'betting_frequency': betting_frequency,
            'positive_profit_mask': positive_profit_mask
        }
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for annealing during training"""
        self.temperature = new_temperature
        
    def update_alpha(self, new_alpha: float):
        """Update confidence multiplier during training"""
        self.alpha = new_alpha


def hard_betting_evaluation(model_probs: torch.Tensor,
                          market_odds: torch.Tensor,
                          actual_winners: torch.Tensor,
                          dog_mask: torch.Tensor,
                          alpha: float = 1.0,
                          commission: float = 0.05,
                          min_expected_profit: float = 0.0) -> Dict[str, float]:
    """
    Evaluate model using hard betting logic (non-differentiable, for evaluation only)
    
    This function simulates actual betting behavior:
    1. Calculate expected profits for each dog
    2. Find dog with highest expected profit above threshold
    3. Place bet on that dog
    4. Calculate actual returns
    
    Args:
        model_probs: [batch_size, max_dogs] model predictions
        market_odds: [batch_size, max_dogs] market odds
        actual_winners: [batch_size, max_dogs] one-hot winners
        dog_mask: [batch_size, max_dogs] valid dog mask
        alpha: confidence multiplier
        commission: betting commission
        min_expected_profit: minimum profit threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = model_probs.device
    batch_size = model_probs.size(0)
    
    # Normalize probabilities for valid dogs
    masked_probs = model_probs * dog_mask.float()
    prob_sums = masked_probs.sum(dim=1, keepdim=True)
    normalized_probs = masked_probs / (prob_sums + 1e-8)
    
    # Calculate expected profits
    expected_profits = market_odds * alpha * normalized_probs - 1
    expected_profits_after_commission = expected_profits * (1 - commission)
    
    # Find valid betting opportunities
    valid_bets_mask = (expected_profits_after_commission > min_expected_profit) & dog_mask.bool()
    
    total_races = 0
    total_bets_placed = 0
    total_profit = 0.0
    total_wins = 0
    individual_profits = []
    
    for i in range(batch_size):
        race_valid_bets = valid_bets_mask[i]
        
        if not race_valid_bets.any():
            # No betting opportunity in this race
            individual_profits.append(0.0)
            continue
            
        # Find dog with highest expected profit
        race_expected_profits = expected_profits_after_commission[i]
        race_expected_profits_masked = torch.where(
            race_valid_bets,
            race_expected_profits,
            torch.full_like(race_expected_profits, -float('inf'))
        )
        
        best_dog_idx = race_expected_profits_masked.argmax().item()
        
        # Place bet on this dog
        total_bets_placed += 1
        
        # Check if this dog won
        if actual_winners[i, best_dog_idx] == 1:
            # Win: profit = odds - 1, minus commission
            profit = (market_odds[i, best_dog_idx].item() - 1) * (1 - commission)
            total_wins += 1
        else:
            # Loss: lose the bet
            profit = -1.0
            
        total_profit += profit
        individual_profits.append(profit)
        total_races += 1
    
    # Calculate metrics
    if total_bets_placed > 0:
        profit_per_bet = total_profit / total_bets_placed
        hit_rate = total_wins / total_bets_placed
        betting_frequency = total_bets_placed / batch_size
    else:
        profit_per_bet = 0.0
        hit_rate = 0.0
        betting_frequency = 0.0
        
    return {
        'total_profit': total_profit,
        'profit_per_bet': profit_per_bet,
        'hit_rate': hit_rate,
        'betting_frequency': betting_frequency,
        'total_bets': total_bets_placed,
        'total_races': batch_size,
        'individual_profits': individual_profits
    }

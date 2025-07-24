"""
Fixed BettingLoss class to solve the training issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


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
        
        # Calculate expected profits for each dog (only for valid races)
        expected_profits_per_dog = torch.zeros_like(predicted_probs)
        
        valid_mask = valid_races.unsqueeze(1).expand_as(predicted_probs)
        with torch.no_grad():
            # Simplified expected profit: (odds * alpha * prob - 1) for unit betting
            expected_profits_per_dog[valid_mask] = (
                market_odds[valid_mask] * self.alpha * predicted_probs[valid_mask] - 1.0
            ) * (1 - self.commission)
        
        # SOFT SELECTION: Use Gumbel-Softmax for differentiable selection
        temperature = 0.1
        selection_logits = expected_profits_per_dog / temperature
        
        # Only apply soft selection to valid races
        soft_selection = torch.zeros_like(predicted_probs)
        if valid_races.any():
            valid_logits = selection_logits[valid_races]
            valid_soft_selection = F.gumbel_softmax(valid_logits, hard=False, dim=1, tau=temperature)
            soft_selection[valid_races] = valid_soft_selection
        
        # Calculate soft expected profit (differentiable)
        soft_expected_profits = (soft_selection * expected_profits_per_dog).sum(dim=1)  # [batch_size]
        
        # Hard selection for actual betting simulation (non-differentiable path)
        with torch.no_grad():
            best_dog_indices = torch.argmax(expected_profits_per_dog, dim=1)
            
            # Calculate actual outcomes for valid races only
            actual_profits = torch.zeros(batch_size, device=device)
            bet_results = []
            
            for i in range(batch_size):
                if valid_races[i]:
                    dog_idx = best_dog_indices[i]
                    
                    # Unit betting: always bet 1 unit
                    bet_amount = 1.0
                    
                    # Check if this dog won
                    won = true_winners[i, dog_idx] > 0.5
                    if won:
                        payout = market_odds[i, dog_idx] * self.alpha * bet_amount
                        actual_profits[i] = (payout - bet_amount) * (1 - self.commission)
                    else:
                        actual_profits[i] = -bet_amount * (1 - self.commission)
                    
                    bet_results.append(1 if won else 0)
        
        # Update alpha based on batch results
        if bet_results:
            self._update_dynamic_alpha(bet_results)
        
        # LOSS CALCULATION
        # 1. Prediction accuracy loss
        if valid_races.any():
            valid_predicted_probs = predicted_probs[valid_races]
            valid_true_winners = true_winners[valid_races]
            valid_winner_indices = valid_true_winners.argmax(dim=1)
            prediction_loss = F.cross_entropy(valid_predicted_probs, valid_winner_indices)
        else:
            prediction_loss = F.cross_entropy(predicted_probs, true_winners.argmax(dim=1))
        
        # 2. Betting profitability loss (using soft selection for gradients)
        if valid_races.any():
            # Use soft expected profits for differentiable loss
            valid_soft_profits = soft_expected_profits[valid_races]
            betting_loss = -valid_soft_profits.mean()  # Maximize expected profit
        else:
            betting_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 3. Combine losses (simplified)
        loss = prediction_loss + 0.5 * betting_loss
        
        # Ensure loss is finite
        if not torch.isfinite(loss):
            loss = prediction_loss  # Fallback to prediction loss only
        
        # Calculate metrics
        with torch.no_grad():
            num_valid_bets = valid_races.sum().item()
            total_actual_profit = actual_profits[valid_races].sum().item() if num_valid_bets > 0 else 0.0
            
            # PPB: Profit Per Bet (for valid races only)
            ppb = total_actual_profit / max(num_valid_bets, 1)
            
            # ROI: Return on Investment (profit / amount bet)
            total_bet_amount = num_valid_bets * 1.0  # Unit betting
            roi = total_actual_profit / max(total_bet_amount, 1e-8)
            
            # Hit rate
            hit_rate = sum(bet_results) / max(len(bet_results), 1) if bet_results else 0.0
            
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
                "total_balance": 1000.0 + total_actual_profit,  # Simplified balance tracking
                "avg_bet_size": 1.0  # Unit betting
            }
        
        return loss, metrics

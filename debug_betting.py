"""
Debug script to understand why no bets are being placed
"""
import torch
import numpy as np

def debug_betting_logic():
    """Debug the betting logic with sample data"""
    
    # Simulate realistic model outputs (early training - mostly uniform)
    batch_size = 4
    num_dogs = 6
    
    # Early training: model outputs are nearly uniform (around 1/6 = 0.167 each)
    predicted_probs = torch.softmax(torch.randn(batch_size, num_dogs) * 0.1, dim=1)
    print("Predicted probabilities:")
    print(predicted_probs)
    print(f"Probability range: {predicted_probs.min().item():.4f} - {predicted_probs.max().item():.4f}")
    
    # Market odds (realistic range 1.5-15.0)
    base_odds = torch.full((batch_size, num_dogs), 6.0)
    noise = torch.randn(batch_size, num_dogs) * 1.5
    market_odds = torch.clamp(base_odds + noise, min=1.2, max=20.0)
    print(f"\nMarket odds:")
    print(market_odds)
    print(f"Odds range: {market_odds.min().item():.2f} - {market_odds.max().item():.2f}")
    
    # Kelly calculation
    alpha = 1.2
    commission = 0.05
    
    adjusted_odds = market_odds * alpha
    net_odds = adjusted_odds - 1
    net_odds = torch.clamp(net_odds, min=0.01, max=49.0)
    
    q = 1 - predicted_probs
    kelly_fractions = predicted_probs - q / net_odds
    kelly_fractions = torch.clamp(kelly_fractions, min=0.0, max=0.15)
    
    print(f"\nKelly fractions before masks:")
    print(kelly_fractions)
    
    # Apply masks
    valid_odds_mask = (adjusted_odds >= 1.05) & (adjusted_odds <= 50.0)
    prob_mask = predicted_probs >= 0.05
    
    print(f"\nValid odds mask (odds 1.05-50.0):")
    print(valid_odds_mask)
    print(f"\nProb mask (prob >= 5%):")
    print(prob_mask)
    
    kelly_fractions = kelly_fractions * valid_odds_mask.float() * prob_mask.float()
    print(f"\nFinal Kelly fractions:")
    print(kelly_fractions)
    
    # Expected profit calculation
    expected_profits = (market_odds * alpha * predicted_probs - 1) * kelly_fractions * (1 - commission)
    print(f"\nExpected profits per dog:")
    print(expected_profits)
    
    # Best bet selection
    best_dog_indices = torch.argmax(expected_profits, dim=1)
    best_expected_profits = torch.gather(expected_profits, 1, best_dog_indices.unsqueeze(1)).squeeze(1)
    
    print(f"\nBest expected profits:")
    print(best_expected_profits)
    
    should_bet = best_expected_profits > 0
    print(f"\nShould bet:")
    print(should_bet)
    print(f"Number of races with bets: {should_bet.sum().item()}")
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    print(f"Issue: Expected profits are all negative or zero!")
    print(f"Reason: (odds * alpha * prob - 1) is typically negative when:")
    print(f"  - odds * alpha * prob < 1")
    print(f"  - prob < 1 / (odds * alpha)")
    
    min_prob_needed = 1 / (market_odds * alpha)
    print(f"\nMinimum probability needed for positive expected value:")
    print(min_prob_needed)
    print(f"But model predicts (uniform-ish):")
    print(predicted_probs)
    
    print(f"\nThe model needs to predict higher probabilities than {min_prob_needed.min().item():.3f}")
    print(f"but it's predicting around {predicted_probs.mean().item():.3f}")

if __name__ == "__main__":
    debug_betting_logic()

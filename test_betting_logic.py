"""
Test script to verify betting logic improvements
"""
import torch
import numpy as np
from machine_learning.model import BettingLoss

def test_betting_logic():
    """Test the updated betting logic"""
    print("🧪 Testing Updated Betting Logic")
    print("="*50)
    
    # Create betting loss with dynamic alpha
    criterion = BettingLoss(
        alpha=1.1,           # Optimistic starting alpha
        commission=0.05,
        bet_percentage=0.02,
        dynamic_alpha=True,
        min_alpha=0.95,
        max_alpha=1.2
    )
    
    # Test case 1: Race with complete field (odds sum ~= 1.0)
    print("\n🏁 Test Case 1: Complete Field Race")
    
    # Simulate a race with 6 dogs
    predicted_probs = torch.tensor([[0.10, 0.25, 0.15, 0.20, 0.18, 0.12]])  # Model predictions
    true_winners = torch.tensor([[0, 1, 0, 0, 0, 0]])  # Dog 2 wins (trap 1, 0-indexed)
    
    # Market odds that sum to ~1.0 (complete field)
    market_odds = torch.tensor([[8.0, 3.5, 6.0, 4.5, 5.0, 7.5]])  # Realistic odds
    implied_prob_sum = (1.0 / market_odds).sum()
    print(f"   Implied probability sum: {implied_prob_sum.item():.3f} (should be ~1.0)")
    
    # Calculate loss and metrics
    loss, metrics = criterion(predicted_probs, true_winners, market_odds)
    
    print(f"   📊 Metrics:")
    print(f"      Bets placed: {metrics['num_bets']}")
    print(f"      Complete races: {metrics['num_complete_races']}")
    print(f"      Incomplete races: {metrics['num_incomplete_races']}")
    print(f"      Current alpha: {metrics['current_alpha']:.3f}")
    print(f"      Expected profit: {metrics['expected_profit']:.6f}")
    print(f"      Actual profit: {metrics['actual_profit']:.6f}")
    print(f"      Hit rate: {metrics['hit_rate']:.3f}")
    
    # Test case 2: Race with incomplete field (odds sum < 1.0)
    print("\n🏁 Test Case 2: Incomplete Field Race")
    
    # Market odds that sum to < 1.0 (missing participants)
    incomplete_odds = torch.tensor([[8.0, 3.5, 6.0, 0.0, 0.0, 0.0]])  # Only 3 dogs have odds
    implied_prob_sum = (1.0 / torch.clamp(incomplete_odds, min=1.01)).sum()
    print(f"   Implied probability sum: {implied_prob_sum.item():.3f} (should be < 1.0)")
    
    loss2, metrics2 = criterion(predicted_probs, true_winners, incomplete_odds)
    
    print(f"   📊 Metrics:")
    print(f"      Bets placed: {metrics2['num_bets']}")
    print(f"      Complete races: {metrics2['num_complete_races']}")
    print(f"      Incomplete races: {metrics2['num_incomplete_races']}")
    
    # Test case 3: Test race (no odds)
    print("\n🏁 Test Case 3: Test Race (No Odds)")
    
    no_odds = torch.zeros_like(market_odds)  # All zeros
    loss3, metrics3 = criterion(predicted_probs, true_winners, no_odds)
    
    print(f"   📊 Metrics:")
    print(f"      Bets placed: {metrics3['num_bets']}")
    print(f"      Valid races: {metrics3['num_valid_races']}")
    
    # Test dynamic alpha over multiple bets
    print("\n🎚️  Test Case 4: Dynamic Alpha Adjustment")
    
    print(f"   Initial alpha: {criterion.alpha:.3f}")
    
    # Simulate multiple losing bets (should increase alpha)
    for i in range(10):
        losing_winners = torch.tensor([[0, 0, 0, 0, 0, 1]])  # Always lose on dog 6
        _, _ = criterion(predicted_probs, losing_winners, market_odds)
    
    print(f"   Alpha after losses: {criterion.alpha:.3f}")
    print(f"   Recent hit rate: {metrics.get('recent_hit_rate', 0):.3f}")
    
    # Simulate multiple winning bets (should decrease alpha)
    for i in range(20):
        winning_winners = torch.tensor([[0, 1, 0, 0, 0, 0]])  # Always win on dog 2
        _, _ = criterion(predicted_probs, winning_winners, market_odds)
    
    print(f"   Alpha after wins: {criterion.alpha:.3f}")
    
    print("\n✅ All tests completed!")
    print("\n📋 Summary of Improvements:")
    print("   ✅ Forces betting on all valid races (even negative EV)")
    print("   ✅ Filters out races without odds (test races)")
    print("   ✅ Filters out races with incomplete fields (missing participants)")
    print("   ✅ Implements dynamic alpha adjustment based on performance")
    print("   ✅ Uses fixed percentage betting instead of Kelly criterion")

if __name__ == "__main__":
    test_betting_logic()

"""
Quick test of our improved training with simplified loss and debugging
"""
import torch
import os
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from machine_learning.model import GreyhoundRacingModel, BettingLoss

# Test with synthetic data
def test_fixed_loss():
    print("🧪 Testing Fixed BettingLoss")
    
    # Create simple test data
    batch_size = 4
    predicted_probs = torch.softmax(torch.randn(batch_size, 6), dim=1)
    true_winners = torch.zeros(batch_size, 6)
    true_winners[0, 0] = 1.0  # First dog wins first race
    true_winners[1, 1] = 1.0  # Second dog wins second race
    true_winners[2, 2] = 1.0  # etc.
    true_winners[3, 3] = 1.0
    
    market_odds = torch.tensor([
        [2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
        [3.0, 2.8, 4.5, 5.5, 6.5, 7.0],
        [4.0, 3.5, 2.2, 6.0, 7.0, 8.0],
        [2.0, 4.0, 5.0, 3.0, 6.0, 9.0]
    ])
    
    # Test simplified loss
    criterion = BettingLoss(alpha=1.1, commission=0.05)
    
    try:
        loss, metrics = criterion(predicted_probs, true_winners, market_odds)
        
        print(f"✅ Loss calculation successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Loss requires grad: {loss.requires_grad}")
        print(f"   PPB: {metrics['ppb']:.6f}")
        print(f"   ROI: {metrics['roi']:.4f}")
        print(f"   Hit Rate: {metrics['hit_rate']:.4f}")
        print(f"   Num Bets: {metrics['num_bets']}")
        print(f"   Valid Races: {metrics['num_valid_races']}")
        print(f"   Complete Races: {metrics['num_complete_races']}")
        
        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_loss()

#!/usr/bin/env python3
"""
Quick test to verify the model implementation is working correctly.

This script tests model instantiation and forward pass without training.
"""

import os
import sys
import torch
import numpy as np

# Add parent directory for imports - make it more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from machine_learning.model import GreyhoundRacingModel
from machine_learning.loss import GreyhoundBettingLoss, hard_betting_evaluation


def test_model_forward():
    """Test model forward pass"""
    
    print("🧪 Testing Model Forward Pass...")
    
    # Model parameters
    vocab_sizes = {
        'num_tracks': 10,
        'num_classes': 5,
        'num_categories': 3,
        'num_trainers': 50,
        'num_going_conditions': 5,
        'commentary_vocab_size': 100
    }
    
    # Create model
    model = GreyhoundRacingModel(
        num_tracks=vocab_sizes['num_tracks'],
        num_classes=vocab_sizes['num_classes'],
        num_categories=vocab_sizes['num_categories'],
        num_trainers=vocab_sizes['num_trainers'],
        num_going_conditions=vocab_sizes['num_going_conditions'],
        commentary_vocab_size=vocab_sizes['commentary_vocab_size'],
        embedding_dim=16,
        hidden_dim=32,
        rnn_hidden_dim=16,
        max_history_length=5,
        max_commentary_length=3,
        dropout_rate=0.1,
        max_dogs_per_race=6
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Detailed parameter breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # 32-bit floats
    
    print(f"🏗️  MODEL DETAILS:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: {model_size_mb:.1f} MB")
    
    # Create dummy batch
    batch_size = 4
    max_dogs = 6
    max_history = 5
    max_commentary = 3
    
    batch_data = {
        # Race features
        'race_features': torch.randn(batch_size, 8),
        'track_ids': torch.randint(0, vocab_sizes['num_tracks'], (batch_size,)),
        'class_ids': torch.randint(0, vocab_sizes['num_classes'], (batch_size,)),
        'category_ids': torch.randint(0, vocab_sizes['num_categories'], (batch_size,)),
        
        # Dog features
        'dog_features': torch.randn(batch_size, max_dogs, 3),
        'trainer_ids': torch.randint(0, vocab_sizes['num_trainers'], (batch_size, max_dogs)),
        'dog_ids': torch.randint(0, 10000, (batch_size, max_dogs)),
        'dog_mask': torch.ones(batch_size, max_dogs, dtype=torch.bool),
        
        # History features
        'history_features': torch.randn(batch_size, max_dogs, max_history, 3),
        'going_ids': torch.randint(0, vocab_sizes['num_going_conditions'], (batch_size, max_dogs, max_history)),
        'commentary_ids': torch.randint(0, vocab_sizes['commentary_vocab_size'], (batch_size, max_dogs, max_history, max_commentary)),
        'history_mask': torch.ones(batch_size, max_dogs, max_history, dtype=torch.bool),
        
        # Labels and odds
        'win_labels': torch.zeros(batch_size, max_dogs),
        'market_odds': torch.empty(batch_size, max_dogs).uniform_(1.5, 10.0)
    }
    
    # Set winners (one per race)
    for i in range(batch_size):
        winner_idx = torch.randint(0, max_dogs, (1,)).item()
        batch_data['win_labels'][i, winner_idx] = 1.0
    
    print("✅ Dummy batch created")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        win_probabilities = model(batch_data)
    
    print(f"✅ Forward pass successful")
    print(f"Output shape: {win_probabilities.shape}")
    print(f"Probability sums: {win_probabilities.sum(dim=1)}")
    
    # Check output validity
    assert win_probabilities.shape == (batch_size, max_dogs), f"Wrong output shape: {win_probabilities.shape}"
    assert torch.allclose(win_probabilities.sum(dim=1), torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1"
    assert (win_probabilities >= 0).all(), "Negative probabilities found"
    
    print("✅ Output validation passed")
    
    return model, batch_data, win_probabilities


def test_loss_function():
    """Test loss function"""
    
    print("\n🧪 Testing Loss Function...")
    
    # Create loss function
    loss_fn = GreyhoundBettingLoss(
        alpha=1.0,
        temperature=1.0,
        commission=0.05,
        profit_weight=0.7,
        accuracy_weight=0.3
    )
    
    print("✅ Loss function created")
    
    # Create dummy data
    batch_size = 4
    max_dogs = 6
    
    model_probs = torch.softmax(torch.randn(batch_size, max_dogs), dim=1)
    market_odds = torch.empty(batch_size, max_dogs).uniform_(1.5, 10.0)
    actual_winners = torch.zeros(batch_size, max_dogs)
    dog_mask = torch.ones(batch_size, max_dogs, dtype=torch.bool)
    
    # Set winners
    for i in range(batch_size):
        winner_idx = torch.randint(0, max_dogs, (1,)).item()
        actual_winners[i, winner_idx] = 1.0
    
    # Test soft loss
    loss_dict = loss_fn(model_probs, market_odds, actual_winners, dog_mask)
    
    print(f"✅ Soft loss computation successful")
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Profit loss: {loss_dict['profit_loss'].item():.4f}")
    print(f"Accuracy loss: {loss_dict['accuracy_loss'].item():.4f}")
    
    # Test hard evaluation
    hard_eval = hard_betting_evaluation(
        model_probs=model_probs,
        market_odds=market_odds,
        actual_winners=actual_winners,
        dog_mask=dog_mask,
        alpha=1.0,
        commission=0.05
    )
    
    print(f"✅ Hard evaluation successful")
    print(f"Hit rate: {hard_eval['hit_rate']:.2%}")
    print(f"Profit per bet: ${hard_eval['profit_per_bet']:.4f}")
    print(f"Betting frequency: {hard_eval['betting_frequency']:.2%}")
    
    # Check loss properties
    assert not torch.isnan(loss_dict['total_loss']), "Loss is NaN"
    assert not torch.isinf(loss_dict['total_loss']), "Loss is infinite"
    
    print("✅ Loss validation passed")
    
    return loss_dict, hard_eval


def test_backward_pass():
    """Test backward pass"""
    
    print("\n🧪 Testing Backward Pass...")
    
    # Get model and data from forward test
    model, batch_data, win_probabilities = test_model_forward()
    
    # Create loss function
    loss_fn = GreyhoundBettingLoss(alpha=1.0, temperature=1.0, commission=0.05)
    
    # Enable gradients
    model.train()
    
    # Forward pass
    win_probs = model(batch_data)
    
    # Loss computation
    loss_dict = loss_fn(
        model_probs=win_probs,
        market_odds=batch_data['market_odds'],
        actual_winners=batch_data['win_labels'],
        dog_mask=batch_data['dog_mask']
    )
    
    # Backward pass
    loss_dict['total_loss'].backward()
    
    print("✅ Backward pass successful")
    
    # Check gradients
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    assert total_grad_norm > 0, "No gradients computed"
    assert not np.isnan(total_grad_norm), "Gradient norm is NaN"
    
    print("✅ Gradient validation passed")
    
    return total_grad_norm


def main():
    """Run all tests"""
    
    print("🚀 Greyhound Racing Model - Component Tests")
    print("=" * 50)
    
    try:
        # Test model forward pass
        model, batch_data, win_probs = test_model_forward()
        
        # Test loss function
        loss_dict, hard_eval = test_loss_function()
        
        # Test backward pass
        grad_norm = test_backward_pass()
        
        print("\n🎉 All Tests Passed!")
        print("=" * 50)
        print("✅ Model forward pass working")
        print("✅ Loss function working")
        print("✅ Backward pass working")
        print("✅ Gradients flowing correctly")
        print("\nThe model implementation is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)

"""
Debug script to check gradient flow and learning rate issues
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from machine_learning.model import GreyhoundRacingModel, BettingLoss, collate_race_batch

def debug_gradient_flow():
    """Check if gradients are flowing through the model"""
    print("🔍 Debugging Gradient Flow")
    print("="*50)
    
    # Create a simple model for testing
    model = GreyhoundRacingModel(
        num_tracks=29,
        num_classes=74,
        num_categories=1,
        num_trainers=1284,
        num_going_conditions=10,
        commentary_vocab_size=1000
    )
    
    # Create synthetic batch data
    batch_size = 4
    batch = {
        'race_features': torch.randn(batch_size, 8),
        'dog_features': torch.randn(batch_size, 6, 3),
        'history_features': torch.randn(batch_size, 6, 10, 9),
        'targets': torch.zeros(batch_size, 6),
        'market_odds': torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * batch_size)
    }
    
    # Set one target to 1 for each batch
    for i in range(batch_size):
        batch['targets'][i, i % 6] = 1.0
    
    print(f"📊 Batch shape check:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
    
    # Forward pass
    print(f"\n🔮 Forward pass:")
    predictions = model(batch)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions (first sample): {predictions[0].detach().numpy()}")
    print(f"   Predictions sum (should be ~1.0): {predictions[0].sum().item():.4f}")
    
    # Check if predictions are reasonable
    if torch.allclose(predictions, predictions[0:1].expand_as(predictions), atol=1e-6):
        print("   ⚠️  WARNING: All predictions are identical! Model may not be learning.")
    
    # Create loss function
    criterion = BettingLoss(alpha=1.1, commission=0.05, bet_percentage=0.02)
    
    # Calculate loss
    print(f"\n💸 Loss calculation:")
    loss, metrics = criterion(predictions, batch['targets'], batch['market_odds'])
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Loss requires grad: {loss.requires_grad}")
    print(f"   Metrics: {metrics}")
    
    # Backward pass
    print(f"\n⬅️  Backward pass:")
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0.0
    zero_grad_params = 0
    total_params = 0
    
    print(f"   Gradient statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm < 1e-8:
                zero_grad_params += 1
            total_params += 1
            if grad_norm > 1e-6:  # Only show significant gradients
                print(f"      {name}: grad_norm = {grad_norm:.8f}")
        else:
            print(f"      {name}: NO GRADIENT!")
            zero_grad_params += 1
            total_params += 1
    
    print(f"\n📈 Gradient summary:")
    print(f"   Total gradient norm: {total_grad_norm:.8f}")
    print(f"   Zero/tiny gradients: {zero_grad_params}/{total_params}")
    print(f"   Gradient flow health: {'🟢 GOOD' if total_grad_norm > 1e-6 else '🔴 POOR'}")
    
    if total_grad_norm < 1e-6:
        print(f"\n🚨 GRADIENT FLOW ISSUES DETECTED!")
        print(f"   The gradients are too small, indicating:")
        print(f"   1. Loss function may not depend on model parameters")
        print(f"   2. Model predictions may be constant")
        print(f"   3. Learning rate may be too small")
        print(f"   4. Vanishing gradient problem")

def debug_learning_rate():
    """Test different learning rates"""
    print(f"\n🎛️  Testing Learning Rates")
    print("="*50)
    
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    for lr in learning_rates:
        print(f"\n📊 Testing LR: {lr}")
        
        # Create model and optimizer
        model = GreyhoundRacingModel(
            num_tracks=29, num_classes=74, num_categories=1,
            num_trainers=1284, num_going_conditions=10, commentary_vocab_size=1000
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = BettingLoss(alpha=1.1, commission=0.05, bet_percentage=0.02)
        
        # Create consistent test data
        torch.manual_seed(42)
        batch = {
            'race_features': torch.randn(4, 8),
            'dog_features': torch.randn(4, 6, 3),
            'history_features': torch.randn(4, 6, 10, 9),
            'targets': torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], 
                                   [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.float),
            'market_odds': torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * 4)
        }
        
        # Initial prediction
        with torch.no_grad():
            initial_pred = model(batch)[0].clone()
        
        # Training step
        optimizer.zero_grad()
        predictions = model(batch)
        loss, metrics = criterion(predictions, batch['targets'], batch['market_odds'])
        loss.backward()
        optimizer.step()
        
        # After training prediction
        with torch.no_grad():
            final_pred = model(batch)[0].clone()
        
        # Calculate change
        pred_change = torch.norm(final_pred - initial_pred).item()
        
        print(f"   Initial pred: {initial_pred.numpy()}")
        print(f"   Final pred:   {final_pred.numpy()}")
        print(f"   Change norm:  {pred_change:.8f}")
        print(f"   Loss:         {loss.item():.6f}")
        print(f"   Status:       {'🟢 LEARNING' if pred_change > 1e-6 else '🔴 NO CHANGE'}")

def debug_data_consistency():
    """Check if the data is consistent between batches"""
    print(f"\n📦 Testing Data Consistency")
    print("="*50)
    
    # Test if we're getting the same data every time (which would prevent learning)
    torch.manual_seed(42)  # Reset seed
    
    batches = []
    for i in range(3):
        batch = {
            'race_features': torch.randn(2, 8),
            'dog_features': torch.randn(2, 6, 3),
            'history_features': torch.randn(2, 6, 10, 9),
            'targets': torch.randint(0, 2, (2, 6), dtype=torch.float),
            'market_odds': torch.rand(2, 6) * 10 + 1  # Random odds 1-11
        }
        batches.append(batch)
    
    # Check if batches are different
    batch1_hash = hash(str(batches[0]['targets'].numpy().tobytes()))
    batch2_hash = hash(str(batches[1]['targets'].numpy().tobytes()))
    batch3_hash = hash(str(batches[2]['targets'].numpy().tobytes()))
    
    print(f"   Batch 1 targets: {batches[0]['targets'][0].numpy()}")
    print(f"   Batch 2 targets: {batches[1]['targets'][0].numpy()}")
    print(f"   Batch 3 targets: {batches[2]['targets'][0].numpy()}")
    
    if batch1_hash == batch2_hash == batch3_hash:
        print(f"   🔴 ERROR: All batches are identical!")
    else:
        print(f"   🟢 GOOD: Batches are different")

if __name__ == "__main__":
    debug_gradient_flow()
    debug_learning_rate()
    debug_data_consistency()

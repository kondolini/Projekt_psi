"""
Evaluation script for Greyhound Racing Model
Calculates betting performance metrics on test data
"""

import torch
import pickle
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List
import argparse

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from machine_learning.model import GreyhoundRacingModel, collate_race_batch
from machine_learning.data_processor import RaceDataProcessor
from machine_learning.train import load_data, create_dataset, RaceDataset
from torch.utils.data import DataLoader


def kelly_criterion(predicted_prob: float, market_odds: float, alpha: float = 0.95, max_bet: float = 0.25) -> float:
    """
    Calculate Kelly criterion bet size
    
    Args:
        predicted_prob: Model's predicted win probability
        market_odds: Market odds (decimal)
        alpha: Odds reduction factor
        max_bet: Maximum fraction of bankroll to bet
    
    Returns:
        Optimal bet size as fraction of bankroll
    """
    adjusted_odds = market_odds * alpha
    net_odds = adjusted_odds - 1
    
    kelly_fraction = predicted_prob - (1 - predicted_prob) / net_odds
    
    # Only bet when Kelly is positive and odds are reasonable
    if kelly_fraction > 0 and 1.1 <= adjusted_odds <= 50.0:
        return min(kelly_fraction, max_bet)
    else:
        return 0.0


def evaluate_model(
    model: GreyhoundRacingModel,
    test_loader: DataLoader,
    device: torch.device,
    alpha: float = 0.95,
    commission: float = 0.05
) -> Dict:
    """
    Evaluate model on test data with betting simulation
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    betting_results = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions
            predictions = model(batch)  # [batch_size, 6]
            targets = batch["targets"]  # [batch_size, 6]
            
            # Convert to numpy for easier processing
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            # Store for overall metrics
            all_predictions.append(pred_np)
            all_targets.append(target_np)
            
            # Simulate betting for each race in batch
            batch_size = pred_np.shape[0]
            
            for race_idx in range(batch_size):
                race_preds = pred_np[race_idx]  # [6]
                race_targets = target_np[race_idx]  # [6]
                
                # Simulate market odds (in practice, extract from race data)
                # For now, use inverse of uniform probability with some noise
                base_odds = 6.0 / race_preds  # If all equal, odds would be 6.0
                market_odds = base_odds + np.random.normal(0, 0.5, 6)  # Add noise
                market_odds = np.clip(market_odds, 1.1, 50.0)  # Reasonable range
                
                # Calculate Kelly bets
                kelly_bets = []
                for trap_idx in range(6):
                    bet_size = kelly_criterion(race_preds[trap_idx], market_odds[trap_idx], alpha)
                    kelly_bets.append(bet_size)
                
                kelly_bets = np.array(kelly_bets)
                
                # Calculate betting outcomes
                total_bet = kelly_bets.sum()
                
                if total_bet > 0:
                    # Expected value
                    expected_values = (market_odds * alpha * race_preds - 1) * kelly_bets * (1 - commission)
                    expected_profit = expected_values.sum()
                    
                    # Actual outcome
                    actual_outcomes = (race_targets * market_odds * alpha - 1) * kelly_bets * (1 - commission)
                    actual_profit = actual_outcomes.sum()
                    
                    betting_results.append({
                        'expected_profit': expected_profit,
                        'actual_profit': actual_profit,
                        'total_bet': total_bet,
                        'won': actual_profit > 0
                    })
    
    # Combine all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate classification metrics
    # Convert to binary classification (win/lose for each trap)
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    
    # Accuracy using top prediction per race
    race_level_accuracy = []
    num_races = all_predictions.shape[0]
    
    for i in range(num_races):
        predicted_winner = np.argmax(all_predictions[i])
        actual_winner = np.argmax(all_targets[i])
        race_level_accuracy.append(predicted_winner == actual_winner)
    
    accuracy = np.mean(race_level_accuracy)
    
    # Betting performance metrics
    if betting_results:
        total_expected_profit = sum(r['expected_profit'] for r in betting_results)
        total_actual_profit = sum(r['actual_profit'] for r in betting_results)
        total_bet_amount = sum(r['total_bet'] for r in betting_results)
        
        roi = total_actual_profit / total_bet_amount if total_bet_amount > 0 else 0
        hit_rate = np.mean([r['won'] for r in betting_results])
        
        # Profit per bet
        ppb = total_actual_profit / len(betting_results) if betting_results else 0
        
        # Sharpe ratio (risk-adjusted return)
        profits = [r['actual_profit'] for r in betting_results]
        sharpe_ratio = np.mean(profits) / (np.std(profits) + 1e-8)
        
        metrics = {
            'accuracy': accuracy,
            'total_races': num_races,
            'betting_races': len(betting_results),
            'total_expected_profit': total_expected_profit,
            'total_actual_profit': total_actual_profit,
            'total_bet_amount': total_bet_amount,
            'roi': roi,
            'hit_rate': hit_rate,
            'profit_per_bet': ppb,
            'sharpe_ratio': sharpe_ratio,
            'avg_bet_size': np.mean([r['total_bet'] for r in betting_results]) if betting_results else 0
        }
    else:
        metrics = {
            'accuracy': accuracy,
            'total_races': num_races,
            'betting_races': 0,
            'error': 'No betting opportunities found'
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Greyhound Racing Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='machine_learning/outputs', help='Model directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_split', type=str, default='2023-01-01', help='Test split date')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data processor
    processor_path = os.path.join(args.model_dir, 'data_processor.pkl')
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Data processor not found: {processor_path}")
    
    processor = RaceDataProcessor()
    processor.load_encoders(processor_path)
    
    # Load data
    print("Loading test data...")
    dogs, train_races, test_races = load_data(args.data_dir, args.test_split)
    
    # Create test dataset
    test_dataset = create_dataset(test_races, dogs, processor)
    test_loader = DataLoader(
        RaceDataset(test_dataset), 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_race_batch
    )
    
    # Load model
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = GreyhoundRacingModel(
        num_tracks=len(processor.track_encoder),
        num_classes=len(processor.class_encoder),
        num_categories=len(processor.category_encoder),
        num_trainers=len(processor.trainer_encoder),
        num_going_conditions=len(processor.going_encoder),
        commentary_vocab_size=processor.commentary_processor.vocab_size
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Race Prediction Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Races Evaluated: {metrics['total_races']}")
    
    if 'betting_races' in metrics:
        print(f"Races with Betting Opportunities: {metrics['betting_races']}")
        print(f"Total Expected Profit: ${metrics['total_expected_profit']:.2f}")
        print(f"Total Actual Profit: ${metrics['total_actual_profit']:.2f}")
        print(f"Total Amount Bet: ${metrics['total_bet_amount']:.2f}")
        print(f"Return on Investment (ROI): {metrics['roi']:.4f} ({metrics['roi']*100:.2f}%)")
        print(f"Hit Rate: {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.2f}%)")
        print(f"Profit Per Bet: ${metrics['profit_per_bet']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Average Bet Size: {metrics['avg_bet_size']:.4f}")
    
    # Save results
    results_path = os.path.join(args.model_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

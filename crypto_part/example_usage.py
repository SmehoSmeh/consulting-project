#!/usr/bin/env python3
"""
Example Usage Script for Improved DeFi Yield Prediction

This script demonstrates how to use the improved XGBoost model with custom metrics
and comprehensive grid search.

Usage:
1. Update the data_path variable with your actual data file path
2. Run the script: python example_usage.py

The script will:
- Load your DeFi data
- Apply custom feature engineering
- Train XGBoost with grid search using your specified metrics
- Evaluate and compare model performance
- Save the best model
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'source_data', 'defilama_data.json')

from main_training_improved import main_training_pipeline, run_experiment_comparison

def run_single_model_example():
    """
    Example of training a single XGBoost model with grid search
    """
    print("=== Single Model Training Example ===")
    
    # Update this path to your actual data file
    data_path = DEFAULT_DATA_FILE  # Replace with your data file
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file path")
        return None
    
    # Run the training pipeline with grid search
    results = main_training_pipeline(
        data_path=data_path,
        use_ensemble=False,
        search_type='grid'  # Use 'randomized' for faster training
    )
    
    if results is not None:
        print("\n=== Training Completed Successfully ===")
        print(f"Best model R² score: {results['metrics']['r2']:.4f}")
        print(f"Best model RMSE: {results['metrics']['rmse']:.4f}")
        
        # Show top features
        if results['importance_df'] is not None:
            print("\nTop 5 most important features:")
            print(results['importance_df'].head())
        
        return results
    else:
        print("Training failed. Please check your data file and try again.")
        return None

def run_ensemble_model_example():
    """
    Example of training an ensemble model
    """
    print("=== Ensemble Model Training Example ===")
    
    # Update this path to your actual data file
    data_path = DEFAULT_DATA_FILE  # Replace with your data file
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file path")
        return None
    
    # Run the training pipeline with ensemble
    results = main_training_pipeline(
        data_path=data_path,
        use_ensemble=True,
        search_type='randomized'  # Randomized search is faster for ensemble
    )
    
    if results is not None:
        print("\n=== Ensemble Training Completed Successfully ===")
        print(f"Ensemble model R² score: {results['metrics']['r2']:.4f}")
        print(f"Ensemble model RMSE: {results['metrics']['rmse']:.4f}")
        
        # Show prediction uncertainty
        if 'pred_std' in results:
            avg_uncertainty = np.mean(results['pred_std'])
            print(f"Average prediction uncertainty: {avg_uncertainty:.4f}")
        
        return results
    else:
        print("Ensemble training failed. Please check your data file and try again.")
        return None

def compare_different_approaches():
    """
    Example of comparing different modeling approaches
    """
    print("=== Comparing Different Approaches ===")
    
    # Update this path to your actual data file
    data_path = DEFAULT_DATA_FILE  # Replace with your data file
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file path")
        return None
    
    approaches = {
        "Grid Search": {"search_type": "grid", "use_ensemble": False},
        "Randomized Search": {"search_type": "randomized", "use_ensemble": False},
        "Ensemble Model": {"search_type": "randomized", "use_ensemble": True}
    }
    
    results = {}
    
    for name, params in approaches.items():
        print(f"\n--- Training {name} ---")
        try:
            result = main_training_pipeline(
                data_path=data_path,
                **params
            )
            results[name] = result
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = None
    
    # Compare results
    print("\n=== Comparison Results ===")
    comparison_data = []
    
    for name, result in results.items():
        if result is not None:
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['R²'].idxmax()]
        print(f"\nBest performing model: {best_model['Model']}")
        print(f"R² Score: {best_model['R²']:.4f}")
    
    return results

def demo_with_sample_data():
    """
    Demo function that creates sample data for testing
    """
    print("=== Creating Sample Data for Demo ===")
    
    # Create sample DeFi data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'project': np.random.choice(['Uniswap', 'Curve', 'Balancer', 'SushiSwap'], n_samples),
        'symbol': np.random.choice(['ETH-USDC', 'WBTC-ETH', 'DAI-USDT', 'LINK-ETH'], n_samples),
        'chain': np.random.choice(['ethereum', 'bsc', 'polygon', 'arbitrum'], n_samples),
        'tvlUsd': np.random.lognormal(15, 2, n_samples),  # Log-normal distribution for TVL
        'volumeUsd1d': np.random.lognormal(12, 2, n_samples),
        'sigma': np.random.gamma(2, 0.1, n_samples),  # Volatility
        'mu': np.random.normal(0.05, 0.02, n_samples),  # Mean return
        'stablecoin': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'exposure': np.random.choice(['single', 'multi'], n_samples),
        'ilRisk': np.random.choice(['yes', 'no'], n_samples),
        'underlyingTokens': np.random.choice(['ETH,USDC', 'WBTC', 'DAI,USDT'], n_samples),
        'rewardTokens': np.random.choice(['NONE', 'UNI', 'CRV,CVX'], n_samples),
        'pool': [f'pool_{i}' for i in range(n_samples)],
        'percentage_of_total_tvl': np.random.exponential(0.1, n_samples),
        'outlier': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    # Create target variable (APY) based on features
    apy_base = (
        2 + 
        sample_data['mu'] * 100 +  # Base return
        sample_data['stablecoin'] * (-1) +  # Stablecoins have lower APY
        np.log(sample_data['tvlUsd']) * 0.1 +  # TVL effect
        np.random.normal(0, 1, n_samples)  # Noise
    )
    
    sample_data['apy'] = np.clip(apy_base, 0.1, 50)  # Clip to reasonable range
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    sample_path = 'sample_defi_data.json'
    df.to_json(sample_path, orient='records')
    print(f"Sample data saved to {sample_path}")
    
    # Run training on sample data
    print("\n=== Training on Sample Data ===")
    results = main_training_pipeline(
        data_path=sample_path,
        use_ensemble=False,
        search_type='grid'
    )
    
    return results, sample_path

if __name__ == "__main__":
    print("DeFi Yield Prediction - Example Usage")
    print("=====================================")
    
    print("\nAvailable examples:")
    print("1. Single model with grid search")
    print("2. Ensemble model training")
    print("3. Compare different approaches")
    print("4. Demo with sample data")
    
    choice = input("\nEnter your choice (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        run_single_model_example()
    elif choice == '2':
        run_ensemble_model_example()
    elif choice == '3':
        compare_different_approaches()
    elif choice == '4':
        demo_with_sample_data()
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice. Running demo with sample data...")
        demo_with_sample_data()
    
    print("\nExample completed!") 
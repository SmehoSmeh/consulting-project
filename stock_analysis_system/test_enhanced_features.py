#!/usr/bin/env python3
"""
Quick test for enhanced stock analysis features
"""

import sys
import os
import pandas as pd
import numpy as np

# Add core directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def test_enhanced_features():
    """Test the new enhanced features"""
    print("="*60)
    print("🧪 TESTING ENHANCED FEATURES")
    print("="*60)
    
    # Test 1: Load data processing functions
    print("\n1️⃣ Testing Enhanced Data Processing...")
    try:
        from data_processing_improved import (
            create_advanced_volatility_features,
            create_seasonal_features,
            create_microstructure_features,
            create_regime_features,
            sophisticated_data_cleaning
        )
        print("  ✅ All enhanced data processing functions imported successfully")
    except Exception as e:
        print(f"  ❌ Error importing data processing functions: {e}")
        return False
    
    # Test 2: Create sample data
    print("\n2️⃣ Creating Sample Data...")
    try:
        # Create sample stock data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        data = []
        for symbol in symbols:
            np.random.seed(42)  # For reproducible results
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Add first element
            
            for i, date in enumerate(dates):
                data.append({
                    'Date': date,
                    'symbol': symbol,
                    'price': prices[i],
                    'return': returns[i],
                    'sector': 'Technology',
                    'industry': 'Software'
                })
        
        df = pd.DataFrame(data)
        print(f"  ✅ Created sample data: {len(df)} records for {len(symbols)} stocks")
    except Exception as e:
        print(f"  ❌ Error creating sample data: {e}")
        return False
    
    # Test 3: Advanced Volatility Features
    print("\n3️⃣ Testing Advanced Volatility Features...")
    try:
        sample_stock = df[df['symbol'] == 'AAPL'].copy()
        enhanced_stock = create_advanced_volatility_features(sample_stock)
        
        volatility_features = [col for col in enhanced_stock.columns if 'vol' in col.lower()]
        print(f"  ✅ Created {len(volatility_features)} volatility features:")
        for feature in volatility_features[:5]:  # Show first 5
            print(f"    • {feature}")
        
    except Exception as e:
        print(f"  ❌ Error in volatility features: {e}")
        return False
    
    # Test 4: Seasonal Features
    print("\n4️⃣ Testing Seasonal Features...")
    try:
        seasonal_stock = create_seasonal_features(enhanced_stock)
        
        seasonal_features = [col for col in seasonal_stock.columns 
                           if any(x in col.lower() for x in ['month', 'day', 'quarter', 'season', 'holiday', 'effect'])]
        print(f"  ✅ Created {len(seasonal_features)} seasonal features:")
        for feature in seasonal_features[:5]:  # Show first 5
            print(f"    • {feature}")
        
    except Exception as e:
        print(f"  ❌ Error in seasonal features: {e}")
        return False
    
    # Test 5: Microstructure Features
    print("\n5️⃣ Testing Microstructure Features...")
    try:
        microstructure_stock = create_microstructure_features(seasonal_stock)
        
        micro_features = [col for col in microstructure_stock.columns 
                         if any(x in col.lower() for x in ['impact', 'spread', 'volume', 'imbalance', 'reversal'])]
        print(f"  ✅ Created {len(micro_features)} microstructure features:")
        for feature in micro_features[:5]:  # Show first 5
            print(f"    • {feature}")
        
    except Exception as e:
        print(f"  ❌ Error in microstructure features: {e}")
        return False
    
    # Test 6: Regime Features
    print("\n6️⃣ Testing Regime Features...")
    try:
        regime_stock = create_regime_features(microstructure_stock)
        
        regime_features = [col for col in regime_stock.columns 
                          if any(x in col.lower() for x in ['regime', 'trend', 'bull', 'bear', 'stress'])]
        print(f"  ✅ Created {len(regime_features)} regime features:")
        for feature in regime_features[:5]:  # Show first 5
            print(f"    • {feature}")
        
    except Exception as e:
        print(f"  ❌ Error in regime features: {e}")
        return False
    
    # Test 7: Data Cleaning
    print("\n7️⃣ Testing Sophisticated Data Cleaning...")
    try:
        # Add some outliers to test cleaning
        test_df = df.copy()
        test_df.loc[100, 'return'] = 5.0  # Extreme return
        test_df.loc[200, 'price'] = 0.001  # Penny stock
        
        cleaned_df = sophisticated_data_cleaning(test_df)
        print(f"  ✅ Data cleaning: {len(test_df)} → {len(cleaned_df)} records")
        
    except Exception as e:
        print(f"  ❌ Error in data cleaning: {e}")
        return False
    
    # Test 8: Hyperparameter Optimization Functions
    print("\n8️⃣ Testing Hyperparameter Optimization...")
    try:
        from financial_models_analysis import get_hyperparameter_grids, hyperparameter_optimization
        
        param_grids = get_hyperparameter_grids()
        print(f"  ✅ Hyperparameter grids available for {len(param_grids)} models")
        
        # Test with a simple model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        if 'Random_Forest' in param_grids:
            # Quick test with minimal parameters
            test_grid = {'grid': {'n_estimators': [5, 10], 'max_depth': [3, 5]}}
            optimized_model, tuning_info = hyperparameter_optimization(
                base_model, test_grid, X, y, method='grid', cv_folds=2, n_iter=2
            )
            print(f"  ✅ Hyperparameter optimization completed")
            print(f"    Best score: {tuning_info.get('best_score', 'N/A')}")
        
    except Exception as e:
        print(f"  ❌ Error in hyperparameter optimization: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL ENHANCED FEATURES TESTED SUCCESSFULLY!")
    print("="*60)
    print("\n💡 Enhanced Features Summary:")
    print("  🔧 Advanced volatility measures (GARCH-like, realized volatility)")
    print("  📅 Seasonal and calendar effects (monthly, quarterly, holiday)")
    print("  🏗️ Market microstructure features (liquidity, order flow)")
    print("  📊 Regime detection (bull/bear markets, volatility regimes)")
    print("  🧹 Sophisticated data cleaning (outlier detection, noise reduction)")
    print("  ⚙️ Hyperparameter optimization (Grid, Random, Bayesian search)")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_features()
    sys.exit(0 if success else 1) 
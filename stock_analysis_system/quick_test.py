#!/usr/bin/env python3
"""
Quick Test Script for Stock Analysis System

Demonstrates the speed improvements with optimized configurations.
"""

import sys
import os
import time

# Add core directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def test_quick_mode():
    """Test the quick mode optimizations"""
    print("🚀 TESTING STOCK ANALYSIS QUICK MODE OPTIMIZATIONS")
    print("="*60)
    
    # Change to core directory
    original_dir = os.getcwd()
    os.chdir('core')
    
    try:
        from data_processing_improved import improved_preprocess_pipeline
        from financial_models_analysis import train_stock_prediction_models
        
        # Load data
        print("\n📊 Loading and preprocessing data...")
        start_time = time.time()
        X, y, feature_cols, df = improved_preprocess_pipeline()
        data_load_time = time.time() - start_time
        print(f"  ✅ Data loaded in {data_load_time:.1f} seconds")
        print(f"  📊 Dataset: {X.shape[0]:,} records, {X.shape[1]} features")
        
        if X is None:
            print("❌ Failed to load data")
            return
        
        # Test regular mode (limited)
        print("\n🐌 Testing REGULAR mode (limited to Random Forest only)...")
        start_time = time.time()
        
        # Just test Random Forest in regular mode
        from sklearn.ensemble import RandomForestRegressor
        rf_regular = RandomForestRegressor(
            n_estimators=100,  # Regular complexity
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Sample for testing
        sample_size = min(10000, len(X))
        sample_indices = X.sample(n=sample_size).index
        X_test = X.loc[sample_indices]
        y_test = y.loc[sample_indices]
        
        rf_regular.fit(X_test, y_test)
        regular_time = time.time() - start_time
        print(f"  ✅ Regular Random Forest: {regular_time:.1f} seconds")
        
        # Test quick mode
        print("\n⚡ Testing QUICK mode...")
        start_time = time.time()
        results = train_stock_prediction_models(X, y, feature_cols, quick_mode=True)
        quick_time = time.time() - start_time
        print(f"  ✅ Quick mode completed in {quick_time:.1f} seconds")
        
        # Compare results
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  Regular Random Forest: {regular_time:.1f}s")
        print(f"  Quick mode (all models): {quick_time:.1f}s")
        
        if quick_time < regular_time * 2:  # Quick mode trains multiple models
            print(f"  🎉 Quick mode efficiency: Training {len(results)} models in {quick_time/regular_time:.1f}x the time of 1 regular model!")
        
        # Show model results
        if results:
            print(f"\n🏆 QUICK MODE RESULTS:")
            for name, result in results.items():
                if result is not None:
                    print(f"  {name}: RMSE={result['rmse_cv']:.6f}, "
                          f"Directional Accuracy={result['directional_accuracy']:.1%}, "
                          f"Time={result['training_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    test_quick_mode() 
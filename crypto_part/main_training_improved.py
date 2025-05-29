#!/usr/bin/env python3
"""
Improved XGBoost Training Script for DeFi Yield Prediction

This script uses the custom feature selection requested by the user and 
applies comprehensive grid search for hyperparameter tuning.

Features used:
- project, symbol, stablecoin, exposure, ilrisk, underlyingtokens, 
  rewardtokens, pool, tvlusd, volumeusd1d, percentage of total tvl, 
  sigma, mu + additional useful metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processing import (
    load_data, preprocess_data, engineer_features, 
    create_filtered_dataset, prepare_model_data_custom
)
from model_training import (
    train_xgboost_comprehensive, create_ensemble_model, 
    predict_ensemble, save_model
)
from model_evaluation import (
    evaluate_model, get_feature_importance, 
    compare_with_original_predictions, summarize_comparison
)
from visualization import (
    plot_feature_importance, plot_predictions_vs_actual,
    plot_residuals, plot_prediction_intervals
)

def main_training_pipeline(data_path, use_ensemble=False, search_type='grid'):
    """
    Main training pipeline with improved feature selection and grid search
    """
    print("=== Improved DeFi Yield Prediction Model Training ===")
    print(f"Data source: {data_path}")
    print(f"Search type: {search_type}")
    print(f"Use ensemble: {use_ensemble}")
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_data(data_path)
    if df is None:
        print("Failed to load data. Exiting...")
        return None
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Display basic data info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Prepare model data with custom feature selection
    print("\n2. Preparing model data with custom feature selection...")
    X, y, numeric_features, categorical_features, binary_features = prepare_model_data_custom(df, target='apy')
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 4: Train model
    print("\n4. Training XGBoost model...")
    
    if use_ensemble:
        # Train ensemble model
        models = create_ensemble_model(
            X_train, y_train, 
            numeric_features + binary_features,  # Combine numeric and binary
            categorical_features,
            n_models=3
        )
        
        # Make ensemble predictions
        y_pred, pred_std = predict_ensemble(models, X_test)
        
        # For evaluation, use the first model
        best_model = models[0]
        best_params = "Ensemble model"
        
    else:
        # Train single model with comprehensive search
        best_model, best_params, search_results = train_xgboost_comprehensive(
            X_train, y_train,
            numeric_features + binary_features,  # Combine numeric and binary
            categorical_features,
            search_type=search_type,
            cv_folds=5
        )
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        pred_std = None
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model performance...")
    _, metrics = evaluate_model(best_model, X_test, y_test)
    
    # Step 6: Feature importance analysis
    print("\n6. Analyzing feature importance...")
    try:
        # Get feature names after preprocessing
        feature_names = (numeric_features + binary_features + categorical_features)
        importance_df = get_feature_importance(best_model, feature_names)
        print("Top 10 most important features:")
        print(importance_df.head(10))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        importance_df = None
    
    # Step 7: Compare with original predictions (if available)
    print("\n7. Comparing with original predictions...")
    test_indices = X_test.index
    comparison_df = compare_with_original_predictions(y_test, y_pred, df, test_indices)
    
    # Step 8: Create visualizations
    print("\n8. Creating visualizations...")
    try:
        if importance_df is not None:
            plot_feature_importance(importance_df, top_n=15, save_path='feature_importance_improved.png')
        
        plot_prediction_vs_actual(y_test, y_pred, save_path='prediction_vs_actual_improved.png')
        plot_residuals(y_test, y_pred, save_path='residuals_improved.png')
        
        if pred_std is not None:
            plot_prediction_intervals(y_test, y_pred, pred_std, save_path='prediction_intervals_improved.png')
        
        print("Visualizations saved successfully!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Step 9: Save model
    print("\n9. Saving model...")
    if use_ensemble:
        # Save ensemble models
        for i, model in enumerate(models):
            save_model(model, model_name=f'defi_yield_ensemble_model_{i}.pkl')
    else:
        save_model(best_model, model_name='defi_yield_improved_model.pkl')
    
    # Step 10: Summary and recommendations
    print("\n10. Summary and Recommendations")
    summarize_comparison(metrics)
    
    # Feature selection summary
    print(f"\nFeature Selection Summary:")
    print(f"- Used {len(numeric_features + binary_features)} numeric/binary features")
    print(f"- Used {len(categorical_features)} categorical features")
    print(f"- Total features: {len(feature_names)}")
    
    # Model performance summary
    print(f"\nModel Performance Summary:")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- R² Score: {metrics['r2']:.4f}")
    print(f"- MAPE: {metrics['mape']:.4f}%")
    
    # Return results for further analysis
    results = {
        'model': best_model,
        'best_params': best_params,
        'metrics': metrics,
        'comparison_df': comparison_df,
        'importance_df': importance_df,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    if use_ensemble:
        results['ensemble_models'] = models
        results['pred_std'] = pred_std
    
    return results

def run_experiment_comparison():
    """
    Run experiments with different configurations to compare performance
    """
    print("=== Running Experiment Comparison ===")
    
    # This function assumes you have your data file
    # Replace with your actual data path
    data_path = "your_data_file.json"  # Update this path
    
    experiments = [
        {"search_type": "grid", "use_ensemble": False, "name": "Grid Search Single Model"},
        {"search_type": "randomized", "use_ensemble": False, "name": "Randomized Search Single Model"},
        {"search_type": "randomized", "use_ensemble": True, "name": "Ensemble Model"}
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n--- Running {exp['name']} ---")
        try:
            result = main_training_pipeline(
                data_path, 
                use_ensemble=exp['use_ensemble'],
                search_type=exp['search_type']
            )
            results[exp['name']] = result
        except Exception as e:
            print(f"Error in {exp['name']}: {e}")
            results[exp['name']] = None
    
    # Compare results
    print("\n=== Experiment Results Comparison ===")
    for name, result in results.items():
        if result is not None:
            metrics = result['metrics']
            print(f"{name}:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.4f}%")
        else:
            print(f"{name}: Failed")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("DeFi Yield Prediction - Improved Training Script")
    print("To use this script, call main_training_pipeline() with your data path")
    print("\nExample:")
    print("results = main_training_pipeline('your_data.json', search_type='grid')")
    print("\nOr run experiment comparison:")
    print("experiment_results = run_experiment_comparison()") 
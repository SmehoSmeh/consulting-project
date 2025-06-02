#!/usr/bin/env python3
"""
Financial Models Analysis for Stock Price Prediction - Enhanced Version

This script implements advanced tree-based modeling approaches for stock returns:
1. Raw stock data exploration with comprehensive visualizations
2. Hyperparameter-tuned XGBoost, LightGBM, CatBoost, Random Forest models
3. Advanced volatility and seasonal features
4. Sophisticated model validation and interpretation
5. Tree-based feature importance analysis
6. Financial-grade validation with market regime awareness
7. Sector-based analysis and predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import uniform, randint
import warnings
import time
warnings.filterwarnings('ignore')

# Import tree-based models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, will skip XGBoost model")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available, will skip LightGBM model")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will skip CatBoost model")

# Optional Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("⚠️ Bayesian optimization not available, will use Grid/Random search")

from data_processing_improved import load_data, improved_preprocess_pipeline

# Set up plotting style for financial reports
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def get_hyperparameter_grids():
    """Define hyperparameter grids for optimization"""
    
    param_grids = {}
    
    # Random Forest hyperparameters
    param_grids['Random_Forest'] = {
        'grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        },
        'random': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(5, 25),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None]
        }
    }
    
    # XGBoost hyperparameters
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            },
            'random': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.29),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 2),
                'reg_lambda': uniform(0.5, 3)
            }
        }
    
    # LightGBM hyperparameters
    if LIGHTGBM_AVAILABLE:
        param_grids['LightGBM'] = {
            'grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 62, 127],
                'min_child_samples': [10, 20, 30]
            },
            'random': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.29),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'num_leaves': randint(20, 200),
                'min_child_samples': randint(5, 50)
            }
        }
    
    # CatBoost hyperparameters
    if CATBOOST_AVAILABLE:
        param_grids['CatBoost'] = {
            'grid': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 64, 128]
            },
            'random': {
                'iterations': randint(50, 500),
                'depth': randint(3, 12),
                'learning_rate': uniform(0.01, 0.29),
                'l2_leaf_reg': uniform(0.1, 10),
                'border_count': randint(16, 255)
            }
        }
    
    return param_grids

def hyperparameter_optimization(model, param_grid, X, y, method='random', cv_folds=3, n_iter=50):
    """Perform hyperparameter optimization"""
    
    print(f"  🔧 Hyperparameter optimization using {method} search...")
    
    # Use TimeSeriesSplit for financial data
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    if method == 'grid' and 'grid' in param_grid:
        optimizer = GridSearchCV(
            model, 
            param_grid['grid'], 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    elif method == 'random' and 'random' in param_grid:
        optimizer = RandomizedSearchCV(
            model,
            param_grid['random'],
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    elif method == 'bayesian' and BAYESIAN_OPT_AVAILABLE and 'random' in param_grid:
        # Convert random distributions to skopt format
        search_spaces = {}
        for param, dist in param_grid['random'].items():
            if hasattr(dist, 'rvs'):  # scipy distribution
                if isinstance(dist, type(randint(1, 2))):  # randint
                    search_spaces[param] = Integer(dist.args[0], dist.args[1])
                elif isinstance(dist, type(uniform(0, 1))):  # uniform
                    search_spaces[param] = Real(dist.args[0], dist.args[0] + dist.args[1])
            else:  # categorical
                search_spaces[param] = param_grid['grid'][param] if param in param_grid['grid'] else [dist]
        
        optimizer = BayesSearchCV(
            model,
            search_spaces,
            n_iter=min(n_iter, 30),  # Bayesian is more efficient
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        print(f"    ⚠️ Method {method} not available, using base model")
        return model, {}
    
    # Fit the optimizer
    start_time = time.time()
    optimizer.fit(X, y)
    optimization_time = time.time() - start_time
    
    print(f"    ✅ Optimization completed in {optimization_time:.1f}s")
    print(f"    ✅ Best score: {-optimizer.best_score_:.6f}")
    print(f"    ✅ Best params: {optimizer.best_params_}")
    
    return optimizer.best_estimator_, {
        'best_score': optimizer.best_score_,
        'best_params': optimizer.best_params_,
        'optimization_time': optimization_time
    }

def explore_raw_stock_data(df):
    """Comprehensive exploration of raw stock data"""
    print("="*80)
    print("📊 RAW STOCK DATA EXPLORATION & ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\n📋 DATASET OVERVIEW:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique stocks: {df['symbol'].nunique():,}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Returns Distribution Analysis
    if 'return' in df.columns:
        print(f"\n📈 RETURNS DISTRIBUTION ANALYSIS:")
        returns_stats = df['return'].describe()
        print(f"  Mean Return: {returns_stats['mean']:.4f} ({returns_stats['mean']*100:.2f}%)")
        print(f"  Median Return: {returns_stats['50%']:.4f} ({returns_stats['50%']*100:.2f}%)")
        print(f"  Standard Deviation: {returns_stats['std']:.4f} ({returns_stats['std']*100:.2f}%)")
        print(f"  Range: {returns_stats['min']:.4f} to {returns_stats['max']:.4f}")
        print(f"  Interquartile Range: {returns_stats['25%']:.4f} to {returns_stats['75%']:.4f}")
        
        # Identify extreme movements
        q99 = df['return'].quantile(0.99)
        q01 = df['return'].quantile(0.01)
        extreme_count = ((df['return'] > q99) | (df['return'] < q01)).sum()
        print(f"  Extreme movements (>99th or <1st percentile): {extreme_count} ({extreme_count/len(df)*100:.1f}%)")
    
    # Price Distribution Analysis
    if 'price' in df.columns:
        print(f"\n💰 PRICE DISTRIBUTION ANALYSIS:")
        price_stats = df['price'].describe()
        print(f"  Mean Price: ${price_stats['mean']:.2f}")
        print(f"  Median Price: ${price_stats['50%']:.2f}")
        print(f"  Price Range: ${price_stats['min']:.2f} to ${price_stats['max']:.2f}")
        
        # Price concentration by stock
        price_by_stock = df.groupby('symbol')['price'].mean().describe()
        print(f"  Average stock price range: ${price_by_stock['min']:.2f} to ${price_by_stock['max']:.2f}")
    
    # Sector Distribution
    if 'sector' in df.columns:
        print(f"\n🏢 SECTOR DISTRIBUTION:")
        sector_counts = df['sector'].value_counts()
        print(f"  Number of sectors: {len(sector_counts)}")
        print(f"  Top 5 sectors by record count:")
        for i, (sector, count) in enumerate(sector_counts.head(5).items()):
            print(f"    {i+1}. {sector}: {count:,} records ({count/len(df)*100:.1f}%)")
    
    # Time series analysis
    if 'Date' in df.columns:
        print(f"\n📅 TIME SERIES ANALYSIS:")
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = df['Date'].max() - df['Date'].min()
        print(f"  Total time span: {date_range.days} days ({date_range.days/365.25:.1f} years)")
        
        # Records per year
        df['year'] = df['Date'].dt.year
        yearly_counts = df['year'].value_counts().sort_index()
        print(f"  Records per year:")
        for year, count in yearly_counts.items():
            print(f"    {year}: {count:,} records")
    
    return df

def create_raw_stock_data_visualizations(df):
    """Create comprehensive visualizations of raw stock data"""
    print(f"\n📊 Creating comprehensive stock data visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('../results/plots', exist_ok=True)
    
    # Figure 1: Returns and Price Distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stock Market Data: Raw Data Overview', fontsize=16, fontweight='bold')
    
    # Returns Distribution
    if 'return' in df.columns:
        # Remove extreme outliers for visualization
        returns_clean = df['return'][(df['return'] > df['return'].quantile(0.01)) & 
                                    (df['return'] < df['return'].quantile(0.99))]
        
        axes[0,0].hist(returns_clean, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Daily Returns')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Daily Returns Distribution (1st-99th percentile)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_return = returns_clean.mean()
        median_return = returns_clean.median()
        axes[0,0].axvline(mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.4f}')
        axes[0,0].axvline(median_return, color='orange', linestyle='--', label=f'Median: {median_return:.4f}')
        axes[0,0].legend()
    
    # Price Distribution (log scale)
    if 'price' in df.columns:
        price_positive = df['price'][df['price'] > 0]
        axes[0,1].hist(np.log10(price_positive), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_xlabel('Price (log10 USD)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Stock Price Distribution (Log Scale)')
        axes[0,1].grid(True, alpha=0.3)
    
    # Returns vs Price Scatter Plot
    if 'return' in df.columns and 'price' in df.columns:
        # Sample data for performance
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size)
        
        # Filter for reasonable values
        mask = (sample_df['return'].abs() < 0.2) & (sample_df['price'] > 1)
        plot_data = sample_df[mask]
        
        scatter = axes[1,0].scatter(plot_data['price'], plot_data['return'], 
                                   alpha=0.3, c=plot_data['return'], cmap='RdYlGn', s=5)
        axes[1,0].set_xscale('log')
        axes[1,0].set_xlabel('Price (USD, log scale)')
        axes[1,0].set_ylabel('Daily Return')
        axes[1,0].set_title('Daily Returns vs Stock Price')
        axes[1,0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,0], label='Return')
    
    # Sector Distribution
    if 'sector' in df.columns:
        sector_counts = df['sector'].value_counts().head(10)
        axes[1,1].bar(range(len(sector_counts)), sector_counts.values, color='coral')
        axes[1,1].set_xlabel('Sector')
        axes[1,1].set_ylabel('Number of Records')
        axes[1,1].set_title('Top 10 Sectors by Record Count')
        axes[1,1].set_xticks(range(len(sector_counts)))
        axes[1,1].set_xticklabels(sector_counts.index, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/tree_models_raw_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Raw stock data visualizations saved to ../results/plots/")

def calculate_mape_safe(y_true, y_pred):
    """Calculate MAPE safely, handling edge cases"""
    # For returns, use absolute values to avoid division by near-zero
    mask = np.abs(y_true) > 1e-6  # Avoid division by very small numbers
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_stock_prediction_models(X, y, feature_names, quick_mode=False, tune_hyperparams=True):
    """Train tree-based models for stock return prediction with hyperparameter tuning"""
    print(f"\n🌳 TRAINING ADVANCED TREE-BASED MODELS FOR STOCK PREDICTION")
    print("="*60)
    
    if quick_mode:
        print("⚡ QUICK MODE ENABLED - Using faster model configurations")
        tune_hyperparams = False  # Skip tuning in quick mode
        # Sample data for faster training in development
        if len(X) > 50000:
            sample_size = 50000
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            print(f"  📊 Sampled {sample_size:,} records from {len(X):,} for faster training")
        else:
            X_sample, y_sample = X, y
    else:
        X_sample, y_sample = X, y
        print(f"  📊 Training on full dataset: {len(X):,} records")
    
    if tune_hyperparams:
        print("  🔧 HYPERPARAMETER TUNING ENABLED")
        print("  📊 This will take longer but provide better performance")
        param_grids = get_hyperparameter_grids()
        tuning_method = 'bayesian' if BAYESIAN_OPT_AVAILABLE else 'random'
        print(f"  🎯 Using {tuning_method} search for optimization")
    else:
        print("  ⚡ Using default hyperparameters for speed")
    
    # Define base models with improved default parameters
    models = {}
    
    # LightGBM - FASTEST MODEL (prioritize first)
    if LIGHTGBM_AVAILABLE:
        base_lgb_params = {
            'n_estimators': 100 if quick_mode else 300,
            'max_depth': 6 if quick_mode else 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 31,
            'min_child_samples': 20,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
            'force_row_wise': True
        }
        
        base_model = lgb.LGBMRegressor(**base_lgb_params)
        
        if tune_hyperparams and 'LightGBM' in param_grids:
            tuned_model, tuning_info = hyperparameter_optimization(
                base_model, param_grids['LightGBM'], X_sample, y_sample, 
                method=tuning_method, n_iter=20 if quick_mode else 50
            )
            models['LightGBM'] = {'model': tuned_model, 'tuning_info': tuning_info}
        else:
            models['LightGBM'] = {'model': base_model, 'tuning_info': {}}
    
    # Random Forest - RELIABLE BASELINE
    base_rf_params = {
        'n_estimators': 50 if quick_mode else 200,
        'max_depth': 10 if quick_mode else 15,
        'min_samples_split': 20 if quick_mode else 10,
        'min_samples_leaf': 10 if quick_mode else 5,
        'random_state': 42,
        'n_jobs': -1,
        'max_features': 'sqrt'
    }
    
    base_model = RandomForestRegressor(**base_rf_params)
    
    if tune_hyperparams and 'Random_Forest' in param_grids:
        tuned_model, tuning_info = hyperparameter_optimization(
            base_model, param_grids['Random_Forest'], X_sample, y_sample,
            method=tuning_method, n_iter=20 if quick_mode else 50
        )
        models['Random_Forest'] = {'model': tuned_model, 'tuning_info': tuning_info}
    else:
        models['Random_Forest'] = {'model': base_model, 'tuning_info': {}}
    
    # XGBoost - HIGH PERFORMANCE
    if XGBOOST_AVAILABLE:
        base_xgb_params = {
            'n_estimators': 100 if quick_mode else 300,
            'max_depth': 6 if quick_mode else 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        base_model = xgb.XGBRegressor(**base_xgb_params)
        
        if tune_hyperparams and 'XGBoost' in param_grids:
            tuned_model, tuning_info = hyperparameter_optimization(
                base_model, param_grids['XGBoost'], X_sample, y_sample,
                method=tuning_method, n_iter=20 if quick_mode else 50
            )
            models['XGBoost'] = {'model': tuned_model, 'tuning_info': tuning_info}
        else:
            models['XGBoost'] = {'model': base_model, 'tuning_info': {}}
    
    # CatBoost - CATEGORICAL EXPERT (skip in quick mode for speed)
    if CATBOOST_AVAILABLE and not quick_mode:
        base_cat_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        base_model = cb.CatBoostRegressor(**base_cat_params)
        
        if tune_hyperparams and 'CatBoost' in param_grids:
            tuned_model, tuning_info = hyperparameter_optimization(
                base_model, param_grids['CatBoost'], X_sample, y_sample,
                method=tuning_method, n_iter=15 if quick_mode else 30
            )
            models['CatBoost'] = {'model': tuned_model, 'tuning_info': tuning_info}
        else:
            models['CatBoost'] = {'model': base_model, 'tuning_info': {}}
    
    # Use faster cross-validation for quick mode
    cv_folds = 3 if quick_mode else 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    print(f"  🔍 Using {cv_folds}-fold time series cross-validation")
    
    results = {}
    total_models = len(models)
    
    for i, (name, model_info) in enumerate(models.items(), 1):
        print(f"\nEvaluating {name} ({i}/{total_models})...")
        model = model_info['model']
        tuning_info = model_info['tuning_info']
        
        start_time = time.time()
        
        try:
            # Time series cross-validation scores
            cv_scores = cross_val_score(model, X_sample, y_sample, cv=tscv, 
                                      scoring='neg_mean_squared_error', n_jobs=1)
            rmse_cv = np.sqrt(-cv_scores.mean())
            rmse_cv_std = np.sqrt(-cv_scores).std()
            
            # Full model training for additional metrics
            model.fit(X_sample, y_sample)
            y_pred_full = model.predict(X_sample)
            
            # Calculate comprehensive metrics
            r2 = r2_score(y_sample, y_pred_full)
            mae = mean_absolute_error(y_sample, y_pred_full)
            mape = calculate_mape_safe(y_sample, y_pred_full)
            
            # Calculate directional accuracy (important for trading)
            direction_actual = (y_sample > 0).astype(int)
            direction_pred = (y_pred_full > 0).astype(int)
            directional_accuracy = (direction_actual == direction_pred).mean()
            
            # Calculate regime-aware metrics
            regime_metrics = calculate_regime_aware_metrics(y_sample, y_pred_full, X_sample)
            
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'model': model,
                'rmse_cv': rmse_cv,
                'rmse_cv_std': rmse_cv_std,
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'training_time': training_time,
                'cv_scores': cv_scores,
                'y_pred': y_pred_full,
                'tuning_info': tuning_info,
                'regime_metrics': regime_metrics
            }
            
            print(f"  ✅ RMSE (CV): {rmse_cv:.6f} ± {rmse_cv_std:.6f}")
            print(f"  ✅ R² Score: {r2:.4f}")
            print(f"  ✅ MAE: {mae:.6f}")
            print(f"  ✅ MAPE: {mape:.2f}%")
            print(f"  ✅ Directional Accuracy: {directional_accuracy:.1%}")
            print(f"  ⏱️ Total Time: {training_time:.1f} seconds")
            
            if tuning_info:
                print(f"  🔧 Hyperparameter tuning improved score by: {abs(tuning_info.get('best_score', 0)) - rmse_cv:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")
            results[name] = None
    
    # Print performance ranking
    valid_results = [(name, result) for name, result in results.items() if result is not None]
    if len(valid_results) > 1:
        print(f"\n🏆 MODEL PERFORMANCE RANKING:")
        performance_ranking = sorted(valid_results, key=lambda x: x[1]['rmse_cv'])
        for i, (name, result) in enumerate(performance_ranking, 1):
            print(f"  {i}. {name}: RMSE={result['rmse_cv']:.6f}, Directional={result['directional_accuracy']:.1%}")
    
    return results

def calculate_regime_aware_metrics(y_true, y_pred, X):
    """Calculate performance metrics aware of market regimes"""
    
    regime_metrics = {}
    
    try:
        # High volatility periods
        if 'vol_regime' in X.columns:
            high_vol_mask = X['vol_regime'] == 1
            if high_vol_mask.sum() > 10:  # Enough data points
                high_vol_rmse = np.sqrt(mean_squared_error(y_true[high_vol_mask], y_pred[high_vol_mask]))
                regime_metrics['high_vol_rmse'] = high_vol_rmse
        
        # Bull vs Bear markets
        if 'bull_market' in X.columns:
            bull_mask = X['bull_market'] == 1
            if bull_mask.sum() > 10:
                bull_rmse = np.sqrt(mean_squared_error(y_true[bull_mask], y_pred[bull_mask]))
                regime_metrics['bull_market_rmse'] = bull_rmse
        
        # Stress periods
        if 'stress_indicator' in X.columns:
            stress_mask = X['stress_indicator'] == 1
            if stress_mask.sum() > 10:
                stress_rmse = np.sqrt(mean_squared_error(y_true[stress_mask], y_pred[stress_mask]))
                regime_metrics['stress_rmse'] = stress_rmse
        
        # Seasonal effects
        if 'monday_effect' in X.columns:
            monday_mask = X['monday_effect'] == 1
            if monday_mask.sum() > 10:
                monday_directional = ((y_true[monday_mask] > 0) == (y_pred[monday_mask] > 0)).mean()
                regime_metrics['monday_directional'] = monday_directional
                
    except Exception as e:
        print(f"    ⚠️ Could not calculate regime metrics: {e}")
    
    return regime_metrics

def analyze_stock_feature_importance(models, feature_names):
    """Analyze feature importance for stock prediction models"""
    print(f"\n📊 STOCK FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance from Random Forest (always available)
    if 'Random_Forest' in models and models['Random_Forest'] is not None:
        rf_model = models['Random_Forest']['model']
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Scaled_Importance': importances / np.max(importances) * 100
        }).sort_values('Importance', ascending=False)
        
        print("\n🔍 TOP 15 MOST IMPORTANT FEATURES (Random Forest):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Feature':<35} {'Importance':<12} {'Scaled %':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"{i+1:<4} {row['Feature'][:34]:<35} {row['Importance']:<12.6f} {row['Scaled_Importance']:<10.1f}%")
        
        return importance_df
    
    return None

def create_stock_model_visualizations(models, y, feature_importance=None):
    """Create comprehensive stock model comparison visualizations"""
    print(f"\n📊 Creating stock model comparison visualizations...")
    
    # Figure 3: Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tree-Based Models: Stock Prediction Performance', fontsize=16, fontweight='bold')
    
    # Model metrics comparison
    model_names = []
    rmse_scores = []
    r2_scores = []
    directional_accuracies = []
    
    for name, result in models.items():
        if result is not None:
            model_names.append(name.replace('_', ' '))
            rmse_scores.append(result['rmse_cv'])
            r2_scores.append(result['r2'])
            directional_accuracies.append(result['directional_accuracy'])
    
    # RMSE Comparison
    axes[0,0].bar(model_names, rmse_scores, color='skyblue', alpha=0.8)
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_title('Model RMSE Comparison (Lower is Better)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # R² Comparison
    axes[0,1].bar(model_names, r2_scores, color='lightgreen', alpha=0.8)
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].set_title('Model R² Comparison (Higher is Better)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Directional Accuracy Comparison
    axes[1,0].bar(model_names, directional_accuracies, color='coral', alpha=0.8)
    axes[1,0].set_ylabel('Directional Accuracy')
    axes[1,0].set_title('Model Directional Accuracy (Higher is Better)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Prediction vs Actual (Best Model by RMSE)
    best_model_name = min(models.keys(), key=lambda k: models[k]['rmse_cv'] if models[k] else float('inf'))
    if models[best_model_name] is not None:
        y_pred_best = models[best_model_name]['y_pred']
        
        # Sample for visualization
        sample_size = min(5000, len(y))
        sample_indices = np.random.choice(len(y), sample_size, replace=False)
        y_sample = y.iloc[sample_indices]
        y_pred_sample = y_pred_best[sample_indices]
        
        axes[1,1].scatter(y_sample, y_pred_sample, alpha=0.5, s=10)
        axes[1,1].plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--', lw=2, label='Perfect Prediction')
        axes[1,1].set_xlabel('Actual Returns')
        axes[1,1].set_ylabel('Predicted Returns')
        axes[1,1].set_title(f'Best Model: {best_model_name.replace("_", " ")}')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/tree_models_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Feature Importance Analysis
    if feature_importance is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stock Models: Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Feature importance plot
        top_features = feature_importance.head(20)  # Show top 20 features
        
        axes[0,0].barh(range(len(top_features)), top_features['Importance'], color='forestgreen', alpha=0.7)
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels(top_features['Feature'])
        axes[0,0].set_xlabel('Feature Importance')
        axes[0,0].set_title('Top 20 Feature Importances (Random Forest)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Feature importance distribution
        axes[0,1].hist(feature_importance['Importance'], bins=30, alpha=0.7, color='darkgreen')
        axes[0,1].set_xlabel('Feature Importance Value')
        axes[0,1].set_ylabel('Number of Features')
        axes[0,1].set_title('Feature Importance Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Cumulative importance
        sorted_importance = feature_importance.sort_values('Importance', ascending=False)
        cumulative_importance = np.cumsum(sorted_importance['Importance'])
        axes[1,0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, color='blue')
        axes[1,0].set_xlabel('Number of Features')
        axes[1,0].set_ylabel('Cumulative Importance')
        axes[1,0].set_title('Cumulative Feature Importance')
        axes[1,0].grid(True, alpha=0.3)
        
        # Feature categories analysis
        # Categorize features
        feature_categories = []
        for feature in feature_importance['Feature']:
            if 'sector_' in feature:
                feature_categories.append('Sector')
            elif 'industry_' in feature:
                feature_categories.append('Industry')
            elif any(x in feature for x in ['sma', 'price_change', 'price_relative']):
                feature_categories.append('Technical')
            elif any(x in feature for x in ['return', 'volatility', 'std']):
                feature_categories.append('Volatility')
            elif 'rsi' in feature:
                feature_categories.append('Momentum')
            else:
                feature_categories.append('Other')
        
        feature_importance['Category'] = feature_categories
        category_importance = feature_importance.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        
        axes[1,1].pie(category_importance.values, labels=category_importance.index, autopct='%1.1f%%')
        axes[1,1].set_title('Feature Importance by Category')
        
        plt.tight_layout()
        plt.savefig('../results/plots/tree_models_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✅ Stock model visualizations saved to ../results/plots/")

def generate_stock_models_summary(models, data_info):
    """Generate comprehensive summary of stock prediction models"""
    print(f"\n📋 GENERATING STOCK MODELS SUMMARY...")
    
    # Prepare model comparison data
    model_comparison = []
    
    for name, result in models.items():
        if result is not None:
            model_comparison.append({
                'Model': name.replace('_', ' '),
                'RMSE_CV': result['rmse_cv'],
                'RMSE_CV_Std': result['rmse_cv_std'],
                'R2_Score': result['r2'],
                'MAE': result['mae'],
                'MAPE': result['mape'],
                'Directional_Accuracy': result['directional_accuracy']
            })
    
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df = comparison_df.sort_values('RMSE_CV')  # Sort by best RMSE
    
    # Save to CSV
    comparison_df.to_csv('../results/reports/tree_models_comparison.csv', index=False)
    
    print("✅ Model comparison saved to ../results/reports/tree_models_comparison.csv")
    
    return comparison_df

def main_tree_analysis(quick_mode=False):
    """Main function for tree-based stock analysis"""
    print("="*80)
    if quick_mode:
        print("🚀 STOCK ANALYSIS: TREE-BASED ML PIPELINE (QUICK MODE)")
    else:
        print("🚀 STOCK ANALYSIS: TREE-BASED MACHINE LEARNING PIPELINE")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n📊 STEP 1: DATA LOADING AND PREPROCESSING")
    X, y, feature_cols, df = improved_preprocess_pipeline(quick_mode=quick_mode)
    
    if X is None:
        print("❌ Failed to load and preprocess data")
        return None, None, None
    
    # Step 2: Explore raw data (skip heavy visualizations in quick mode)
    print("\n📈 STEP 2: RAW DATA EXPLORATION")
    explore_raw_stock_data(df)
    if not quick_mode:
        create_raw_stock_data_visualizations(df)
    else:
        print("  ⚡ Skipping detailed visualizations in quick mode")
    
    # Step 3: Train models
    print("\n🤖 STEP 3: MACHINE LEARNING MODEL TRAINING")
    models = train_stock_prediction_models(X, y, feature_cols, quick_mode=quick_mode)
    
    # Step 4: Feature importance analysis
    print("\n🔍 STEP 4: FEATURE IMPORTANCE ANALYSIS")
    feature_importance = analyze_stock_feature_importance(models, feature_cols)
    
    # Step 5: Create visualizations (skip in quick mode)
    if not quick_mode:
        print("\n📊 STEP 5: MODEL VISUALIZATION AND COMPARISON")
        create_stock_model_visualizations(models, y, feature_importance)
    else:
        print("\n📊 STEP 5: MODEL VISUALIZATION AND COMPARISON")
        print("  ⚡ Skipping detailed visualizations in quick mode")
    
    # Step 6: Generate summary
    print("\n📋 STEP 6: SUMMARY GENERATION")
    data_info = {
        'total_records': len(df),
        'stocks': df['symbol'].nunique(),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'sectors': df['sector'].nunique(),
        'features': len(feature_cols)
    }
    
    summary_df = generate_stock_models_summary(models, data_info)
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv('../results/reports/tree_models_feature_importance.csv', index=False)
        print("✅ Feature importance saved to ../results/reports/tree_models_feature_importance.csv")
    
    print("\n" + "="*80)
    if quick_mode:
        print("✅ QUICK TREE-BASED STOCK ANALYSIS COMPLETE!")
    else:
        print("✅ TREE-BASED STOCK ANALYSIS COMPLETE!")
    print("="*80)
    
    # Return results in format expected by main launcher
    results_list = []
    for name, result in models.items():
        if result is not None:
            results_list.append((name, result))
    
    # Sort by RMSE (best first)
    results_list.sort(key=lambda x: x[1]['rmse_cv'])
    
    return results_list, models, df

if __name__ == "__main__":
    # Run the analysis
    results, models, df = main_tree_analysis()
    if results:
        print(f"\n🏆 Best model: {results[0][0]} with RMSE: {results[0][1]['rmse_cv']:.6f}")
        print(f"🎯 Directional accuracy: {results[0][1]['directional_accuracy']:.1%}") 
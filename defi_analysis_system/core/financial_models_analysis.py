#!/usr/bin/env python3
"""
Financial Models Analysis for DeFi Yield Prediction

This script implements tree-based modeling approaches:
1. Raw data exploration with comprehensive visualizations
2. XGBoost, LightGBM, CatBoost, Random Forest models
3. Comprehensive model comparison and metrics
4. Tree-based feature importance analysis
5. Financial-grade validation and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
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

from data_processing_improved import load_data, improved_preprocess_pipeline

# Set up plotting style for financial reports
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def explore_raw_data(df):
    """Comprehensive exploration of raw DeFi data"""
    print("="*80)
    print("📊 RAW DATA EXPLORATION & ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\n📋 DATASET OVERVIEW:")
    print(f"  Total protocols: {len(df):,}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # APY Distribution Analysis
    if 'apy' in df.columns:
        print(f"\n💰 APY DISTRIBUTION ANALYSIS:")
        apy_stats = df['apy'].describe()
        print(f"  Mean APY: {apy_stats['mean']:.2f}%")
        print(f"  Median APY: {apy_stats['50%']:.2f}%")
        print(f"  Standard Deviation: {apy_stats['std']:.2f}%")
        print(f"  Range: {apy_stats['min']:.2f}% to {apy_stats['max']:.2f}%")
        print(f"  Interquartile Range: {apy_stats['25%']:.2f}% to {apy_stats['75%']:.2f}%")
        
        # Identify extreme values
        q99 = df['apy'].quantile(0.99)
        extreme_count = (df['apy'] > q99).sum()
        print(f"  Protocols with APY > 99th percentile ({q99:.1f}%): {extreme_count} ({extreme_count/len(df)*100:.1f}%)")
    
    # TVL Distribution Analysis
    if 'tvlUsd' in df.columns:
        print(f"\n🏦 TVL DISTRIBUTION ANALYSIS:")
        tvl_stats = df['tvlUsd'].describe()
        total_tvl = df['tvlUsd'].sum()  # Calculate sum separately
        print(f"  Total TVL: ${total_tvl:,.0f}")
        print(f"  Mean TVL: ${tvl_stats['mean']:,.0f}")
        print(f"  Median TVL: ${tvl_stats['50%']:,.0f}")
        print(f"  Largest pool: ${tvl_stats['max']:,.0f}")
        
        # TVL concentration
        tvl_sorted = df['tvlUsd'].sort_values(ascending=False)
        top_10_pct = tvl_sorted.head(int(len(df) * 0.1)).sum()
        concentration = (top_10_pct / tvl_sorted.sum()) * 100
        print(f"  Top 10% pools control: {concentration:.1f}% of total TVL")
    
    # Volume Analysis
    if 'volumeUsd1d' in df.columns:
        print(f"\n📈 TRADING VOLUME ANALYSIS:")
        vol_stats = df['volumeUsd1d'].describe()
        total_volume = df['volumeUsd1d'].sum()  # Calculate sum separately
        print(f"  Total daily volume: ${total_volume:,.0f}")
        print(f"  Mean daily volume: ${vol_stats['mean']:,.0f}")
        print(f"  Median daily volume: ${vol_stats['50%']:,.0f}")
        
        # Volume-to-TVL ratio
        if 'tvlUsd' in df.columns:
            df['turnover_ratio'] = df['volumeUsd1d'] / df['tvlUsd']
            avg_turnover = df['turnover_ratio'].mean() * 100
            print(f"  Average turnover ratio: {avg_turnover:.2f}% per day")
    
    # Chain and Project Distribution
    if 'chain' in df.columns:
        print(f"\n⛓️ BLOCKCHAIN DISTRIBUTION:")
        chain_counts = df['chain'].value_counts()
        print(f"  Number of chains: {len(chain_counts)}")
        print(f"  Top 5 chains by protocol count:")
        for i, (chain, count) in enumerate(chain_counts.head(5).items()):
            print(f"    {i+1}. {chain}: {count} protocols ({count/len(df)*100:.1f}%)")
    
    if 'project' in df.columns:
        print(f"\n🚀 PROJECT DISTRIBUTION:")
        project_counts = df['project'].value_counts()
        print(f"  Number of projects: {len(project_counts)}")
        print(f"  Top 5 projects by pool count:")
        for i, (project, count) in enumerate(project_counts.head(5).items()):
            print(f"    {i+1}. {project}: {count} pools ({count/len(df)*100:.1f}%)")
    
    return df

def create_raw_data_visualizations(df):
    """Create comprehensive visualizations of raw data"""
    print(f"\n📊 Creating comprehensive data visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('../results/plots', exist_ok=True)
    
    # Figure 1: APY and TVL Distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DeFi Protocols: Raw Data Overview', fontsize=16, fontweight='bold')
    
    # APY Distribution (log scale for extreme values)
    if 'apy' in df.columns:
        # Remove extreme outliers for visualization
        apy_clean = df['apy'][(df['apy'] > 0) & (df['apy'] < df['apy'].quantile(0.99))]
        
        axes[0,0].hist(apy_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('APY (%)')
        axes[0,0].set_ylabel('Number of Protocols')
        axes[0,0].set_title('APY Distribution (99th percentile filtered)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_apy = apy_clean.mean()
        median_apy = apy_clean.median()
        axes[0,0].axvline(mean_apy, color='red', linestyle='--', label=f'Mean: {mean_apy:.1f}%')
        axes[0,0].axvline(median_apy, color='orange', linestyle='--', label=f'Median: {median_apy:.1f}%')
        axes[0,0].legend()
    
    # TVL Distribution (log scale)
    if 'tvlUsd' in df.columns:
        tvl_positive = df['tvlUsd'][df['tvlUsd'] > 0]
        axes[0,1].hist(np.log10(tvl_positive), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_xlabel('TVL (log10 USD)')
        axes[0,1].set_ylabel('Number of Protocols')
        axes[0,1].set_title('TVL Distribution (Log Scale)')
        axes[0,1].grid(True, alpha=0.3)
    
    # APY vs TVL Scatter Plot
    if 'apy' in df.columns and 'tvlUsd' in df.columns:
        # Filter for reasonable values
        mask = (df['apy'] > 0) & (df['apy'] < 200) & (df['tvlUsd'] > 1000)
        plot_data = df[mask]
        
        scatter = axes[1,0].scatter(plot_data['tvlUsd'], plot_data['apy'], 
                                   alpha=0.6, c=plot_data['apy'], cmap='viridis', s=20)
        axes[1,0].set_xscale('log')
        axes[1,0].set_xlabel('TVL (USD, log scale)')
        axes[1,0].set_ylabel('APY (%)')
        axes[1,0].set_title('APY vs TVL Relationship')
        axes[1,0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,0], label='APY (%)')
    
    # Chain Distribution
    if 'chain' in df.columns:
        chain_counts = df['chain'].value_counts().head(10)
        axes[1,1].bar(range(len(chain_counts)), chain_counts.values, color='coral')
        axes[1,1].set_xlabel('Blockchain')
        axes[1,1].set_ylabel('Number of Protocols')
        axes[1,1].set_title('Top 10 Blockchains by Protocol Count')
        axes[1,1].set_xticks(range(len(chain_counts)))
        axes[1,1].set_xticklabels(chain_counts.index, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/tree_models_raw_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Quality Metrics Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Quality and Investment Criteria Analysis', fontsize=16, fontweight='bold')
    
    # TVL Threshold Analysis
    if 'tvlUsd' in df.columns:
        tvl_thresholds = [1e6, 5e6, 10e6, 20e6, 50e6, 100e6]
        counts = [len(df[df['tvlUsd'] >= threshold]) for threshold in tvl_thresholds]
        
        axes[0,0].plot([t/1e6 for t in tvl_thresholds], counts, marker='o', linewidth=2, markersize=8)
        axes[0,0].set_xlabel('Minimum TVL (Million USD)')
        axes[0,0].set_ylabel('Number of Protocols')
        axes[0,0].set_title('Protocol Count by TVL Threshold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Highlight our chosen threshold
        axes[0,0].axvline(10, color='red', linestyle='--', label='Our threshold: $10M')
        axes[0,0].legend()
    
    # APY Range Analysis
    if 'apy' in df.columns:
        apy_ranges = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 100), (100, float('inf'))]
        range_counts = []
        range_labels = []
        
        for min_apy, max_apy in apy_ranges:
            count = len(df[(df['apy'] >= min_apy) & (df['apy'] < max_apy)])
            range_counts.append(count)
            if max_apy == float('inf'):
                range_labels.append(f'{min_apy}%+')
            else:
                range_labels.append(f'{min_apy}-{max_apy}%')
        
        axes[0,1].bar(range_labels, range_counts, color='lightblue', alpha=0.8)
        axes[0,1].set_xlabel('APY Range')
        axes[0,1].set_ylabel('Number of Protocols')
        axes[0,1].set_title('Protocol Distribution by APY Range')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # Volume vs TVL Analysis
    if 'volumeUsd1d' in df.columns and 'tvlUsd' in df.columns:
        # Calculate turnover ratio
        df_vol = df[(df['volumeUsd1d'] > 0) & (df['tvlUsd'] > 0)].copy()
        df_vol['turnover'] = df_vol['volumeUsd1d'] / df_vol['tvlUsd']
        
        # Filter reasonable values
        df_vol_clean = df_vol[df_vol['turnover'] < df_vol['turnover'].quantile(0.95)]
        
        axes[1,0].scatter(df_vol_clean['tvlUsd'], df_vol_clean['turnover'], alpha=0.6, s=20)
        axes[1,0].set_xscale('log')
        axes[1,0].set_xlabel('TVL (USD, log scale)')
        axes[1,0].set_ylabel('Daily Turnover Ratio')
        axes[1,0].set_title('Trading Activity vs Pool Size')
        axes[1,0].grid(True, alpha=0.3)
    
    # Quality Score Distribution
    if all(col in df.columns for col in ['tvlUsd', 'volumeUsd1d', 'apy']):
        # Create a simple quality score
        df_quality = df.copy()
        df_quality['tvl_score'] = (df_quality['tvlUsd'] >= 10e6).astype(int)
        df_quality['volume_score'] = (df_quality['volumeUsd1d'] >= 500e3).astype(int)
        df_quality['apy_score'] = ((df_quality['apy'] >= 1) & (df_quality['apy'] <= 150)).astype(int)
        df_quality['quality_score'] = df_quality['tvl_score'] + df_quality['volume_score'] + df_quality['apy_score']
        
        quality_counts = df_quality['quality_score'].value_counts().sort_index()
        
        axes[1,1].bar(quality_counts.index, quality_counts.values, color='gold', alpha=0.8)
        axes[1,1].set_xlabel('Quality Score (0-3)')
        axes[1,1].set_ylabel('Number of Protocols')
        axes[1,1].set_title('Protocol Quality Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add annotations
        for i, v in enumerate(quality_counts.values):
            axes[1,1].text(quality_counts.index[i], v + len(df)*0.01, str(v), 
                          ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/plots/tree_models_data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Raw data visualizations saved to ../results/plots/")

def calculate_mape_safe(y_true, y_pred):
    """Calculate MAPE safely, handling edge cases"""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_tree_models(X, y, feature_names):
    """Train tree-based models for yield prediction"""
    print(f"\n🌳 TRAINING TREE-BASED MODELS")
    print("="*60)
    
    # Define tree-based models
    models = {}
    
    # Random Forest (always available)
    models['Random_Forest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=False
        )
    
    # Use Leave-One-Out cross-validation for small datasets
    loo = LeaveOneOut()
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
            rmse_cv = np.sqrt(-cv_scores.mean())
            
            # Full model training for additional metrics
            model.fit(X, y)
            y_pred_full = model.predict(X)
            
            # Calculate comprehensive metrics
            r2 = r2_score(y, y_pred_full)
            mae = mean_absolute_error(y, y_pred_full)
            mape = calculate_mape_safe(y, y_pred_full)
            
            # Store results
            results[name] = {
                'model': model,
                'rmse_cv': rmse_cv,
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'cv_scores': cv_scores,
                'y_pred': y_pred_full
            }
            
            print(f"  ✅ RMSE (CV): {rmse_cv:.3f}")
            print(f"  ✅ R² Score: {r2:.3f}")
            print(f"  ✅ MAE: {mae:.3f}")
            print(f"  ✅ MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")
            results[name] = None
    
    return results

def analyze_tree_feature_importance(models, feature_names):
    """Analyze feature importance for tree-based models"""
    print(f"\n📊 TREE-BASED FEATURE IMPORTANCE ANALYSIS")
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
        
        print("\n🔍 TOP 10 MOST IMPORTANT FEATURES (Random Forest):")
        print("-" * 70)
        print(f"{'Rank':<4} {'Feature':<25} {'Importance':<12} {'Scaled %':<10}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1:<4} {row['Feature'][:24]:<25} {row['Importance']:<12.4f} {row['Scaled_Importance']:<10.1f}%")
        
        return importance_df
    
    return None

def create_tree_model_visualizations(models, y, feature_importance=None):
    """Create comprehensive tree model comparison visualizations"""
    print(f"\n📊 Creating tree model comparison visualizations...")
    
    # Figure 3: Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tree-Based Models: Performance Comparison', fontsize=16, fontweight='bold')
    
    # Model metrics comparison
    model_names = []
    rmse_scores = []
    r2_scores = []
    mape_scores = []
    
    for name, result in models.items():
        if result is not None:
            model_names.append(name.replace('_', ' '))
            rmse_scores.append(result['rmse_cv'])
            r2_scores.append(result['r2'])
            mape_scores.append(result['mape'])
    
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
    
    # MAPE Comparison
    axes[1,0].bar(model_names, mape_scores, color='coral', alpha=0.8)
    axes[1,0].set_ylabel('MAPE (%)')
    axes[1,0].set_title('Model MAPE Comparison (Lower is Better)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Prediction vs Actual (Best Model)
    best_model_name = min(models.keys(), key=lambda k: models[k]['mape'] if models[k] else float('inf'))
    if models[best_model_name] is not None:
        y_pred_best = models[best_model_name]['y_pred']
        
        axes[1,1].scatter(y, y_pred_best, alpha=0.7, s=50)
        axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        axes[1,1].set_xlabel('Actual APY (%)')
        axes[1,1].set_ylabel('Predicted APY (%)')
        axes[1,1].set_title(f'Best Model: {best_model_name.replace("_", " ")}')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/tree_models_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Feature Importance Analysis
    if feature_importance is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tree Models: Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Feature importance plot
        top_features = feature_importance.head(15)  # Show more features for trees
        
        axes[0,0].barh(range(len(top_features)), top_features['Importance'], color='forestgreen', alpha=0.7)
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels(top_features['Feature'])
        axes[0,0].set_xlabel('Feature Importance')
        axes[0,0].set_title('Top 15 Feature Importances (Random Forest)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Feature importance distribution
        axes[0,1].hist(feature_importance['Importance'], bins=20, alpha=0.7, color='darkgreen')
        axes[0,1].set_xlabel('Feature Importance Value')
        axes[0,1].set_ylabel('Number of Features')
        axes[0,1].set_title('Feature Importance Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Cumulative importance
        sorted_importance = feature_importance.sort_values('Importance', ascending=False)
        cumsum = np.cumsum(sorted_importance['Importance'])
        
        axes[1,0].plot(range(1, len(cumsum)+1), cumsum, marker='o', linewidth=2, markersize=4)
        axes[1,0].set_xlabel('Number of Features')
        axes[1,0].set_ylabel('Cumulative Importance')
        axes[1,0].set_title('Cumulative Feature Importance')
        axes[1,0].grid(True, alpha=0.3)
        
        # Top features pie chart
        top_10_features = feature_importance.head(10)
        others_importance = feature_importance.iloc[10:]['Importance'].sum()
        
        pie_data = list(top_10_features['Importance']) + [others_importance]
        pie_labels = list(top_10_features['Feature']) + ['Others']
        
        axes[1,1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Feature Importance Distribution (Top 10 + Others)')
        
        plt.tight_layout()
        plt.savefig('../results/plots/tree_models_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_tree_models_summary(models, data_info):
    """Generate comprehensive tree models analysis summary"""
    print(f"\n📋 COMPREHENSIVE TREE MODELS ANALYSIS SUMMARY")
    print("="*80)
    
    # Model Performance Summary
    print(f"\n🏆 TREE MODEL PERFORMANCE RANKING:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Model':<20} {'RMSE':<8} {'R²':<8} {'MAPE':<10} {'Status':<15}")
    print("-" * 80)
    
    # Sort models by MAPE
    sorted_models = sorted(
        [(name, result) for name, result in models.items() if result is not None],
        key=lambda x: x[1]['mape']
    )
    
    for i, (name, result) in enumerate(sorted_models):
        status = "🥇 Best" if i == 0 else "🥈 Good" if i == 1 else "📊 Baseline"
        print(f"{i+1:<4} {name.replace('_', ' '):<20} {result['rmse_cv']:<8.3f} {result['r2']:<8.3f} "
              f"{result['mape']:<10.1f}% {status:<15}")
    
    # Data Quality Summary
    print(f"\n📊 DATA QUALITY ASSESSMENT:")
    print("-" * 50)
    print(f"  Dataset size: {data_info['size']} high-quality protocols")
    print(f"  Feature count: {data_info['features']} clean features")
    print(f"  APY range: {data_info['apy_min']:.1f}% to {data_info['apy_max']:.1f}%")
    print(f"  Average TVL: ${data_info['avg_tvl']:,.0f}")
    print(f"  Quality criteria: TVL ≥ $10M, Volume ≥ $500K, APY 1-150%")
    
    # Model Interpretation
    print(f"\n💡 TREE MODELS INTERPRETATION:")
    print("-" * 50)
    best_model_name, best_result = sorted_models[0]
    
    if best_result['mape'] < 20:
        quality = "EXCELLENT"
        recommendation = "Suitable for institutional investment decisions"
    elif best_result['mape'] < 30:
        quality = "GOOD"
        recommendation = "Acceptable for informed investment analysis"
    elif best_result['mape'] < 50:
        quality = "MODERATE"
        recommendation = "Use with caution, consider as one factor among many"
    else:
        quality = "POOR"
        recommendation = "Not recommended for investment decisions"
    
    print(f"  Model quality: {quality}")
    print(f"  Best model: {best_model_name.replace('_', ' ')}")
    print(f"  Prediction accuracy: {best_result['mape']:.1f}% average error")
    print(f"  Investment recommendation: {recommendation}")
    
    # Risk Assessment
    print(f"\n⚠️ TREE MODELS RISK ASSESSMENT:")
    print("-" * 50)
    print(f"  ✅ Data leakage: ELIMINATED (no APY-derived features)")
    print(f"  ⚠️ Model type: TREE-BASED (complex, potential overfitting)")
    print(f"  ✅ Validation: Leave-one-out (maximizes small dataset usage)")
    print(f"  ⚠️ Overfitting risk: MODERATE (tree models on small datasets)")
    print(f"  ✅ Feature importance: AVAILABLE (tree-based importances)")
    print(f"  ⚠️ Interpretability: MODERATE (ensemble models less interpretable)")
    
    return sorted_models

def main_tree_analysis():
    """Main function for comprehensive tree-based analysis"""
    print("="*80)
    print("🌳 COMPREHENSIVE TREE-BASED MODELING ANALYSIS")
    print("   DeFi Yield Prediction with Tree Models (XGBoost, LightGBM, CatBoost, RF)")
    print("="*80)
    
    data_path = '../data/sample_defi_data_small.json'  # Updated path for new structure
    
    # Step 1: Load and explore raw data
    print("\n📊 Step 1: Loading and exploring raw data...")
    df_raw = load_data(data_path)
    if df_raw is None:
        print("Failed to load data")
        return None
    
    # Explore raw data comprehensively
    df_explored = explore_raw_data(df_raw)
    
    # Create raw data visualizations
    create_raw_data_visualizations(df_explored)
    
    # Step 2: Apply proper preprocessing with financial constraints
    print(f"\n🔧 Step 2: Applying financial-grade preprocessing...")
    result = improved_preprocess_pipeline(
        df_raw, target='apy', min_apy=1, max_apy=150, 
        min_tvl=10000000, min_volume=500000
    )
    
    if result[0] is None:
        print("No data remaining after preprocessing")
        return None
    
    X, y, numeric_features, categorical_features, binary_features, processed_df = result
    
    # Combine numeric and binary features for modeling
    model_features = numeric_features + binary_features
    X_model = X[model_features]
    
    print(f"\nFinal modeling dataset:")
    print(f"  Samples: {len(X_model)}")
    print(f"  Features: {len(model_features)}")
    print(f"  Feature types: {len(numeric_features)} numeric + {len(binary_features)} binary")
    
    # Step 3: Train tree-based models
    print(f"\n🌳 Step 3: Training tree-based models...")
    models = train_tree_models(X_model.values, y.values, model_features)
    
    # Step 4: Feature importance analysis
    print(f"\n📊 Step 4: Analyzing tree-based feature importance...")
    feature_importance = analyze_tree_feature_importance(models, model_features)
    
    # Step 5: Create comprehensive visualizations
    print(f"\n📈 Step 5: Creating comprehensive visualizations...")
    create_tree_model_visualizations(models, y.values, feature_importance)
    
    # Step 6: Generate tree models summary
    print(f"\n📋 Step 6: Generating tree models analysis summary...")
    data_info = {
        'size': len(X_model),
        'features': len(model_features),
        'apy_min': y.min(),
        'apy_max': y.max(),
        'avg_tvl': processed_df['tvlUsd'].mean()
    }
    
    final_results = generate_tree_models_summary(models, data_info)
    
    # Step 7: Save results
    print(f"\n💾 Step 7: Saving tree models analysis results...")
    
    # Save model comparison
    if final_results:
        comparison_df = pd.DataFrame([
            {
                'Model': name.replace('_', ' '),
                'RMSE': result['rmse_cv'],
                'R2_Score': result['r2'],
                'MAPE_Percent': result['mape'],
                'MAE': result['mae']
            }
            for name, result in final_results
        ])
        comparison_df.to_csv('../results/reports/tree_models_comparison.csv', index=False)
        print("   ✅ Saved: ../results/reports/tree_models_comparison.csv")
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv('../results/reports/tree_models_feature_importance.csv', index=False)
        print("   ✅ Saved: ../results/reports/tree_models_feature_importance.csv")
    
    print(f"\n✅ COMPREHENSIVE TREE MODELS ANALYSIS COMPLETED!")
    print(f"📁 Generated files (organized structure):")
    print(f"   • ../results/plots/tree_models_raw_data_overview.png")
    print(f"   • ../results/plots/tree_models_data_quality_analysis.png") 
    print(f"   • ../results/plots/tree_models_performance_comparison.png")
    print(f"   • ../results/plots/tree_models_feature_importance.png")
    print(f"   • ../results/reports/tree_models_comparison.csv")
    print(f"   • ../results/reports/tree_models_feature_importance.csv")
    
    return final_results, models, processed_df

if __name__ == "__main__":
    results = main_tree_analysis() 
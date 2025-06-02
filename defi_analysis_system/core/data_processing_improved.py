#!/usr/bin/env python3
"""
Improved Data Processing Module for DeFi Yield Prediction

This module implements the recommendations from data quality analysis:
1. Removes data leakage features
2. Filters problematic target values
3. Uses only clean, reliable features
4. Implements proper data quality controls
"""

import pandas as pd
import numpy as np
import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_data(file_path):
    """
    Load data with improved error handling and proper JSON structure parsing
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            # Try to read as JSON first, then CSV
            try:
                df = pd.read_json(file_path)
            except:
                df = pd.read_csv(file_path)
        
        print(f"Raw data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check if data is nested (common structure: status, data columns)
        if 'data' in df.columns and 'status' in df.columns:
            print("Detected nested JSON structure, extracting data...")
            
            # Filter successful entries
            successful_rows = df[df['status'] == 'success']
            print(f"Found {len(successful_rows)} successful entries")
            
            # Extract data from nested structure
            data_list = []
            for idx, row in successful_rows.iterrows():
                try:
                    if isinstance(row['data'], dict):
                        data_list.append(row['data'])
                    elif isinstance(row['data'], str):
                        # Try to parse as JSON if it's a string
                        data_list.append(json.loads(row['data']))
                except Exception as e:
                    print(f"Warning: Could not parse row {idx}: {e}")
                    continue
            
            if data_list:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(data_list)
                print(f"Extracted data: {df.shape[0]} rows and {df.shape[1]} columns")
            else:
                print("No valid data found in nested structure")
                return None
        
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def filter_target_variable(df, target='apy', min_apy=1, max_apy=150, min_tvl=10000000, min_volume=500000):
    """
    Filter target variable, TVL, and Volume to focus on realistic, significant, and active pools
    
    New criteria:
    - APY range: [1-150]% (ultra-comprehensive yield range)
    - Minimum TVL: $10M (high-quality pools only)
    - Minimum Daily Volume: $500K (active trading requirement)
    """
    print(f"=== Filtering Target Variable, TVL, and Volume ===")
    
    if target not in df.columns:
        print(f"Target column '{target}' not found")
        return df
    
    if 'tvlUsd' not in df.columns:
        print("TVL column 'tvlUsd' not found")
        return df
    
    if 'volumeUsd1d' not in df.columns:
        print("Volume column 'volumeUsd1d' not found - continuing without volume filter")
        min_volume = 0  # Disable volume filtering if column doesn't exist
    
    original_count = len(df)
    
    print(f"Original dataset size: {original_count}")
    print(f"Target statistics before filtering:")
    print(f"  Min APY: {df[target].min():.4f}")
    print(f"  Max APY: {df[target].max():.4f}")
    print(f"  Mean APY: {df[target].mean():.4f}")
    print(f"  Median APY: {df[target].median():.4f}")
    
    print(f"TVL statistics before filtering:")
    print(f"  Min TVL: ${df['tvlUsd'].min():,.0f}")
    print(f"  Max TVL: ${df['tvlUsd'].max():,.0f}")
    print(f"  Mean TVL: ${df['tvlUsd'].mean():,.0f}")
    print(f"  Median TVL: ${df['tvlUsd'].median():,.0f}")
    
    if 'volumeUsd1d' in df.columns and min_volume > 0:
        print(f"Volume statistics before filtering:")
        print(f"  Min Volume: ${df['volumeUsd1d'].min():,.0f}")
        print(f"  Max Volume: ${df['volumeUsd1d'].max():,.0f}")
        print(f"  Mean Volume: ${df['volumeUsd1d'].mean():,.0f}")
        print(f"  Median Volume: ${df['volumeUsd1d'].median():,.0f}")
    
    # Count pools meeting each criterion
    apy_in_range = ((df[target] >= min_apy) & (df[target] <= max_apy)).sum()
    tvl_above_min = (df['tvlUsd'] >= min_tvl).sum()
    
    if 'volumeUsd1d' in df.columns and min_volume > 0:
        volume_above_min = (df['volumeUsd1d'] >= min_volume).sum()
        all_criteria = ((df[target] >= min_apy) & (df[target] <= max_apy) & 
                       (df['tvlUsd'] >= min_tvl) & (df['volumeUsd1d'] >= min_volume)).sum()
        
        print(f"\nFiltering criteria:")
        print(f"  APY in [{min_apy}-{max_apy}]: {apy_in_range} pools ({apy_in_range/original_count*100:.1f}%)")
        print(f"  TVL >= ${min_tvl:,}: {tvl_above_min} pools ({tvl_above_min/original_count*100:.1f}%)")
        print(f"  Volume >= ${min_volume:,}: {volume_above_min} pools ({volume_above_min/original_count*100:.1f}%)")
        print(f"  All criteria: {all_criteria} pools ({all_criteria/original_count*100:.1f}%)")
    else:
        all_criteria = ((df[target] >= min_apy) & (df[target] <= max_apy) & 
                       (df['tvlUsd'] >= min_tvl)).sum()
        
        print(f"\nFiltering criteria:")
        print(f"  APY in [{min_apy}-{max_apy}]: {apy_in_range} pools ({apy_in_range/original_count*100:.1f}%)")
        print(f"  TVL >= ${min_tvl:,}: {tvl_above_min} pools ({tvl_above_min/original_count*100:.1f}%)")
        print(f"  Both criteria: {all_criteria} pools ({all_criteria/original_count*100:.1f}%)")
    
    # Apply filters
    filters_applied = []
    
    # Filter by APY range
    apy_filter = (df[target] >= min_apy) & (df[target] <= max_apy)
    if not apy_filter.all():
        removed_apy = (~apy_filter).sum()
        df = df[apy_filter]
        filters_applied.append(f"Removed {removed_apy} pools outside APY range [{min_apy}-{max_apy}]")
    
    # Filter by minimum TVL
    tvl_filter = df['tvlUsd'] >= min_tvl
    if not tvl_filter.all():
        removed_tvl = (~tvl_filter).sum()
        df = df[tvl_filter]
        filters_applied.append(f"Removed {removed_tvl} pools with TVL < ${min_tvl:,}")
    
    # Filter by minimum volume if available
    if 'volumeUsd1d' in df.columns and min_volume > 0:
        volume_filter = df['volumeUsd1d'] >= min_volume
        if not volume_filter.all():
            removed_volume = (~volume_filter).sum()
            df = df[volume_filter]
            filters_applied.append(f"Removed {removed_volume} pools with Volume < ${min_volume:,}")
    
    final_count = len(df)
    print(f"\nFinal dataset size: {final_count}")
    print(f"Removed {original_count - final_count} pools ({(original_count - final_count)/original_count*100:.1f}%)")
    
    if filters_applied:
        print("Filters applied:")
        for filter_desc in filters_applied:
            print(f"  - {filter_desc}")
    
    if final_count > 0:
        print(f"\nTarget statistics after filtering:")
        print(f"  Min APY: {df[target].min():.4f}")
        print(f"  Max APY: {df[target].max():.4f}")
        print(f"  Mean APY: {df[target].mean():.4f}")
        print(f"  Median APY: {df[target].median():.4f}")
        
        print(f"TVL statistics after filtering:")
        print(f"  Min TVL: ${df['tvlUsd'].min():,.0f}")
        print(f"  Max TVL: ${df['tvlUsd'].max():,.0f}")
        print(f"  Mean TVL: ${df['tvlUsd'].mean():,.0f}")
        print(f"  Median TVL: ${df['tvlUsd'].median():,.0f}")
        
        if 'volumeUsd1d' in df.columns and min_volume > 0:
            print(f"Volume statistics after filtering:")
            print(f"  Min Volume: ${df['volumeUsd1d'].min():,.0f}")
            print(f"  Max Volume: ${df['volumeUsd1d'].max():,.0f}")
            print(f"  Mean Volume: ${df['volumeUsd1d'].mean():,.0f}")
            print(f"  Median Volume: ${df['volumeUsd1d'].median():,.0f}")
    else:
        print("⚠️ No pools remain after filtering!")
    
    return df

def remove_problematic_features(df):
    """
    Remove features identified as problematic by data quality analysis
    """
    print("=== Removing Problematic Features ===")
    
    # Features that cause data leakage (directly related to APY)
    data_leakage_features = [
        'apyBase', 'apyReward', 'apyPct1D', 'apyPct7D', 'apyPct30D', 'apyMean30d',
        'yield_efficiency', 'yield_per_risk', 'relative_apy_to_mean', 'recent_trend',
        'apyBase7d', 'apyBaseInception'
    ]
    
    # Features with too many missing values
    high_missing_features = [
        'il7d', 'apyBase7d', 'apyBaseInception'
    ]
    
    # Combine all problematic features
    problematic_features = list(set(data_leakage_features + high_missing_features))
    
    # Remove features that exist in the dataframe
    removed_features = []
    for feature in problematic_features:
        if feature in df.columns:
            df = df.drop(feature, axis=1)
            removed_features.append(feature)
    
    if removed_features:
        print(f"Removed {len(removed_features)} problematic features:")
        for feature in removed_features:
            print(f"  - {feature}")
    else:
        print("No problematic features found to remove")
    
    return df

def create_safe_engineered_features(df):
    """
    Create only safe engineered features that don't leak information
    """
    print("=== Creating Safe Engineered Features ===")
    
    df = df.copy()
    epsilon = 0.001  # Small constant to avoid division by zero
    
    # TVL-based features (safe)
    if 'tvlUsd' in df.columns:
        # Log transformation to handle exponential distribution
        df['log_tvl'] = np.log1p(df['tvlUsd'])
        
        # Percentile-based features to handle outliers
        df['tvl_percentile'] = df['tvlUsd'].rank(pct=True)
        
        # TVL rank within chain (if chain exists)
        if 'chain' in df.columns:
            df['tvl_chain_rank'] = df.groupby('chain')['tvlUsd'].rank(pct=True)
        
        # TVL rank within project (if project exists)
        if 'project' in df.columns:
            df['tvl_project_rank'] = df.groupby('project')['tvlUsd'].rank(pct=True)
    
    # Volume-based features (safe)
    if 'volumeUsd1d' in df.columns:
        # Log transformation for volume
        df['log_volume_1d'] = np.log1p(df['volumeUsd1d'])
        
        # Volume percentile
        df['volume_percentile'] = df['volumeUsd1d'].rank(pct=True)
        
        # Volume to TVL ratio (turnover ratio) - only if both exist
        if 'tvlUsd' in df.columns:
            df['volume_tvl_ratio'] = df['volumeUsd1d'] / (df['tvlUsd'] + epsilon)
            df['log_volume_tvl_ratio'] = np.log1p(df['volume_tvl_ratio'])
    
    # Risk-based features (safe if not derived from APY)
    if 'sigma' in df.columns and 'mu' in df.columns:
        # Sharpe-like ratio (risk-adjusted return)
        df['risk_adjusted_return'] = df['mu'] / (df['sigma'] + epsilon)
        
        # Volatility rank
        df['sigma_percentile'] = df['sigma'].rank(pct=True)
        df['mu_percentile'] = df['mu'].rank(pct=True)
    
    # Token-based features
    if 'underlyingTokens' in df.columns:
        def count_tokens_safe(tokens):
            try:
                if pd.isna(tokens) or tokens is None:
                    return 0
                if isinstance(tokens, str):
                    if tokens.lower() in ['nan', 'none', '']:
                        return 0
                    if ',' in tokens:
                        return len([t.strip() for t in tokens.split(',') if t.strip()])
                    else:
                        return 1 if tokens.strip() else 0
                return 1
            except:
                return 0
        
        df['n_underlying_tokens'] = df['underlyingTokens'].apply(count_tokens_safe)
        df['is_single_token'] = (df['n_underlying_tokens'] == 1).astype(int)
    
    if 'rewardTokens' in df.columns:
        def count_rewards_safe(tokens):
            try:
                if pd.isna(tokens) or tokens is None:
                    return 0
                if isinstance(tokens, str):
                    if tokens.lower().strip() in ['nan', 'none', '']:
                        return 0
                    if ',' in tokens:
                        return len([t.strip() for t in tokens.split(',') if t.strip()])
                    else:
                        return 1 if tokens.strip() else 0
                return 1
            except:
                return 0
        
        df['n_reward_tokens'] = df['rewardTokens'].apply(count_rewards_safe)
        df['has_multiple_rewards'] = (df['n_reward_tokens'] > 1).astype(int)
    
    # Chain-based features
    if 'chain' in df.columns:
        # Major chains identification
        major_chains = ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche']
        df['is_major_chain'] = df['chain'].str.lower().isin(major_chains).astype(int)
        
        # Chain statistics (aggregated safely)
        try:
            chain_tvl_stats = df.groupby('chain')['tvlUsd'].agg(['count', 'median']).reset_index()
            chain_tvl_stats.columns = ['chain', 'chain_protocol_count', 'chain_median_tvl']
            df = df.merge(chain_tvl_stats, on='chain', how='left')
        except:
            pass
    
    # Project-based features
    if 'project' in df.columns:
        # Project size (number of pools)
        try:
            project_counts = df['project'].value_counts().to_dict()
            df['project_pool_count'] = df['project'].map(project_counts)
            df['is_large_project'] = (df['project_pool_count'] > 10).astype(int)
        except:
            pass
    
    # Symbol-based features
    if 'symbol' in df.columns:
        # Popular token identification
        popular_tokens = ['USDC', 'USDT', 'DAI', 'WETH', 'WBTC', 'ETH']
        df['has_popular_token'] = df['symbol'].str.contains('|'.join(popular_tokens), case=False, na=False).astype(int)
        
        # Stablecoin pair identification
        stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD', 'FRAX']
        df['is_stablecoin_pair'] = df['symbol'].str.contains('|'.join(stablecoins), case=False, na=False).astype(int)
    
    print("Safe engineered features created")
    return df

def prepare_clean_model_data(df, target='apy'):
    """
    Prepare data for modeling using only clean, reliable features
    """
    print("=== Preparing Clean Model Data ===")
    
    # Define clean feature sets based on data quality analysis
    clean_numeric_features = [
        # Core safe numeric features
        'tvlUsd', 'volumeUsd1d', 'sigma', 'mu', 'count',
        # Safe engineered features
        'log_tvl', 'log_volume_1d', 'volume_tvl_ratio', 'log_volume_tvl_ratio',
        'tvl_percentile', 'volume_percentile', 'sigma_percentile', 'mu_percentile',
        'risk_adjusted_return', 'tvl_chain_rank', 'tvl_project_rank',
        'n_underlying_tokens', 'n_reward_tokens', 'project_pool_count',
        'chain_protocol_count', 'chain_median_tvl'
    ]
    
    clean_categorical_features = [
        'project', 'symbol', 'chain', 'exposure', 'ilRisk', 
        'underlyingTokens', 'rewardTokens', 'pool'
    ]
    
    clean_binary_features = [
        'stablecoin', 'is_single_token', 'has_multiple_rewards',
        'is_major_chain', 'is_large_project', 'has_popular_token', 'is_stablecoin_pair'
    ]
    
    # Filter to only include features that exist in the dataframe
    available_numeric = [f for f in clean_numeric_features if f in df.columns]
    available_categorical = [f for f in clean_categorical_features if f in df.columns]
    available_binary = [f for f in clean_binary_features if f in df.columns]
    
    print(f"Available clean features:")
    print(f"  Numeric: {len(available_numeric)} features")
    print(f"  Categorical: {len(available_categorical)} features") 
    print(f"  Binary: {len(available_binary)} features")
    
    # Combine all features
    all_features = available_numeric + available_categorical + available_binary
    
    # Remove target from features if it exists
    if target in all_features:
        all_features.remove(target)
        if target in available_numeric:
            available_numeric.remove(target)
    
    # Create X and y
    X = df[all_features].copy()
    y = df[target].copy()
    
    # Handle missing values properly
    print("Processing missing values...")
    
    # Numeric features: fill with median
    for col in available_numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())
    
    # Categorical features: fill with 'unknown'
    for col in available_categorical:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna('unknown')
    
    # Binary features: fill with 0
    for col in available_binary:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset prepared:")
    print(f"  Rows: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target missing values: {y.isna().sum()}")
    
    return X, y, available_numeric, available_categorical, available_binary

def improved_preprocess_pipeline(df, target='apy', min_apy=1, max_apy=150, min_tvl=10000000, min_volume=500000):
    """
    Complete improved preprocessing pipeline with stricter filtering
    """
    print("=== Improved Data Preprocessing Pipeline (Strict Filtering) ===")
    
    # Step 1: Filter target variable and TVL for significant pools only
    df = filter_target_variable(df, target, min_apy, max_apy, min_tvl, min_volume)
    
    if len(df) == 0:
        print("No data remaining after filtering. Consider relaxing criteria.")
        return None, None, None, None, None, None
    
    # Step 2: Remove problematic features
    df = remove_problematic_features(df)
    
    # Step 3: Create safe engineered features
    df = create_safe_engineered_features(df)
    
    # Step 4: Prepare clean model data
    X, y, numeric_features, categorical_features, binary_features = prepare_clean_model_data(df, target)
    
    print(f"\n=== Preprocessing Summary (Enhanced Filtering) ===")
    print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Feature types:")
    print(f"  - Numeric: {len(numeric_features)}")
    print(f"  - Categorical: {len(categorical_features)}")
    print(f"  - Binary: {len(binary_features)}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    print(f"Target mean: {y.mean():.4f} ± {y.std():.4f}")
    print(f"Focused on premium pools (TVL >= ${min_tvl:,}, Volume >= ${min_volume:,}) with realistic yields ({min_apy}-{max_apy}%)")
    
    return X, y, numeric_features, categorical_features, binary_features, df

if __name__ == "__main__":
    # Test the improved preprocessing
    import os
    DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'source_data', 'defilama_data.json')
    
    print("Testing improved preprocessing pipeline...")
    
    # Load data
    df = load_data(DEFAULT_DATA_FILE)
    if df is None:
        print("Failed to load data")
        exit(1)
    
    # Run improved preprocessing
    X, y, numeric_features, categorical_features, binary_features, processed_df = improved_preprocess_pipeline(df)
    
    print("\nPreprocessing completed successfully!")
    print(f"Ready for model training with {X.shape[1]} clean features") 
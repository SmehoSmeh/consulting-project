#!/usr/bin/env python3
"""
Stock Data Processing Module - Enhanced Version

This module handles loading and preprocessing of stock price data:
1. Load stock price, profile, and return data from parquet files
2. Create advanced features for machine learning models
3. Handle missing data and outliers with robust methods
4. Normalize and scale features for better ML performance
5. Create sector and industry classifications
6. Clean feature names for ML library compatibility
"""

import pandas as pd
import numpy as np
import os
import warnings
import re
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
warnings.filterwarnings('ignore')

def load_stock_data():
    """Load stock data from the hse-portfolio-stocks data directory"""
    print("📊 LOADING STOCK DATA...")
    
    # Get the absolute path to the workspace root
    current_dir = os.path.dirname(os.path.abspath(__file__))  # core directory
    workspace_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from core/
    data_root = os.path.join(workspace_root, "hse-portfolio-stocks", "data")
    
    # Alternative path if running from different location
    if not os.path.exists(data_root):
        # Try from current working directory
        data_root = os.path.join(os.getcwd(), "hse-portfolio-stocks", "data")
    
    # Another alternative if still not found
    if not os.path.exists(data_root):
        # Try parent directory of current working directory
        data_root = os.path.join(os.path.dirname(os.getcwd()), "hse-portfolio-stocks", "data")
    
    raw_path = os.path.join(data_root, "raw")
    processed_path = os.path.join(data_root, "processed")
    
    # Debug: print the actual paths being used
    print(f"  📍 Looking for data in: {os.path.abspath(data_root)}")
    print(f"  📍 Current working directory: {os.getcwd()}")
    print(f"  📍 Script location: {current_dir}")
    
    try:
        # Load profile data
        profile_path = os.path.join(raw_path, "profile.parquet")
        print(f"  📍 Profile path: {os.path.abspath(profile_path)}")
        
        if not os.path.exists(profile_path):
            print(f"  ❌ Profile file not found at: {profile_path}")
            print(f"  📁 Raw directory exists: {os.path.exists(raw_path)}")
            if os.path.exists(raw_path):
                print(f"  📁 Raw directory contents: {os.listdir(raw_path)}")
            print(f"  📁 Data root exists: {os.path.exists(data_root)}")
            if os.path.exists(data_root):
                print(f"  📁 Data root contents: {os.listdir(data_root)}")
            return None, None, None
            
        profile_df = pd.read_parquet(profile_path)
        print(f"  ✅ Profile data loaded: {profile_df.shape[0]} stocks")
        
        # Load master processed data (has prices, returns, and metadata)
        master_path = os.path.join(processed_path, "master.parquet")
        if not os.path.exists(master_path):
            print(f"  ❌ Master file not found at: {master_path}")
            print(f"  📁 Processed directory exists: {os.path.exists(processed_path)}")
            if os.path.exists(processed_path):
                print(f"  📁 Processed directory contents: {os.listdir(processed_path)}")
            return None, None, None
            
        master_df = pd.read_parquet(master_path)
        print(f"  ✅ Master data loaded: {master_df.shape[0]} records")
        
        # Load existing predictions if available
        predictions_df = None
        pred_path = os.path.join(processed_path, "predictions.parquet")
        if os.path.exists(pred_path):
            predictions_df = pd.read_parquet(pred_path)
            print(f"  ✅ Existing predictions loaded: {predictions_df.shape[0]} records")
        
        return master_df, profile_df, predictions_df
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        print(f"  📍 Current working directory: {os.getcwd()}")
        return None, None, None

def clean_feature_names(feature_names):
    """Clean feature names to be compatible with all ML libraries"""
    cleaned_names = []
    for name in feature_names:
        # Replace special characters with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Remove multiple consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'feat_' + cleaned
        cleaned_names.append(cleaned)
    return cleaned_names

def create_advanced_technical_features(df, quick_mode=False):
    """Create advanced technical analysis features for stock prediction"""
    print("🔧 CREATING ADVANCED TECHNICAL FEATURES...")
    
    if quick_mode:
        print("  ⚡ QUICK MODE: Creating essential features only")
    
    # Sort by symbol and date to ensure proper time series order
    df = df.sort_values(['symbol', 'Date']).copy()
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    total_symbols = len(symbols)
    print(f"  📊 Processing {total_symbols} stocks with {len(df):,} total records...")
    
    # Create features for each stock
    features_list = []
    processed_count = 0
    
    for i, symbol in enumerate(symbols):
        if i % 50 == 0 or i == total_symbols - 1:  # Progress every 50 stocks
            progress = (i + 1) / total_symbols * 100
            print(f"  📈 Progress: {i+1}/{total_symbols} stocks ({progress:.1f}%) - Processing {symbol}")
        
        try:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:  # Skip stocks with very little data
                print(f"    ⚠️ Skipping {symbol}: insufficient data ({len(symbol_data)} records)")
                continue
            
            # Ensure price and return are numeric
            symbol_data['price'] = pd.to_numeric(symbol_data['price'], errors='coerce')
            symbol_data['return'] = pd.to_numeric(symbol_data['return'], errors='coerce')
            
            # Fill any NaN values that might cause issues
            symbol_data['price'] = symbol_data['price'].fillna(method='ffill')
            symbol_data['return'] = symbol_data['return'].fillna(0)
            
            if quick_mode:
                # Essential features only for quick mode
                symbol_data = create_essential_features(symbol_data)
            else:
                # Full feature set
                symbol_data = create_comprehensive_features(symbol_data)
            
            # Replace infinite values with NaN, then fill with 0
            symbol_data = symbol_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            numeric_cols = symbol_data.select_dtypes(include=[np.number]).columns
            symbol_data[numeric_cols] = symbol_data[numeric_cols].fillna(0)
            
            features_list.append(symbol_data)
            processed_count += 1
            
        except Exception as e:
            print(f"    ❌ Error processing {symbol}: {e}")
            continue
    
    if not features_list:
        print("  ❌ No stocks processed successfully")
        return df
    
    # Combine all symbol data
    print(f"  🔄 Combining data from {processed_count} successfully processed stocks...")
    enhanced_df = pd.concat(features_list, ignore_index=True)
    
    print(f"  ✅ Technical features created successfully!")
    print(f"  📊 Final shape: {enhanced_df.shape}")
    print(f"  📈 Features per stock: {enhanced_df.shape[1] - df.shape[1]} new features added")
    
    return enhanced_df

def create_essential_features(symbol_data):
    """Create essential technical features for quick mode"""
    
    # Basic moving averages (most important)
    symbol_data['price_sma_5'] = symbol_data['price'].rolling(window=5, min_periods=1).mean().astype('float64')
    symbol_data['price_sma_20'] = symbol_data['price'].rolling(window=20, min_periods=1).mean().astype('float64')
    symbol_data['price_sma_50'] = symbol_data['price'].rolling(window=50, min_periods=1).mean().astype('float64')
    
    # Basic volatility
    symbol_data['return_std_5'] = symbol_data['return'].rolling(window=5, min_periods=1).std().astype('float64')
    symbol_data['return_std_20'] = symbol_data['return'].rolling(window=20, min_periods=1).std().astype('float64')
    
    # Price momentum
    symbol_data['price_change_1d'] = symbol_data['price'].pct_change(1).astype('float64')
    symbol_data['price_change_5d'] = symbol_data['price'].pct_change(5).astype('float64')
    symbol_data['price_change_20d'] = symbol_data['price'].pct_change(20).astype('float64')
    
    # RSI (essential momentum indicator)
    symbol_data['rsi_14'] = calculate_rsi_fast(symbol_data['price'], 14).astype('float64')
    
    # Relative price position
    symbol_data['price_relative_to_sma20'] = (symbol_data['price'] / symbol_data['price_sma_20']).astype('float64')
    
    # Return magnitude (volume proxy)
    symbol_data['return_magnitude'] = symbol_data['return'].abs().astype('float64')
    symbol_data['return_magnitude_sma_5'] = symbol_data['return_magnitude'].rolling(window=5, min_periods=1).mean().astype('float64')
    
    return symbol_data

def create_comprehensive_features(symbol_data):
    """Create comprehensive technical features for full mode"""
    
    # Start with essential features
    symbol_data = create_essential_features(symbol_data)
    
    # Extended moving averages
    symbol_data['price_sma_10'] = symbol_data['price'].rolling(window=10, min_periods=1).mean().astype('float64')
    
    # Exponential moving averages
    symbol_data['price_ema_12'] = symbol_data['price'].ewm(span=12, min_periods=1).mean().astype('float64')
    symbol_data['price_ema_26'] = symbol_data['price'].ewm(span=26, min_periods=1).mean().astype('float64')
    
    # MACD
    symbol_data['macd'] = (symbol_data['price_ema_12'] - symbol_data['price_ema_26']).astype('float64')
    symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, min_periods=1).mean().astype('float64')
    symbol_data['macd_histogram'] = (symbol_data['macd'] - symbol_data['macd_signal']).astype('float64')
    
    # Extended volatility
    symbol_data['return_std_10'] = symbol_data['return'].rolling(window=10, min_periods=1).std().astype('float64')
    
    # Bollinger Bands
    symbol_data['bb_upper'] = (symbol_data['price_sma_20'] + 2 * symbol_data['return_std_20']).astype('float64')
    symbol_data['bb_lower'] = (symbol_data['price_sma_20'] - 2 * symbol_data['return_std_20']).astype('float64')
    bb_range = symbol_data['bb_upper'] - symbol_data['bb_lower']
    bb_range = bb_range.replace(0, 1)  # Avoid division by zero
    symbol_data['bb_position'] = ((symbol_data['price'] - symbol_data['bb_lower']) / bb_range).astype('float64')
    
    # Extended momentum
    symbol_data['rsi_7'] = calculate_rsi_fast(symbol_data['price'], 7).astype('float64')
    symbol_data['price_change_3d'] = symbol_data['price'].pct_change(3).astype('float64')
    symbol_data['price_change_10d'] = symbol_data['price'].pct_change(10).astype('float64')
    
    # Momentum ratios
    symbol_data['momentum_5'] = (symbol_data['price'] / symbol_data['price'].shift(5)).astype('float64')
    symbol_data['momentum_10'] = (symbol_data['price'] / symbol_data['price'].shift(10)).astype('float64')
    symbol_data['momentum_20'] = (symbol_data['price'] / symbol_data['price'].shift(20)).astype('float64')
    
    # Extended volume proxy features
    symbol_data['return_magnitude_sma_20'] = symbol_data['return_magnitude'].rolling(window=20, min_periods=1).mean().astype('float64')
    
    # Relative strength features
    symbol_data['price_relative_to_sma5'] = (symbol_data['price'] / symbol_data['price_sma_5']).astype('float64')
    symbol_data['price_relative_to_sma10'] = (symbol_data['price'] / symbol_data['price_sma_10']).astype('float64')
    symbol_data['price_relative_to_sma50'] = (symbol_data['price'] / symbol_data['price_sma_50']).astype('float64')
    
    # Rate of change features
    symbol_data['roc_5'] = (((symbol_data['price'] - symbol_data['price'].shift(5)) / symbol_data['price'].shift(5)) * 100).astype('float64')
    symbol_data['roc_10'] = (((symbol_data['price'] - symbol_data['price'].shift(10)) / symbol_data['price'].shift(10)) * 100).astype('float64')
    symbol_data['roc_20'] = (((symbol_data['price'] - symbol_data['price'].shift(20)) / symbol_data['price'].shift(20)) * 100).astype('float64')
    
    # Williams %R and Stochastic (simplified)
    symbol_data['williams_r'] = calculate_williams_r_fast(symbol_data, 14).astype('float64')
    symbol_data['stoch_k'] = calculate_stochastic_k_fast(symbol_data, 14).astype('float64')
    symbol_data['stoch_d'] = symbol_data['stoch_k'].rolling(window=3, min_periods=1).mean().astype('float64')
    
    # Price percentiles
    symbol_data['price_percentile_20'] = symbol_data['price'].rolling(window=20, min_periods=1).rank(pct=True).astype('float64')
    symbol_data['price_percentile_50'] = symbol_data['price'].rolling(window=50, min_periods=1).rank(pct=True).astype('float64')
    
    # Return percentile
    symbol_data['return_percentile_20'] = symbol_data['return'].rolling(window=20, min_periods=1).rank(pct=True).astype('float64')
    
    # Trend strength
    symbol_data['trend_strength'] = (abs(symbol_data['price_sma_5'] - symbol_data['price_sma_20']) / symbol_data['price_sma_20']).astype('float64')
    
    # Advanced features (only in full mode)
    symbol_data = create_advanced_volatility_features_fast(symbol_data)
    symbol_data = create_seasonal_features_fast(symbol_data)
    symbol_data = create_microstructure_features_fast(symbol_data)
    symbol_data = create_regime_features_fast(symbol_data)
    
    return symbol_data

def calculate_rsi_fast(prices, window=14):
    """Fast RSI calculation with better error handling"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        # Handle division by zero more robustly
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    except:
        return pd.Series(50, index=prices.index)  # Return neutral RSI if calculation fails

def calculate_williams_r_fast(df, window=14):
    """Fast Williams %R calculation"""
    try:
        price_std = df['return_std_20'].fillna(df['return'].rolling(window=20, min_periods=1).std())
        high = df['price'] + price_std
        low = df['price'] - price_std
        
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        williams_r = ((highest_high - df['price']) / (highest_high - lowest_low)) * -100
        return williams_r.fillna(-50)
    except:
        return pd.Series(-50, index=df.index)

def calculate_stochastic_k_fast(df, window=14):
    """Fast Stochastic K calculation"""
    try:
        price_std = df['return_std_20'].fillna(df['return'].rolling(window=20, min_periods=1).std())
        high = df['price'] + price_std
        low = df['price'] - price_std
        
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        stoch_k = ((df['price'] - lowest_low) / (highest_high - lowest_low)) * 100
        return stoch_k.fillna(50)
    except:
        return pd.Series(50, index=df.index)

def create_target_variables(df):
    """Create target variables for prediction"""
    print("🎯 CREATING TARGET VARIABLES...")
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'Date']).copy()
    
    targets_list = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # Forward-looking returns (what we want to predict)
        symbol_data['target_return_1d'] = symbol_data['return'].shift(-1)  # Next day return
        symbol_data['target_return_3d'] = symbol_data['return'].rolling(window=3).mean().shift(-3)  # 3-day forward return
        symbol_data['target_return_5d'] = symbol_data['return'].rolling(window=5).mean().shift(-5)  # 5-day forward return
        symbol_data['target_return_10d'] = symbol_data['return'].rolling(window=10).mean().shift(-10)  # 10-day forward return
        
        # Classification targets (up/down predictions)
        symbol_data['target_direction_1d'] = (symbol_data['target_return_1d'] > 0).astype(int)
        symbol_data['target_direction_3d'] = (symbol_data['target_return_3d'] > 0).astype(int)
        symbol_data['target_direction_5d'] = (symbol_data['target_return_5d'] > 0).astype(int)
        
        # Volatility targets
        symbol_data['target_volatility_5d'] = symbol_data['return'].rolling(window=5).std().shift(-5)
        symbol_data['target_volatility_10d'] = symbol_data['return'].rolling(window=10).std().shift(-10)
        
        # Extreme movement targets (for risk management)
        symbol_data['target_extreme_up'] = (symbol_data['target_return_1d'] > symbol_data['return'].quantile(0.95)).astype(int)
        symbol_data['target_extreme_down'] = (symbol_data['target_return_1d'] < symbol_data['return'].quantile(0.05)).astype(int)
        
        targets_list.append(symbol_data)
    
    result_df = pd.concat(targets_list, ignore_index=True)
    
    print(f"  ✅ Target variables created")
    return result_df

def create_sector_features(df):
    """Create sector and industry-based features"""
    print("🏢 CREATING SECTOR FEATURES...")
    
    # Clean sector and industry names first
    df['sector'] = df['sector'].fillna('Unknown')
    df['industry'] = df['industry'].fillna('Unknown')
    
    # Create dummy variables for sector and industry
    sector_dummies = pd.get_dummies(df['sector'], prefix='sector', dtype='float64')
    industry_dummies = pd.get_dummies(df['industry'], prefix='industry', dtype='float64')
    
    # Limit industry dummies to top 20 to avoid feature explosion
    if industry_dummies.shape[1] > 20:
        # Keep top 20 industries by frequency
        industry_counts = df['industry'].value_counts()
        top_industries = industry_counts.head(20).index
        industry_dummies = pd.get_dummies(df['industry'][df['industry'].isin(top_industries)], prefix='industry', dtype='float64')
        
        # Reindex to match original dataframe
        industry_dummies = industry_dummies.reindex(df.index, fill_value=0.0)
    
    # Combine with original data
    enhanced_df = pd.concat([df, sector_dummies, industry_dummies], axis=1)
    
    # Create sector-level aggregated features
    sector_stats = df.groupby(['Date', 'sector'])['return'].agg(['mean', 'std', 'count']).reset_index()
    sector_stats.columns = ['Date', 'sector', 'sector_mean_return', 'sector_std_return', 'sector_count']
    
    # Ensure sector stats are numeric
    sector_stats['sector_mean_return'] = pd.to_numeric(sector_stats['sector_mean_return'], errors='coerce').astype('float64')
    sector_stats['sector_std_return'] = pd.to_numeric(sector_stats['sector_std_return'], errors='coerce').astype('float64')
    sector_stats['sector_count'] = pd.to_numeric(sector_stats['sector_count'], errors='coerce').astype('float64')
    
    # Only include sector stats if there are enough stocks in sector that day
    sector_stats = sector_stats[sector_stats['sector_count'] >= 3]
    
    # Merge sector statistics
    enhanced_df = enhanced_df.merge(sector_stats, on=['Date', 'sector'], how='left')
    
    # Fill missing sector stats with overall market stats
    enhanced_df['sector_mean_return'] = enhanced_df['sector_mean_return'].fillna(enhanced_df.groupby('Date')['return'].transform('mean')).astype('float64')
    enhanced_df['sector_std_return'] = enhanced_df['sector_std_return'].fillna(enhanced_df.groupby('Date')['return'].transform('std')).astype('float64')
    
    # Create relative performance features
    enhanced_df['return_vs_sector'] = (enhanced_df['return'] - enhanced_df['sector_mean_return']).astype('float64')
    enhanced_df['volatility_vs_sector'] = (enhanced_df['return_std_20'] - enhanced_df['sector_std_return']).astype('float64')
    
    # Clean up any remaining NaN values in new features
    enhanced_df['sector_mean_return'] = enhanced_df['sector_mean_return'].fillna(0.0)
    enhanced_df['sector_std_return'] = enhanced_df['sector_std_return'].fillna(0.0)
    enhanced_df['return_vs_sector'] = enhanced_df['return_vs_sector'].fillna(0.0)
    enhanced_df['volatility_vs_sector'] = enhanced_df['volatility_vs_sector'].fillna(0.0)
    
    print(f"  ✅ Sector features created. New shape: {enhanced_df.shape}")
    return enhanced_df

def robust_outlier_handling(df, feature_cols, method='iqr'):
    """Handle outliers using robust methods"""
    print("🔧 HANDLING OUTLIERS...")
    
    df_clean = df.copy()
    outlier_counts = {}
    
    for col in feature_cols:
        if df_clean[col].dtype in ['float64', 'int64']:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More conservative than 1.5
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    outlier_counts[col] = outlier_count
            
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap using percentiles
                    df_clean[col] = df_clean[col].clip(df_clean[col].quantile(0.01), df_clean[col].quantile(0.99))
                    outlier_counts[col] = outlier_count
    
    total_outliers = sum(outlier_counts.values())
    print(f"  ✅ Handled {total_outliers} outliers across {len(outlier_counts)} features")
    
    return df_clean

def preprocess_for_modeling(df, target_column='target_return_1d', scale_features=True, select_features=True):
    """Prepare data for machine learning models with enhanced preprocessing"""
    print(f"⚙️ ADVANCED PREPROCESSING FOR MODELING (Target: {target_column})...")
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target_column]).copy()
    
    # Define feature columns (exclude non-feature columns)
    exclude_cols = ['Date', 'symbol', 'price', 'return', 'sector', 'industry', 
                   'target_return_1d', 'target_return_3d', 'target_return_5d', 'target_return_10d',
                   'target_direction_1d', 'target_direction_3d', 'target_direction_5d', 
                   'target_volatility_5d', 'target_volatility_10d', 
                   'target_extreme_up', 'target_extreme_down']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # Clean feature names for ML library compatibility
    original_feature_cols = feature_cols.copy()
    cleaned_feature_cols = clean_feature_names(feature_cols)
    
    # Create mapping for renaming
    feature_mapping = dict(zip(original_feature_cols, cleaned_feature_cols))
    df_clean = df_clean.rename(columns=feature_mapping)
    feature_cols = cleaned_feature_cols
    
    # Handle missing values in features
    X = df_clean[feature_cols].copy()
    y = df_clean[target_column].copy()
    
    # CRITICAL: Convert all feature columns to proper numeric types
    print("  🔧 Converting data types...")
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert object columns to numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                print(f"    ✅ Converted {col} from object to numeric")
            except:
                print(f"    ⚠️ Could not convert {col} to numeric, treating as categorical")
                # For categorical columns, convert to category type
                X[col] = X[col].astype('category').cat.codes
        
        # Ensure numeric columns are float64
        if X[col].dtype in ['int64', 'int32', 'float32']:
            X[col] = X[col].astype('float64')
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)  # For dummy variables
    
    # Remove any remaining infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Final data type verification
    print("  🔍 Final data type verification...")
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"    ⚠️ Warning: {len(object_cols)} columns still have object dtype: {list(object_cols)}")
        # Force convert remaining object columns
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype('float64')
        print(f"    ✅ Force converted remaining object columns to float64")
    
    # Verify all columns are now numeric
    final_dtypes = X.dtypes.value_counts()
    print(f"    ✅ Final data types: {dict(final_dtypes)}")
    
    # Advanced outlier detection and handling
    outlier_methods = advanced_outlier_detection(X, feature_cols)
    
    # Apply outlier capping based on multiple methods
    for col in feature_cols:
        if X[col].dtype in ['float64', 'int64']:
            # Use consensus approach - if multiple methods flag as outlier, cap it
            outlier_flags = []
            for method_name, outliers in outlier_methods.items():
                if col in method_name:
                    outlier_flags.append(outliers)
            
            if outlier_flags:
                # Consensus outlier detection
                consensus_outliers = pd.concat(outlier_flags, axis=1).sum(axis=1) >= 2
                if consensus_outliers.sum() > 0:
                    # Cap outliers using robust percentiles
                    Q1 = X[col].quantile(0.05)
                    Q99 = X[col].quantile(0.95)
                    X.loc[consensus_outliers, col] = X.loc[consensus_outliers, col].clip(Q1, Q99)
    
    # Filter out extreme outliers in target variable (more conservative)
    y_clean_mask = (y > y.quantile(0.005)) & (y < y.quantile(0.995))  # Keep 99% of data
    X = X[y_clean_mask]
    y = y[y_clean_mask]
    
    # Feature scaling
    if scale_features:
        print("  🔧 Scaling features...")
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Only scale numeric features, not binary dummy variables
        numeric_features = []
        binary_features = []
        
        for col in feature_cols:
            if X[col].nunique() == 2 and set(X[col].unique()).issubset({0, 1, True, False}):
                binary_features.append(col)
            else:
                numeric_features.append(col)
        
        if numeric_features:
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
            print(f"    ✅ Scaled {len(numeric_features)} numeric features")
            print(f"    ✅ Kept {len(binary_features)} binary features unscaled")
    
    # Feature selection
    if select_features and len(feature_cols) > 50:
        print("  🎯 Selecting top features...")
        
        # Select top K features based on correlation with target
        k = min(50, len(feature_cols))  # Limit to 50 features for better performance
        selector = SelectKBest(score_func=f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        feature_cols = selected_features
        
        print(f"    ✅ Selected top {len(feature_cols)} features")
    
    print(f"  ✅ Data preprocessed. Final shape: {X.shape}, Target shape: {y.shape}")
    print(f"  ✅ Features: {len(feature_cols)} columns")
    print(f"  📊 Target statistics: Mean={y.mean():.6f}, Std={y.std():.6f}")
    print(f"  📊 Target range: [{y.min():.6f}, {y.max():.6f}]")
    
    return X, y, feature_cols

def create_advanced_volatility_features_fast(df):
    """Fast version of advanced volatility features"""
    
    # EWMA volatility (essential)
    df['ewma_volatility'] = df['return'].ewm(alpha=0.06, min_periods=1).std().astype('float64')
    
    # Realized volatility (key timeframes only)
    df['realized_vol_20'] = df['return'].rolling(window=20, min_periods=1).apply(lambda x: np.sqrt(np.sum(x**2))).astype('float64')
    
    # Volatility regime
    df['vol_regime'] = (df['ewma_volatility'] > df['ewma_volatility'].rolling(window=60, min_periods=1).mean()).astype('float64')
    
    # Jump detection
    df['vol_shock'] = (df['return'].abs() > 2 * df['ewma_volatility']).astype('float64')
    
    return df

def create_seasonal_features_fast(df):
    """Fast version of seasonal features"""
    
    # Convert Date to datetime if not already
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic calendar features
        df['month'] = df['Date'].dt.month.astype('float64')
        df['day_of_week'] = df['Date'].dt.dayofweek.astype('float64')
        
        # Key market effects
        df['monday_effect'] = (df['day_of_week'] == 0).astype('float64')
        df['january_effect'] = (df['month'] == 1).astype('float64')
        
        # Cyclical encoding (essential)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float64')
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float64')
    
    return df

def create_microstructure_features_fast(df):
    """Fast version of microstructure features"""
    
    # Ensure return_magnitude exists
    if 'return_magnitude' not in df.columns:
        df['return_magnitude'] = df['return'].abs().astype('float64')
    
    # Price impact proxy
    df['effective_spread'] = df['return_magnitude'].rolling(window=5, min_periods=1).mean().astype('float64')
    
    # Order flow proxy
    df['buy_volume_proxy'] = (df['return'] > 0).astype('float64') * df['return_magnitude']
    df['sell_volume_proxy'] = (df['return'] < 0).astype('float64') * df['return_magnitude']
    
    # Return patterns
    df['return_reversal'] = (df['return'] * df['return'].shift(1) < 0).astype('float64')
    
    return df

def create_regime_features_fast(df):
    """Fast version of regime features"""
    
    # Ensure required features exist
    if 'price_sma_20' not in df.columns:
        df['price_sma_20'] = df['price'].rolling(window=20, min_periods=1).mean().astype('float64')
    if 'price_sma_50' not in df.columns:
        df['price_sma_50'] = df['price'].rolling(window=50, min_periods=1).mean().astype('float64')
    if 'return_std_20' not in df.columns:
        df['return_std_20'] = df['return'].rolling(window=20, min_periods=1).std().astype('float64')
    
    # Trend regime
    df['trend_regime'] = (df['price_sma_20'] > df['price_sma_50']).astype('float64')
    
    # Volatility regime
    current_vol = df['return_std_20']
    long_term_vol = df['return_std_20'].rolling(window=100, min_periods=1).mean()
    df['high_vol_regime'] = (current_vol > 1.5 * long_term_vol).astype('float64')
    
    # Bull/Bear market proxy
    df['bull_market'] = (df['price'] > df['price'].rolling(window=252, min_periods=50).max() * 0.8).astype('float64')
    
    return df

def calculate_hurst_exponent(ts):
    """Calculate Hurst exponent for regime detection"""
    if len(ts) < 20:
        return 0.5
    
    try:
        lags = range(2, min(20, len(ts)//4))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Linear regression on log-log plot
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return 0.5

def sophisticated_data_cleaning(df):
    """Advanced data cleaning and quality checks"""
    print("🧹 SOPHISTICATED DATA CLEANING...")
    
    initial_shape = df.shape
    
    # 1. Remove stocks with insufficient data
    stock_counts = df.groupby('symbol').size()
    valid_stocks = stock_counts[stock_counts >= 100].index  # At least 100 observations
    df = df[df['symbol'].isin(valid_stocks)].copy()
    print(f"  ✅ Filtered stocks: {len(valid_stocks)} stocks with sufficient data")
    
    # 2. Remove obvious data errors
    # Price consistency checks
    df = df[df['price'] > 0.01].copy()  # Remove penny stocks and errors
    df = df[df['price'] < 10000].copy()  # Remove obvious price errors
    
    # Return consistency checks
    df = df[df['return'].abs() < 1.0].copy()  # Remove impossible returns (>100% daily)
    
    # Reset index after filtering to avoid index compatibility issues
    df = df.reset_index(drop=True)
    
    # 3. Detect and handle jumps/gaps
    price_changes = df.groupby('symbol')['price'].pct_change().abs()
    df['price_gap'] = (price_changes > 0.5).astype('float64')
    gap_count = df['price_gap'].sum()
    if gap_count > 0:
        print(f"  ⚠️ Detected {gap_count} potential price gaps/splits")
        # Flag them for future reference
        df['has_gap'] = df['price_gap']
    else:
        df['has_gap'] = 0.0
    
    # 4. Noise reduction using median filters
    print("  🔧 Applying noise reduction...")
    
    # Initialize smoothed columns
    df['price_smoothed'] = df['price'].copy()
    df['return_smoothed'] = df['return'].copy()
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df.loc[mask].copy()
        
        if len(symbol_data) > 5:  # Need minimum data for median filter
            # Apply median filter to smooth extreme noise
            df.loc[mask, 'price_smoothed'] = symbol_data['price'].rolling(window=3, center=True).median().fillna(symbol_data['price'])
            df.loc[mask, 'return_smoothed'] = symbol_data['return'].rolling(window=3, center=True).median().fillna(symbol_data['return'])
    
    # Fill smoothed values where original data is extreme
    extreme_returns = df['return'].abs() > 3 * df['return'].std()
    extreme_count = extreme_returns.sum()
    if extreme_count > 0:
        df.loc[extreme_returns, 'return'] = df.loc[extreme_returns, 'return_smoothed']
        print(f"  ✅ Smoothed {extreme_count} extreme returns")
    
    print(f"  ✅ Data cleaning complete: {initial_shape[0]:,} → {df.shape[0]:,} records")
    
    return df

def advanced_outlier_detection(df, feature_cols):
    """Multi-method outlier detection and handling"""
    print("🔍 ADVANCED OUTLIER DETECTION...")
    
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    
    outlier_methods = {}
    
    # Sample for efficiency (define upfront)
    sample_size = min(50000, len(df))
    sample_idx = df.sample(n=sample_size, random_state=42).index
    
    # 1. Statistical outliers (IQR method)
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR  # Slightly more aggressive
            upper_bound = Q3 + 2.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_methods[f'{col}_iqr'] = outliers
    
    # 2. Isolation Forest for multivariate outliers
    if len(feature_cols) > 5:
        # Select numeric features only
        numeric_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']][:20]  # Limit to 20 features
        
        if len(numeric_cols) > 3:
            try:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                sample_data = df.loc[sample_idx, numeric_cols].fillna(0)
                
                outlier_scores = iso_forest.fit_predict(sample_data)
                outliers_iso = pd.Series(outlier_scores == -1, index=sample_idx)
                outlier_methods['isolation_forest'] = outliers_iso.reindex(df.index, fill_value=False)
            except Exception as e:
                print(f"    ⚠️ Isolation Forest failed: {e}")
    
    # 3. Robust covariance outliers
    if len(feature_cols) >= 3:
        try:
            # Use the same numeric columns as isolation forest
            numeric_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']][:10]  # Limit features
            
            if len(numeric_cols) >= 3:
                robust_cov = EllipticEnvelope(contamination=0.05, random_state=42)
                sample_data = df.loc[sample_idx, numeric_cols].fillna(0)
                
                outlier_scores = robust_cov.fit_predict(sample_data)
                outliers_cov = pd.Series(outlier_scores == -1, index=sample_idx)
                outlier_methods['robust_covariance'] = outliers_cov.reindex(df.index, fill_value=False)
        except Exception as e:
            print(f"    ⚠️ Robust covariance failed: {e}")
    
    # Combine outlier detection methods
    total_outliers = 0
    for method, outliers in outlier_methods.items():
        if outliers.sum() > 0:
            total_outliers += outliers.sum()
            print(f"  📊 {method}: {outliers.sum():,} outliers detected")
    
    print(f"  ✅ Total outlier instances: {total_outliers:,}")
    
    return outlier_methods

def improved_preprocess_pipeline(quick_mode=False):
    """Complete enhanced preprocessing pipeline for stock data"""
    print("="*80)
    if quick_mode:
        print("🚀 ENHANCED STOCK DATA PREPROCESSING PIPELINE (QUICK MODE)")
    else:
        print("🚀 ENHANCED STOCK DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    master_df, profile_df, predictions_df = load_stock_data()
    if master_df is None:
        print("❌ Failed to load data")
        return None, None, None, None
    
    # Step 1.5: Sophisticated data cleaning
    if not quick_mode:
        master_df = sophisticated_data_cleaning(master_df)
    else:
        print("  ⚡ Skipping advanced data cleaning in quick mode")
    
    # Step 2: Create advanced technical features
    enhanced_df = create_advanced_technical_features(master_df, quick_mode=quick_mode)
    
    # Step 3: Create target variables
    target_df = create_target_variables(enhanced_df)
    
    # Step 4: Create sector features
    final_df = create_sector_features(target_df)
    
    # Step 5: Advanced preprocessing for modeling (default target: next day return)
    X, y, feature_cols = preprocess_for_modeling(final_df, target_column='target_return_1d', 
                                                scale_features=True, select_features=True)
    
    print("="*80)
    if quick_mode:
        print("✅ ENHANCED PREPROCESSING PIPELINE COMPLETE (QUICK MODE)")
    else:
        print("✅ ENHANCED PREPROCESSING PIPELINE COMPLETE")
    print("="*80)
    print(f"📊 Final dataset shape: {X.shape}")
    print(f"🎯 Target variable: next-day returns")
    print(f"📈 Data range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"🏢 Sectors: {final_df['sector'].nunique()} unique sectors")
    print(f"🏭 Industries: {final_df['industry'].nunique()} unique industries")
    print(f"📊 Stocks: {final_df['symbol'].nunique()} unique stocks")
    print(f"🎯 Features selected: {len(feature_cols)} (optimized for ML performance)")
    
    return X, y, feature_cols, final_df

def load_data(quick_mode=False):
    """Wrapper function for compatibility with existing code"""
    return improved_preprocess_pipeline(quick_mode=quick_mode)

if __name__ == "__main__":
    # Test the preprocessing pipeline
    X, y, feature_cols, df = improved_preprocess_pipeline()
    if X is not None:
        print("\n🎉 Enhanced preprocessing test successful!")
        print(f"Features sample: {feature_cols[:10]}")
        print(f"Target statistics: Mean={y.mean():.4f}, Std={y.std():.4f}")
        print(f"Data quality: {(~X.isnull()).all().sum()}/{len(feature_cols)} features have no missing values") 
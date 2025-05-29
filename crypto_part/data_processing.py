import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load DeFi yield data from JSON file
    """
    try:
        # Try to load directly as DataFrame if it's a CSV
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        # Load as JSON if it's a JSON file
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = pd.read_json(file_path)
                # Check if the data has a nested structure
                if 'data' in data.columns:
                    df = pd.DataFrame(data['data'].tolist())
                else:
                    df = data
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the DeFi yield data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert binary fields to int
    binary_features = ['stablecoin', 'outlier']
    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Handle missing values in numeric columns
    numeric_cols = ['tvlUsd', 'apyBase', 'apyReward', 'apy', 'apyPct1D', 
                   'apyPct7D', 'apyPct30D', 'mu', 'sigma', 'count', 'apyMean30d']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Extract nested fields if needed
    if 'predictions' in df.columns and isinstance(df['predictions'].iloc[0], dict):
        # Extract nested prediction fields
        prediction_df = pd.json_normalize(df['predictions'])
        
        # Rename columns to avoid confusion
        prediction_df.columns = ['prediction_' + col for col in prediction_df.columns]
        
        # Drop the original predictions column and add the extracted columns
        df = df.drop('predictions', axis=1).reset_index(drop=True)
        df = pd.concat([df, prediction_df], axis=1)
    
    # Create engineered features
    df = engineer_features(df)
    
    return df

def engineer_features(df):
    """
    Create engineered features for improved modeling
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Add small constants to avoid division by zero
    epsilon = 0.001
    
    # Create yield efficiency (yield per TVL dollar)
    if 'apyBase' in df.columns and 'tvlUsd' in df.columns:
        df['yield_efficiency'] = df['apyBase'] / (df['tvlUsd'] + epsilon)
    
    # Create yield per risk
    if 'apy' in df.columns and 'sigma' in df.columns:
        df['yield_per_risk'] = df['apy'] / (df['sigma'] + epsilon)
    
    # Create relative APY to mean
    if 'apy' in df.columns and 'apyMean30d' in df.columns:
        df['relative_apy_to_mean'] = df['apy'] / (df['apyMean30d'] + epsilon)
    
    # Create volatility ratio
    if 'sigma' in df.columns and 'mu' in df.columns:
        df['volatility_ratio'] = df['sigma'] / (df['mu'] + epsilon)
    
    # Create has_rewards binary feature
    if 'apyReward' in df.columns:
        df['has_rewards'] = (df['apyReward'] > 0).astype(int)
    
    # Create recent trend indicator
    if 'apyPct7D' in df.columns:
        df['recent_trend'] = np.sign(df['apyPct7D'])
    
    # Create APY stability indicator
    if 'apyPct30D' in df.columns:
        df['apy_stability'] = (abs(df['apyPct30D']) < 1).astype(int)
    
    return df

def create_filtered_dataset(df, min_apy=3, max_apy=150, min_tvl=None):
    """
    Create a filtered dataset based on APY and TVL criteria
    """
    filtered_df = df.copy()
    
    # Filter by APY range
    filtered_df = filtered_df[(filtered_df['apy'] >= min_apy) & 
                              (filtered_df['apy'] <= max_apy)]
    
    # Filter by minimum TVL if specified
    if min_tvl is not None:
        filtered_df = filtered_df[filtered_df['tvlUsd'] >= min_tvl]
    
    print(f"Filtered dataset created with {filtered_df.shape[0]} rows")
    return filtered_df

def prepare_model_data(df, target='apy'):
    """
    Prepare data for modeling by defining features and target
    """
    # Define feature groups
    numeric_features = ['tvlUsd', 'apyBase', 'apyPct1D', 'apyPct7D', 'apyPct30D', 
                      'mu', 'sigma', 'count', 'apyMean30d', 'percentage_of_total_tvl',
                      'yield_efficiency', 'yield_per_risk', 'relative_apy_to_mean', 
                      'volatility_ratio', 'has_rewards', 'recent_trend', 'apy_stability']
    
    categorical_features = ['chain', 'project', 'exposure', 'ilRisk']
    
    binary_features = ['stablecoin', 'outlier']
    
    # Filter to only include features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    
    # Combine all features
    features = numeric_features + categorical_features + binary_features
    
    # Create X and y
    X = df[features]
    y = df[target]
    
    return X, y, numeric_features, categorical_features, binary_features

def prepare_model_data_no_leakage(df, target='apy'):
    """
    Prepare data for modeling by defining features and target, excluding potential data leakage
    """
    # Define feature groups but exclude potentially leaking features
    numeric_features = ['tvlUsd', 'mu', 'sigma', 'count', 'percentage_of_total_tvl',
                      'volatility_ratio', 'has_rewards', 'apy_stability']
    
    # Remove direct components of APY: apyBase, apyReward
    # Remove percentage changes of APY: apyPct1D, apyPct7D, apyPct30D
    # Remove derived features that use APY: yield_efficiency, relative_apy_to_mean, recent_trend, yield_per_risk
    
    categorical_features = ['chain', 'project', 'exposure', 'ilRisk']
    
    binary_features = ['stablecoin', 'outlier']
    
    # Filter to only include features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    
    # Combine all features
    features = numeric_features + categorical_features + binary_features
    
    # Create X and y
    X = df[features]
    y = df[target]
    
    return X, y, numeric_features, categorical_features, binary_features

def prepare_model_data_custom(df, target='apy'):
    """
    Prepare data for modeling using custom feature selection requested by user
    """
    # Apply custom feature engineering first
    df = engineer_custom_features(df)
    
    # Core requested features
    numeric_features = [
        'tvlUsd',           # tvlusd (exact match)
        'volumeUsd1d',      # volumeusd1d (exact match) 
        'percentage_of_total_tvl',  # percentage of total tvl
        'sigma',            # sigma (exact match)
        'mu'                # mu (exact match)
    ]
    
    categorical_features = [
        'project',          # project (exact match)
        'symbol',           # symbol (exact match)
        'exposure',         # exposure (exact match)
        'ilRisk',           # ilrisk (case match)
        'underlyingTokens', # underlyingtokens (case match)
        'rewardTokens',     # rewardtokens (case match)  
        'pool'              # pool (exact match)
    ]
    
    binary_features = [
        'stablecoin'        # stablecoin (exact match)
    ]
    
    # Additional useful metrics for better model performance
    additional_numeric = [
        'apyBase',          # Base APY component
        'apyReward',        # Reward APY component
        'count',            # Number of observations
        'volumeUsd7d',      # 7-day volume if available
        # New engineered features
        'log_tvl',          # Log-transformed TVL
        'log_volume_1d',    # Log-transformed volume
        'volume_tvl_ratio', # Volume to TVL ratio
        'risk_adjusted_return', # Sharpe-like ratio
        'tvl_chain_rank',   # TVL rank within chain
        'tvl_project_rank', # TVL rank within project
        'n_underlying_tokens', # Number of underlying tokens
        'n_reward_tokens',  # Number of reward tokens
        'project_pool_count', # Number of pools in project
        'chain_avg_tvl',    # Average TVL in chain
        'chain_median_tvl', # Median TVL in chain
        'chain_protocol_count', # Number of protocols in chain
        'log_tvl_percentage', # Log percentage of total TVL
        'stability_score',  # Stablecoin stability score
        'liquidity_risk_ratio' # Liquidity-adjusted risk
    ]
    
    additional_categorical = [
        'chain',            # Blockchain network
        'category',         # Protocol category if available
        'risk_level',       # Risk level categorization
        'market_dominance'  # Market dominance category
    ]
    
    additional_binary = [
        'outlier',          # Outlier detection flag
        'high_il_risk',     # High IL risk flag
        'is_single_token',  # Single token flag
        'has_multiple_rewards', # Multiple rewards flag
        'is_major_chain',   # Major chain flag
        'is_large_project', # Large project flag
        'has_popular_token', # Popular token flag
        'is_stablecoin_pair' # Stablecoin pair flag
    ]
    
    # Combine core and additional features
    all_numeric = numeric_features + additional_numeric
    all_categorical = categorical_features + additional_categorical  
    all_binary = binary_features + additional_binary
    
    # Filter to only include features that exist in the dataframe
    available_numeric = [f for f in all_numeric if f in df.columns]
    available_categorical = [f for f in all_categorical if f in df.columns]
    available_binary = [f for f in all_binary if f in df.columns]
    
    # Print available features for verification
    print("Available numeric features:", len(available_numeric))
    print("Available categorical features:", len(available_categorical))
    print("Available binary features:", len(available_binary))
    
    # Combine all available features
    all_features = available_numeric + available_categorical + available_binary
    
    # Remove target from features if it exists
    if target in all_features:
        all_features.remove(target)
        if target in available_numeric:
            available_numeric.remove(target)
    
    # Create X and y
    X = df[all_features].copy()
    y = df[target].copy()
    
    # Handle missing values for the specific features
    print("Processing missing values...")
    
    # Fill numeric features with median
    for col in available_numeric:
        if col != target:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
            except Exception as e:
                print(f"Warning: Error processing numeric column {col}: {e}")
    
    # Fill categorical features with 'unknown'
    for col in available_categorical:
        try:
            # Convert categorical columns to string first to avoid category restrictions
            if pd.api.types.is_categorical_dtype(X[col]):
                X[col] = X[col].astype(str)
            X[col] = X[col].fillna('unknown').astype(str)
        except Exception as e:
            print(f"Warning: Error processing categorical column {col}: {e}")
    
    # Fill binary features with 0
    for col in available_binary:
        try:
            X[col] = X[col].fillna(0).astype(int)
        except Exception as e:
            print(f"Warning: Error processing binary column {col}: {e}")
    
    print(f"Prepared dataset with {X.shape[0]} rows and {X.shape[1]} features")
    print(f"Target variable '{target}' has {y.isna().sum()} missing values")
    
    return X, y, available_numeric, available_categorical, available_binary

def engineer_custom_features(df):
    """
    Create additional engineered features specific to the custom feature selection
    """
    df = df.copy()
    epsilon = 0.001  # Small constant to avoid division by zero
    
    # Calculate percentage of total TVL (core requested feature)
    if 'tvlUsd' in df.columns:
        total_tvl = df['tvlUsd'].sum()
        if total_tvl > 0:
            df['percentage_of_total_tvl'] = (df['tvlUsd'] / total_tvl) * 100
        else:
            df['percentage_of_total_tvl'] = 0.0
    
    # TVL-based features
    if 'tvlUsd' in df.columns:
        # Log transformation for TVL (often has exponential distribution)
        df['log_tvl'] = np.log1p(df['tvlUsd'])
        
        # TVL percentile rank within each chain
        if 'chain' in df.columns:
            df['tvl_chain_rank'] = df.groupby('chain')['tvlUsd'].rank(pct=True)
        
        # TVL percentile rank within each project
        if 'project' in df.columns:
            df['tvl_project_rank'] = df.groupby('project')['tvlUsd'].rank(pct=True)
    
    # Volume-based features
    if 'volumeUsd1d' in df.columns and 'tvlUsd' in df.columns:
        # Volume to TVL ratio (turnover ratio)
        df['volume_tvl_ratio'] = df['volumeUsd1d'] / (df['tvlUsd'] + epsilon)
        
        # Log volume
        df['log_volume_1d'] = np.log1p(df['volumeUsd1d'])
    
    # Risk-adjusted features
    if 'sigma' in df.columns and 'mu' in df.columns:
        # Sharpe-like ratio
        df['risk_adjusted_return'] = df['mu'] / (df['sigma'] + epsilon)
        
        # Risk level categorization
        df['risk_level'] = pd.cut(df['sigma'], 
                                 bins=[0, 0.1, 0.3, 0.5, float('inf')], 
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    
    # IL Risk encoding
    if 'ilRisk' in df.columns:
        # Create binary high IL risk flag
        df['high_il_risk'] = (df['ilRisk'] == 'no').astype(int)  # Assuming 'no' means high risk
        
    # Token diversity features
    if 'underlyingTokens' in df.columns:
        # Count number of underlying tokens (if it's a list/string)
        def count_tokens(tokens):
            try:
                # Handle None/NaN values
                if tokens is None:
                    return 0
                if isinstance(tokens, float) and np.isnan(tokens):
                    return 0
                if pd.isna(tokens):
                    return 0
                
                # Handle different data types
                if isinstance(tokens, list):
                    return len(tokens)
                elif isinstance(tokens, np.ndarray):
                    return len(tokens)
                elif isinstance(tokens, str):
                    # Count commas + 1, or count spaces, depending on format
                    if ',' in tokens:
                        return len([t.strip() for t in tokens.split(',') if t.strip()])
                    elif ' ' in tokens:
                        return len([t.strip() for t in tokens.split() if t.strip()])
                    else:
                        return 1 if tokens.strip() else 0
                else:
                    # Try to convert to string and count
                    str_tokens = str(tokens)
                    if str_tokens.lower() in ['nan', 'none', '']:
                        return 0
                    return 1
            except Exception:
                return 0
        
        df['n_underlying_tokens'] = df['underlyingTokens'].apply(count_tokens)
        df['is_single_token'] = (df['n_underlying_tokens'] == 1).astype(int)
    
    if 'rewardTokens' in df.columns:
        # Count reward tokens
        def count_rewards(tokens):
            try:
                # Handle None/NaN values
                if tokens is None:
                    return 0
                if isinstance(tokens, float) and np.isnan(tokens):
                    return 0
                if pd.isna(tokens):
                    return 0
                
                # Handle different data types
                if isinstance(tokens, list):
                    return len(tokens)
                elif isinstance(tokens, np.ndarray):
                    return len(tokens)
                elif isinstance(tokens, str):
                    str_tokens = tokens.strip().lower()
                    if str_tokens in ['none', 'nan', '']:
                        return 0
                    if ',' in tokens:
                        return len([t.strip() for t in tokens.split(',') if t.strip()])
                    elif ' ' in tokens:
                        return len([t.strip() for t in tokens.split() if t.strip()])
                    else:
                        return 1 if tokens.strip() else 0
                else:
                    # Try to convert to string and count
                    str_tokens = str(tokens)
                    if str_tokens.lower() in ['nan', 'none', '']:
                        return 0
                    return 1
            except Exception:
                return 0
        
        df['n_reward_tokens'] = df['rewardTokens'].apply(count_rewards)
        df['has_multiple_rewards'] = (df['n_reward_tokens'] > 1).astype(int)
    
    # Chain-based features
    if 'chain' in df.columns:
        # Major chains flag
        major_chains = ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche']
        df['is_major_chain'] = df['chain'].str.lower().isin(major_chains).astype(int)
        
        # Chain TVL aggregations
        chain_stats = df.groupby('chain')['tvlUsd'].agg(['mean', 'median', 'count']).reset_index()
        chain_stats.columns = ['chain', 'chain_avg_tvl', 'chain_median_tvl', 'chain_protocol_count']
        df = df.merge(chain_stats, on='chain', how='left')
    
    # Project-based features
    if 'project' in df.columns:
        # Project size (number of pools)
        project_counts = df['project'].value_counts().to_dict()
        df['project_pool_count'] = df['project'].map(project_counts)
        df['is_large_project'] = (df['project_pool_count'] > df['project_pool_count'].median()).astype(int)
    
    # Symbol-based features
    if 'symbol' in df.columns:
        # Popular token flags
        popular_tokens = ['USDC', 'USDT', 'DAI', 'WETH', 'WBTC']
        df['has_popular_token'] = df['symbol'].str.contains('|'.join(popular_tokens), case=False, na=False).astype(int)
        
        # Stablecoin pairs
        stablecoin_symbols = ['USDC', 'USDT', 'DAI', 'BUSD', 'FRAX']
        df['is_stablecoin_pair'] = df['symbol'].str.contains('|'.join(stablecoin_symbols), case=False, na=False).astype(int)
    
    # Percentage of total TVL features
    if 'percentage_of_total_tvl' in df.columns:
        # Market dominance categories
        df['market_dominance'] = pd.cut(df['percentage_of_total_tvl'],
                                       bins=[0, 0.01, 0.1, 1, float('inf')],
                                       labels=['Niche', 'Small', 'Medium', 'Large'])
        
        # Log transformation
        df['log_tvl_percentage'] = np.log1p(df['percentage_of_total_tvl'])
    
    # Interaction features
    if 'stablecoin' in df.columns and 'sigma' in df.columns:
        # Stability score (higher for stablecoins with low volatility)
        df['stability_score'] = df['stablecoin'] * (1 / (df['sigma'] + epsilon))
    
    if 'tvlUsd' in df.columns and 'volumeUsd1d' in df.columns and 'sigma' in df.columns:
        # Liquidity-adjusted risk
        liquidity_measure = df['tvlUsd'] + df['volumeUsd1d']
        df['liquidity_risk_ratio'] = liquidity_measure / (df['sigma'] + epsilon)
    
    print(f"Added engineered features to dataset")
    return df 
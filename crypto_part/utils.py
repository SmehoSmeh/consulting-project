import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

def setup_environment():
    """
    Setup the environment for analysis - create necessary directories
    """
    # Create directories if they don't exist
    dirs = ['data', 'models', 'plots', 'results']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
    
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 100)
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

def get_timestamp():
    """
    Get a formatted timestamp for file naming
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_tvl_distribution(df):
    """
    Calculate TVL distribution statistics
    """
    # Calculate total TVL
    total_tvl = df['tvlUsd'].sum()
    
    # Calculate percentage of total TVL for each pool
    df = df.copy()
    df['percentage_of_total_tvl'] = (df['tvlUsd'] / total_tvl) * 100
    
    # Create summary dataframe
    tvl_summary = df[['project', 'symbol', 'tvlUsd', 'percentage_of_total_tvl']]
    
    # Add chain information if available
    if 'chain' in df.columns:
        tvl_summary['chain'] = df['chain']
    
    return tvl_summary, total_tvl

def calculate_apy_statistics(df):
    """
    Calculate APY statistics by project and chain
    """
    # Project level statistics
    project_stats = df.groupby('project').agg({
        'apy': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'tvlUsd': ['sum', 'mean']
    }).sort_values(('tvlUsd', 'sum'), ascending=False)
    
    # Chain level statistics if chain information available
    chain_stats = None
    if 'chain' in df.columns:
        chain_stats = df.groupby('chain').agg({
            'apy': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'tvlUsd': ['sum', 'mean']
        }).sort_values(('tvlUsd', 'sum'), ascending=False)
    
    return project_stats, chain_stats

def save_results(df, filename, directory='results'):
    """
    Save DataFrame results to CSV
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path
    file_path = os.path.join(directory, filename)
    
    # Save to CSV
    df.to_csv(file_path)
    print(f"Results saved to {file_path}")
    
    return file_path

def get_project_recommendations(comparison_df, top_n=10, min_apy=3, max_apy=150):
    """
    Get project recommendations based on model predictions
    """
    # Filter by APY range
    filtered_df = comparison_df[
        (comparison_df['XGBoost_Predicted_APY'] >= min_apy) & 
        (comparison_df['XGBoost_Predicted_APY'] <= max_apy)
    ].copy()
    
    # Calculate prediction accuracy
    filtered_df['Accuracy'] = 1 - (
        np.abs(filtered_df['Prediction_Error']) / filtered_df['Actual_APY']
    )
    
    # Group by project
    if 'project' in filtered_df.columns:
        project_metrics = filtered_df.groupby('project').agg({
            'XGBoost_Predicted_APY': ['mean', 'std'],
            'Accuracy': 'mean',
            'Actual_APY': ['mean', 'count']
        }).sort_values(('XGBoost_Predicted_APY', 'mean'), ascending=False)
        
        # Get top N projects by predicted APY
        top_projects = project_metrics.head(top_n)
        
        return top_projects
    else:
        return None

def prepare_feature_names(model, numeric_features, categorical_features, binary_features):
    """
    Prepare feature names for feature importance visualization
    """
    # Start with numeric and binary features
    feature_names = numeric_features + binary_features
    
    # Add categorical features with one-hot encoding
    for i, col in enumerate(categorical_features):
        # Get the categories from the OneHotEncoder
        cats = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_[i]
        feature_names.extend([f"{col}_{cat}" for cat in cats])
    
    return feature_names

def generate_summary_report(metrics, top_features, top_projects, direction_metrics=None):
    """
    Generate a text summary report of the analysis
    """
    report = []
    
    # Add header
    report.append("# DeFi Yield Prediction Analysis Summary")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Add model performance metrics
    report.append("## Model Performance Metrics")
    report.append(f"- R² Score: {metrics.get('r2', 0):.4f}")
    report.append(f"- Root Mean Square Error (RMSE): {metrics.get('rmse', 0):.4f}")
    report.append(f"- Mean Absolute Error (MAE): {metrics.get('mae', 0):.4f}")
    report.append(f"- Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 0):.4f}%")
    report.append("")
    
    # Add top features
    report.append("## Top 10 Important Features")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'].head(10), 
                                                top_features['Importance'].head(10))):
        report.append(f"{i+1}. {feature}: {importance:.4f}")
    report.append("")
    
    # Add direction prediction metrics if available
    if direction_metrics:
        report.append("## Direction Prediction Performance")
        report.append(f"- Directional Accuracy: {direction_metrics.get('accuracy', 0)*100:.1f}%")
        report.append("")
    
    # Add top project recommendations
    report.append("## Top Project Recommendations")
    if top_projects is not None:
        for i, (project, row) in enumerate(top_projects.iterrows()):
            report.append(f"{i+1}. {project}")
            report.append(f"   - Predicted APY: {row[('XGBoost_Predicted_APY', 'mean')]:.2f}%")
            report.append(f"   - Actual APY: {row[('Actual_APY', 'mean')]:.2f}%")
            report.append(f"   - Prediction Accuracy: {row[('Accuracy', 'mean')]*100:.1f}%")
            report.append(f"   - Sample Count: {row[('Actual_APY', 'count')]}")
            report.append("")
    
    # Join all lines
    return "\n".join(report)

def save_report(report, filename='analysis_summary.md', directory='results'):
    """
    Save the generated report to a file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path
    file_path = os.path.join(directory, filename)
    
    # Save to file
    with open(file_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {file_path}")
    
    return file_path 
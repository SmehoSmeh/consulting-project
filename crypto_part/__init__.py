"""
DeFi Yield Prediction Analysis Package
"""

__version__ = "0.1.0"
__author__ = "DeFi Analysis Team"

# Import key modules
from . import data_processing
from . import model_training
from . import model_evaluation
from . import visualization
from . import utils

# Import key functions for easier access
from .data_processing import load_data, preprocess_data, create_filtered_dataset
from .model_training import train_xgboost_model, predict_with_intervals
from .model_evaluation import evaluate_model, compare_with_original_predictions
from .visualization import create_comparison_dashboard, plot_feature_importance
from .utils import setup_environment, calculate_tvl_distribution

# Define what gets imported with "from crypto_part import *"
__all__ = [
    'data_processing',
    'model_training',
    'model_evaluation',
    'visualization',
    'utils',
    'load_data',
    'preprocess_data',
    'create_filtered_dataset',
    'train_xgboost_model',
    'predict_with_intervals',
    'evaluate_model',
    'compare_with_original_predictions',
    'create_comparison_dashboard',
    'plot_feature_importance',
    'setup_environment',
    'calculate_tvl_distribution'
] 
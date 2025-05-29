import os
import sys
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd


# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_processing import load_data, preprocess_data, create_filtered_dataset, prepare_model_data, prepare_model_data_no_leakage
from model_training import train_xgboost_model, save_model, predict_with_intervals
from model_evaluation import (
    evaluate_model, get_feature_importance, compare_with_original_predictions,
    create_filtered_comparison, compare_error_metrics, calculate_direction_metrics,
    summarize_comparison
)
from visualization import (
    plot_feature_importance, plot_predictions_vs_actual, plot_error_distribution,
    plot_confusion_matrix, create_comparison_dashboard, plot_feature_importance_plotly,
    plot_correlation_heatmap_plotly, create_apy_plotly_dashboard
)
from utils import (
    setup_environment, get_timestamp, calculate_tvl_distribution,
    calculate_apy_statistics, save_results, get_project_recommendations,
    prepare_feature_names, generate_summary_report, save_report
)

# Default data file path - use relative path from project root
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'source_data', 'defilama_data.json')

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='DeFi Yield Prediction Analysis')
    
    parser.add_argument('--data_file', type=str, default=DEFAULT_DATA_FILE,
                        help=f'Path to the data file (CSV or JSON) (default: {DEFAULT_DATA_FILE})')
    
    parser.add_argument('--target', type=str, default='apy',
                        help='Target variable to predict (default: apy)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size as a fraction (default: 0.2)')
    
    parser.add_argument('--min_apy', type=float, default=3.0,
                        help='Minimum APY for filtered dataset (default: 3.0)')
    
    parser.add_argument('--max_apy', type=float, default=150.0,
                        help='Maximum APY for filtered dataset (default: 150.0)')
    
    parser.add_argument('--no_grid_search', action='store_true',
                        help='Skip hyperparameter tuning with grid search')
    
    parser.add_argument('--no_leakage', action='store_true',
                        help='Use features with no potential data leakage (removes apyBase, apyReward, apyPct1D, etc.)')
    
    parser.add_argument('--use_plotly', action='store_true',
                        help='Use Plotly for interactive visualizations instead of matplotlib')
    
    parser.add_argument('--save_model_path', type=str, default=None,
                        help='Path to save the trained model')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save output files (default: results)')
    
    return parser.parse_args()

def main():
    """
    Main function to run the DeFi yield prediction analysis
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Generate timestamp for file naming
    timestamp = get_timestamp()
    
    print(f"Starting DeFi Yield Prediction Analysis at {timestamp}")
    print(f"Loading data from {args.data_file}")
    
    # Load and preprocess data
    df = load_data(args.data_file)
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    df = preprocess_data(df)
    
    # Create filtered dataset
    filtered_df = create_filtered_dataset(df, min_apy=args.min_apy, max_apy=args.max_apy)
    
    # Calculate TVL distribution
    tvl_summary, total_tvl = calculate_tvl_distribution(df)
    print(f"Total TVL: ${total_tvl:,.2f}")
    
    # Calculate APY statistics
    project_stats, chain_stats = calculate_apy_statistics(df)
    print(f"Number of projects: {len(project_stats)}")
    if chain_stats is not None:
        print(f"Number of chains: {len(chain_stats)}")
    
    # Save TVL and APY statistics
    save_results(tvl_summary, f"{timestamp}_tvl_distribution.csv", args.output_dir)
    save_results(project_stats, f"{timestamp}_project_stats.csv", args.output_dir)
    if chain_stats is not None:
        save_results(chain_stats, f"{timestamp}_chain_stats.csv", args.output_dir)
    
    # Create correlation heatmap to visualize potential data leakage
    if args.use_plotly:
        plot_correlation_heatmap_plotly(
            df, 
            target=args.target,
            save_path=os.path.join(args.output_dir, f"{timestamp}_correlation_heatmap.html")
        )
        
        # Create APY dashboard
        create_apy_plotly_dashboard(
            df, 
            filtered_df, 
            save_prefix=timestamp
        )
    
    # Prepare data for modeling
    if args.no_leakage:
        print("Using features with no potential data leakage")
        X, y, numeric_features, categorical_features, binary_features = prepare_model_data_no_leakage(df, target=args.target)
    else:
        X, y, numeric_features, categorical_features, binary_features = prepare_model_data(df, target=args.target)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    print("Training model...")
    model, best_params = train_xgboost_model(
        X_train, y_train, numeric_features, categorical_features, 
        do_grid_search=not args.no_grid_search
    )
    
    # Save model if requested
    if args.save_model_path:
        save_model(model, model_dir=args.output_dir, model_name=args.save_model_path)
    else:
        save_model(model, model_dir=args.output_dir, model_name=f"{timestamp}_defi_model.pkl")
    
    # Evaluate model
    print("Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    feature_names = prepare_feature_names(model, numeric_features, categorical_features, binary_features)
    importance_df = get_feature_importance(model, feature_names)
    
    # Plot feature importance
    if args.use_plotly:
        plot_feature_importance_plotly(
            importance_df, top_n=20, 
            save_path=os.path.join(args.output_dir, f"{timestamp}_feature_importance.html")
        )
    else:
        plot_feature_importance(
            importance_df, top_n=20, 
            save_path=os.path.join(args.output_dir, f"{timestamp}_feature_importance.png")
        )
    
    # Create comparison with original predictions
    comparison_df = compare_with_original_predictions(y_test, y_pred, df, X_test.index)
    
    # Create filtered comparison if there's a filtered dataset
    filtered_comparison = create_filtered_comparison(comparison_df, filtered_df)
    
    # Compare error metrics
    compare_error_metrics(comparison_df, filtered_comparison)
    
    # Calculate direction metrics
    direction_metrics = calculate_direction_metrics(comparison_df)
    
    # Create visualization dashboard
    create_comparison_dashboard(
        comparison_df, filtered_comparison, save_prefix=timestamp
    )
    
    # Get project recommendations
    top_projects = get_project_recommendations(
        comparison_df, top_n=10, min_apy=args.min_apy, max_apy=args.max_apy
    )
    
    # Generate and save summary report
    report = generate_summary_report(metrics, importance_df, top_projects, direction_metrics)
    save_report(report, filename=f"{timestamp}_analysis_summary.md", directory=args.output_dir)
    
    # Print summary
    summarize_comparison(metrics, direction_metrics)
    
    # Add a note about data leakage if the no_leakage flag was not used
    if not args.no_leakage:
        print("\n⚠️ WARNING: This model may be affected by data leakage from features directly related to APY.")
        print("             For a more realistic evaluation, run with the --no_leakage flag.")
    
    print(f"Analysis completed. Results saved to {args.output_dir} directory.")

if __name__ == "__main__":
    main() 
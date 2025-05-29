# DeFi Yield Prediction Analysis

This project provides tools for analyzing and predicting yields in decentralized finance (DeFi) protocols using machine learning, specifically XGBoost regression models.

## Features

- Data loading and preprocessing for DeFi yield data
- Feature engineering for better prediction accuracy
- XGBoost model training with hyperparameter tuning
- Comprehensive model evaluation and comparison with original predictions
- Visualization of results and model performance
- Project recommendations based on predicted yields
- Detailed reporting and result summaries

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

You can install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

## Usage

### Basic Usage

The main script ties together all components of the analysis. The simplest way to run it is:

```bash
python main.py --data_file path/to/your/data.json
```

This will:
1. Load and preprocess the data
2. Train an XGBoost model with hyperparameter tuning
3. Evaluate the model
4. Create visualizations
5. Generate a comprehensive report

### Command Line Arguments

The main script accepts several command line arguments:

- `--data_file`: Path to the data file (CSV or JSON) **(required)**
- `--target`: Target variable to predict (default: "apy")
- `--test_size`: Test set size as a fraction (default: 0.2)
- `--min_apy`: Minimum APY for filtered dataset (default: 3.0)
- `--max_apy`: Maximum APY for filtered dataset (default: 150.0)
- `--no_grid_search`: Skip hyperparameter tuning with grid search
- `--save_model_path`: Custom path to save the trained model
- `--output_dir`: Directory to save output files (default: "results")

Example with custom parameters:

```bash
python main.py --data_file data/defilama_yields.json --min_apy 5 --max_apy 100 --output_dir custom_results --no_grid_search
```

## Module Overview

The project is organized into several modules:

1. **data_processing.py**: Functions for loading, preprocessing, and feature engineering
2. **model_training.py**: XGBoost model training and prediction functions
3. **model_evaluation.py**: Comprehensive evaluation metrics and comparisons
4. **visualization.py**: Plotting functions for data visualization
5. **utils.py**: Utility functions for file handling, reporting, etc.
6. **main.py**: Main script that ties everything together

## Example Workflow

1. **Data Preparation**:
   ```python
   from data_processing import load_data, preprocess_data
   
   # Load data
   df = load_data('data/defilama_yields.json')
   
   # Preprocess data
   processed_df = preprocess_data(df)
   ```

2. **Model Training**:
   ```python
   from data_processing import prepare_model_data
   from model_training import train_xgboost_model
   from sklearn.model_selection import train_test_split
   
   # Prepare features and target
   X, y, numeric_features, categorical_features, binary_features = prepare_model_data(processed_df)
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Train model
   model, best_params = train_xgboost_model(X_train, y_train, numeric_features, categorical_features)
   ```

3. **Evaluation and Visualization**:
   ```python
   from model_evaluation import evaluate_model, get_feature_importance
   from visualization import plot_feature_importance, plot_predictions_vs_actual
   from utils import prepare_feature_names
   
   # Evaluate model
   y_pred, metrics = evaluate_model(model, X_test, y_test)
   
   # Get feature importance
   feature_names = prepare_feature_names(model, numeric_features, categorical_features, binary_features)
   importance_df = get_feature_importance(model, feature_names)
   
   # Plot results
   plot_feature_importance(importance_df)
   plot_predictions_vs_actual(y_test, y_pred)
   ```

## Output Files

The analysis generates several output files in the specified output directory:

- Model file (PKL)
- Feature importance plot (PNG)
- Actual vs predicted values plot (PNG)
- Error distribution plot (PNG)
- Confusion matrix plot (PNG) if applicable
- Project performance plot (PNG)
- TVL distribution (CSV)
- Project statistics (CSV)
- Chain statistics (CSV) if applicable
- Comprehensive analysis summary (MD)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
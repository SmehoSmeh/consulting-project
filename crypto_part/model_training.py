import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import joblib
import os
from scipy import stats

def create_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline for mixed data types
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ], remainder='passthrough')
    
    return preprocessor

def create_xgboost_model(preprocessor, params=None):
    """
    Create an XGBoost model with preprocessing pipeline
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Create the pipeline with preprocessor and XGBoost
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBRegressor(**params))
    ])
    
    return model

def train_xgboost_comprehensive(X_train, y_train, numeric_features, categorical_features, 
                               search_type='grid', cv_folds=5, random_state=42):
    """
    Train XGBoost model with comprehensive hyperparameter tuning
    """
    print(f"Starting comprehensive hyperparameter tuning using {search_type} search...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Create base model
    model = create_xgboost_model(preprocessor)
    
    # Define comprehensive parameter grids
    if search_type == 'grid':
        # Grid search with focused parameter ranges
        param_grid = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'xgb__max_depth': [3, 5, 7, 9],
            'xgb__min_child_weight': [1, 3, 5],
            'xgb__subsample': [0.7, 0.8, 0.9, 1.0],
            'xgb__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'xgb__gamma': [0, 0.1, 0.2],
            'xgb__reg_alpha': [0, 0.1, 0.5],
            'xgb__reg_lambda': [0.1, 1, 1.5]
        }
        
        search = GridSearchCV(
            model, param_grid, cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
    else:  # randomized search
        # Randomized search with distributions
        param_dist = {
            'xgb__n_estimators': stats.randint(50, 500),
            'xgb__learning_rate': stats.uniform(0.01, 0.3),
            'xgb__max_depth': stats.randint(3, 12),
            'xgb__min_child_weight': stats.randint(1, 10),
            'xgb__subsample': stats.uniform(0.6, 0.4),
            'xgb__colsample_bytree': stats.uniform(0.6, 0.4),
            'xgb__gamma': stats.uniform(0, 0.5),
            'xgb__reg_alpha': stats.uniform(0, 1),
            'xgb__reg_lambda': stats.uniform(0.1, 2)
        }
        
        search = RandomizedSearchCV(
            model, param_dist, cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_iter=100, n_jobs=-1, verbose=1,
            random_state=random_state
        )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Print results
    print(f"Best CV score: {-search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    
    # Additional analysis
    print("\n--- Hyperparameter Analysis ---")
    results_df = pd.DataFrame(search.cv_results_)
    
    # Show top 5 parameter combinations
    top_results = results_df.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    print("Top 5 parameter combinations:")
    for idx, row in top_results.iterrows():
        print(f"Score: {-row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f}) - {row['params']}")
    
    return search.best_estimator_, search.best_params_, search

def train_xgboost_model(X_train, y_train, numeric_features, categorical_features, do_grid_search=True):
    """
    Train an XGBoost model with optional hyperparameter tuning
    """
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Create base model
    model = create_xgboost_model(preprocessor)
    
    # If grid search is requested, perform hyperparameter tuning
    if do_grid_search:
        print("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'xgb__n_estimators': [50, 100, 200],
            'xgb__learning_rate': [0.01, 0.1, 0.2],
            'xgb__max_depth': [3, 5, 7],
            'xgb__min_child_weight': [1, 3, 5],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.8, 1.0]
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        return best_model, grid_search.best_params_
    
    else:
        # Just fit the base model
        print("Training XGBoost model with default parameters...")
        model.fit(X_train, y_train)
        
        return model, None

def create_ensemble_model(X_train, y_train, numeric_features, categorical_features, n_models=3):
    """
    Create an ensemble of XGBoost models with different random seeds
    """
    print(f"Creating ensemble of {n_models} XGBoost models...")
    
    models = []
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")
        
        # Use different random seeds for diversity
        model, _ = train_xgboost_comprehensive(
            X_train, y_train, numeric_features, categorical_features,
            search_type='randomized', random_state=42+i
        )
        models.append(model)
    
    return models

def predict_ensemble(models, X):
    """
    Make predictions using ensemble of models
    """
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Calculate prediction std for uncertainty estimation
    pred_std = np.std(predictions, axis=0)
    
    return ensemble_pred, pred_std

def save_model(model, model_dir='models', model_name='defi_yield_model.pkl'):
    """
    Save the trained model to disk
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Full path to save the model
    model_path = os.path.join(model_dir, model_name)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load a trained model from disk
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_with_intervals(model, X, confidence=0.95):
    """
    Make predictions with confidence intervals
    
    This is a simple approach that assumes errors are normally distributed
    and uses the standard deviation of training errors to create intervals.
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # For real confidence intervals, we'd need quantile regression or bootstrapping
    # This is a simplified approach
    error_margin = 1.0  # This could be calibrated based on training errors
    
    # Create intervals (simplified approach)
    y_lower = y_pred - error_margin
    y_upper = y_pred + error_margin
    
    return y_pred, y_lower, y_upper 
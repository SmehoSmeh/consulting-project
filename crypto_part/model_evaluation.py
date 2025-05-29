import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print("\n--- Model Evaluation Metrics ---")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Return predictions and metrics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    return y_pred, metrics

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the XGBoost model
    """
    # Extract the XGBoost model from the pipeline
    xgb_model = model.named_steps['xgb']
    importances = xgb_model.feature_importances_
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def compare_with_original_predictions(y_test, y_pred, original_df, test_indices):
    """
    Compare XGBoost predictions with original predictedClass and predictedProbability
    """
    # Prepare dataframe for comparison
    comparison_df = pd.DataFrame({
        'Actual_APY': y_test,
        'XGBoost_Predicted_APY': y_pred,
        'Prediction_Error': y_pred - y_test
    })
    
    # Add original predictions if they exist
    prediction_cols = [col for col in original_df.columns if 'predict' in col.lower()]
    if prediction_cols:
        for col in prediction_cols:
            comparison_df[f'Original_{col}'] = original_df.loc[test_indices, col].values
    
    # Add XGBoost directional prediction
    comparison_df['XGBoost_Direction'] = np.where(
        np.abs(comparison_df['Prediction_Error']) < 0.5, 
        'Stable',
        np.where(comparison_df['Prediction_Error'] > 0, 'Up', 'Down')
    )
    
    # Convert original predictedClass to direction if it exists
    if 'Original_predictedClass' in comparison_df.columns:
        comparison_df['Original_Direction'] = comparison_df['Original_predictedClass'].apply(
            lambda x: 'Up' if 'Up' in str(x) else ('Down' if 'Down' in str(x) else 'Stable')
        )
        
        # Calculate agreement rate
        mask = ~comparison_df['Original_Direction'].isna()
        if mask.any():
            agreement = (comparison_df.loc[mask, 'XGBoost_Direction'] == 
                        comparison_df.loc[mask, 'Original_Direction'])
            agreement_rate = agreement.mean() * 100
            print(f"\nAgreement rate between XGBoost and original predictions: {agreement_rate:.1f}%")
    
    return comparison_df

def create_filtered_comparison(comparison_df, filtered_df):
    """
    Create comparison with a filtered dataset (e.g., APY 3-150%)
    """
    # Get indices that are in both datasets
    common_indices = set(comparison_df.index) & set(filtered_df.index)
    
    if not common_indices:
        print("No common indices found between comparison and filtered datasets")
        return None
    
    # Convert set to list for pandas indexing
    common_indices_list = list(common_indices)
    
    # Create filtered comparison dataframe
    filtered_comparison = comparison_df.loc[common_indices_list].copy()
    
    # Add any additional columns from filtered_df that might be useful
    for col in filtered_df.columns:
        if col not in filtered_comparison.columns:
            filtered_comparison[col] = filtered_df.loc[common_indices_list, col].values
    
    print(f"Created filtered comparison with {len(filtered_comparison)} samples")
    return filtered_comparison

def compare_error_metrics(comparison_df, filtered_comparison=None):
    """
    Compare error metrics between different datasets
    """
    # Overall metrics
    overall_mae = comparison_df['Prediction_Error'].abs().mean()
    overall_mape = (comparison_df['Prediction_Error'].abs() / comparison_df['Actual_APY']).mean() * 100
    
    print("\n--- Error Metrics ---")
    print(f"Overall Mean Absolute Error: {overall_mae:.4f}")
    print(f"Overall Mean Absolute Percentage Error: {overall_mape:.4f}%")
    
    # Filtered metrics if available
    if filtered_comparison is not None:
        filtered_mae = filtered_comparison['Prediction_Error'].abs().mean()
        filtered_mape = (filtered_comparison['Prediction_Error'].abs() / 
                        filtered_comparison['Actual_APY']).mean() * 100
        
        print(f"Filtered Mean Absolute Error: {filtered_mae:.4f}")
        print(f"Filtered Mean Absolute Percentage Error: {filtered_mape:.4f}%")
        
        # Compare performance
        if filtered_mae < overall_mae:
            print(f"Model performs better on filtered data by {(1 - filtered_mae/overall_mae)*100:.1f}% (MAE)")
        else:
            print(f"Model performs worse on filtered data by {(filtered_mae/overall_mae - 1)*100:.1f}% (MAE)")

def calculate_direction_metrics(comparison_df):
    """
    Calculate metrics for directional predictions
    """
    if 'Original_Direction' not in comparison_df.columns:
        print("Original direction predictions not available")
        return None
    
    # Remove rows with missing original direction
    valid_rows = comparison_df.dropna(subset=['Original_Direction'])
    
    if len(valid_rows) == 0:
        print("No valid original direction predictions found")
        return None
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        valid_rows['Original_Direction'], 
        valid_rows['XGBoost_Direction'],
        labels=['Up', 'Stable', 'Down']
    )
    
    # Calculate directional accuracy
    accuracy = np.diag(cm).sum() / cm.sum()
    
    # Return results
    direction_metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'labels': ['Up', 'Stable', 'Down']
    }
    
    print(f"\nDirectional prediction accuracy: {accuracy*100:.1f}%")
    
    return direction_metrics

def summarize_comparison(comparison_metrics, direction_metrics=None):
    """
    Provide a summary of the comparison results
    """
    print("\n--- Summary of Model Comparison ---")
    
    # Overall performance assessment
    r2 = comparison_metrics.get('r2', 0)
    if r2 > 0.7:
        performance = "excellent"
    elif r2 > 0.5:
        performance = "good"
    elif r2 > 0.3:
        performance = "moderate"
    else:
        performance = "limited"
    
    print(f"1. The XGBoost model provides {performance} predictions, explaining {r2*100:.1f}% of the variance in APY values.")
    
    # Direction agreement if available
    if direction_metrics:
        accuracy = direction_metrics.get('accuracy', 0)
        if accuracy > 0.7:
            agreement = "strong"
        elif accuracy > 0.5:
            agreement = "moderate"
        else:
            agreement = "weak"
        
        print(f"2. The model shows {agreement} agreement ({accuracy*100:.1f}%) with original direction predictions.")
    
    # Recommendations
    print("\nRecommendations:")
    if r2 > 0.5:
        print("- Use the XGBoost model for APY predictions")
        print("- Focus on protocols with higher predicted APY and lower uncertainty")
    else:
        print("- Use the model as one of multiple signals for investment decisions")
        print("- Consider the model's strengths and limitations when making predictions")
    
    if direction_metrics and direction_metrics.get('accuracy', 0) > 0.6:
        print("- Consider ensemble approaches combining both prediction systems") 
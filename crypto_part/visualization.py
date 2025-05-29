import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def create_plots_dir():
    """
    Create a directory for plots if it doesn't exist
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    return 'plots'

def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Plot feature importance from XGBoost model
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_predictions_vs_actual(y_test, y_pred, save_path=None):
    """
    Create scatter plot of predicted vs actual values
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('Actual APY (%)')
    plt.ylabel('Predicted APY (%)')
    plt.title('Actual vs Predicted APY')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_error_distribution(y_test, y_pred, save_path=None):
    """
    Plot distribution of prediction errors
    """
    errors = y_pred - y_test
    
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.axvline(0, color='red', linestyle='dashed')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(confusion_matrix, labels, save_path=None):
    """
    Plot confusion matrix for directional predictions
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Original vs XGBoost Direction Predictions')
    plt.xlabel('XGBoost Predicted Direction')
    plt.ylabel('Original Predicted Direction')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_apy_distribution_comparison(y_test, y_pred, filtered_df=None, save_path=None):
    """
    Plot distribution comparison between actual, predicted, and filtered APY
    """
    plt.figure(figsize=(15, 8))
    
    # Actual test data distribution
    plt.subplot(1, 3, 1)
    sns.histplot(y_test, bins=30, kde=True, color='blue')
    plt.axvline(y_test.mean(), color='red', linestyle='dashed', linewidth=1, 
                label=f'Mean: {y_test.mean():.2f}%')
    plt.title('Test Set Actual APY')
    plt.xlabel('APY (%)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # XGBoost predictions distribution
    plt.subplot(1, 3, 2)
    sns.histplot(y_pred, bins=30, kde=True, color='green')
    plt.axvline(np.mean(y_pred), color='red', linestyle='dashed', linewidth=1, 
                label=f'Mean: {np.mean(y_pred):.2f}%')
    plt.title('XGBoost Predicted APY')
    plt.xlabel('APY (%)')
    plt.legend()
    
    # Filtered data distribution (if provided)
    plt.subplot(1, 3, 3)
    if filtered_df is not None and 'apy' in filtered_df.columns:
        sns.histplot(filtered_df['apy'], bins=30, kde=True, color='orange')
        plt.axvline(filtered_df['apy'].mean(), color='red', linestyle='dashed', linewidth=1, 
                    label=f'Mean: {filtered_df["apy"].mean():.2f}%')
        plt.title('Filtered Data APY')
    else:
        plt.text(0.5, 0.5, 'Filtered data not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Filtered Data')
    plt.xlabel('APY (%)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_project_performance(comparison_df, metric='Prediction_Error', top_n=10, save_path=None):
    """
    Plot model performance by project
    """
    if 'project' not in comparison_df.columns:
        print("Project column not found in comparison dataframe")
        return
    
    # Group by project and calculate mean, std, count of the metric
    project_metrics = comparison_df.groupby('project').agg({
        metric: ['mean', 'std', 'count']
    }).sort_values((metric, 'count'), ascending=False)
    
    # Get top N projects by count
    top_projects = project_metrics.head(top_n)
    
    plt.figure(figsize=(14, 8))
    plt.bar(
        range(len(top_projects)), 
        top_projects[(metric, 'mean')].abs(),
        yerr=top_projects[(metric, 'std')],
        alpha=0.7
    )
    plt.xticks(range(len(top_projects)), top_projects.index, rotation=45)
    plt.title(f'Mean Absolute {metric} by Project')
    plt.ylabel(f'Mean Absolute {metric}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confidence_vs_error(comparison_df, save_path=None):
    """
    Plot relationship between prediction confidence and error
    """
    if 'Original_PredictedProbability' not in comparison_df.columns:
        print("Original prediction probability not found in comparison dataframe")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        comparison_df['Original_PredictedProbability'], 
        comparison_df['Prediction_Error'].abs(),
        alpha=0.5, c=comparison_df['Actual_APY'], cmap='viridis'
    )
    plt.colorbar(label='Actual APY')
    plt.title('Original Prediction Probability vs Absolute Error')
    plt.xlabel('Original Predicted Probability')
    plt.ylabel('Absolute Prediction Error')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def create_comparison_dashboard(comparison_df, filtered_comparison=None, save_prefix=None):
    """
    Create a comprehensive dashboard of comparison visualizations
    """
    # Create plots directory
    plots_dir = create_plots_dir()
    
    # 1. Actual vs Predicted
    actual_vs_pred_path = os.path.join(plots_dir, f'{save_prefix}_actual_vs_pred.png') if save_prefix else None
    plot_predictions_vs_actual(
        comparison_df['Actual_APY'], 
        comparison_df['XGBoost_Predicted_APY'],
        actual_vs_pred_path
    )
    
    # 2. Error distribution
    error_dist_path = os.path.join(plots_dir, f'{save_prefix}_error_dist.png') if save_prefix else None
    plot_error_distribution(
        comparison_df['Actual_APY'], 
        comparison_df['XGBoost_Predicted_APY'],
        error_dist_path
    )
    
    # 3. Project performance if available
    if 'project' in comparison_df.columns:
        project_perf_path = os.path.join(plots_dir, f'{save_prefix}_project_perf.png') if save_prefix else None
        plot_project_performance(
            comparison_df,
            save_path=project_perf_path
        )
    
    # 4. Confusion matrix if original direction available
    if 'Original_Direction' in comparison_df.columns:
        from sklearn.metrics import confusion_matrix
        
        # Remove rows with missing original direction
        valid_rows = comparison_df.dropna(subset=['Original_Direction'])
        
        if len(valid_rows) > 0:
            cm = confusion_matrix(
                valid_rows['Original_Direction'], 
                valid_rows['XGBoost_Direction'],
                labels=['Up', 'Stable', 'Down']
            )
            
            cm_path = os.path.join(plots_dir, f'{save_prefix}_confusion_matrix.png') if save_prefix else None
            plot_confusion_matrix(cm, ['Up', 'Stable', 'Down'], cm_path)
    
    # 5. Confidence vs error if available
    if 'Original_PredictedProbability' in comparison_df.columns:
        conf_vs_error_path = os.path.join(plots_dir, f'{save_prefix}_conf_vs_error.png') if save_prefix else None
        plot_confidence_vs_error(comparison_df, conf_vs_error_path)
    
    # 6. APY distribution comparison
    apy_dist_path = os.path.join(plots_dir, f'{save_prefix}_apy_dist.png') if save_prefix else None
    filtered_df = filtered_comparison if filtered_comparison is not None else None
    plot_apy_distribution_comparison(
        comparison_df['Actual_APY'], 
        comparison_df['XGBoost_Predicted_APY'],
        filtered_df,
        apy_dist_path
    )

def plot_feature_importance_plotly(importance_df, top_n=20, save_path=None):
    """
    Plot feature importance from XGBoost model using Plotly
    """
    try:
        # Ensure the dataframe has the necessary columns
        if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
            print("Error: importance_df must have 'Feature' and 'Importance' columns")
            return
            
        # Get top N features (handling case where there are fewer features than top_n)
        top_n = min(top_n, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                # For image formats like .png, .jpg, etc.
                image_path = save_path
                if save_path.endswith('.html'):
                    image_path = save_path.replace('.html', '.png')
                fig.write_image(image_path)
                
                # Also save an HTML version for interactivity
                html_path = image_path.replace('.png', '.html')
                if html_path != save_path:  # Don't write html if save_path is already html
                    fig.write_html(html_path)
        
        # Show
        fig.show()
        
    except Exception as e:
        print(f"Error in plot_feature_importance_plotly: {str(e)}")
        print("Continuing with the rest of the analysis...")

def plot_predictions_vs_actual_plotly(y_test, y_pred, save_path=None):
    """
    Create scatter plot of predicted vs actual values using Plotly
    """
    try:
        # Validate inputs
        if len(y_test) != len(y_pred):
            print(f"Error: y_test and y_pred must have the same length. Got {len(y_test)} and {len(y_pred)}")
            return
            
        if len(y_test) == 0:
            print("Error: Empty data provided for plotting")
            return
        
        # Create dataframe
        df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_pred - y_test
        })
        
        # Handle potential NaN values
        df = df.dropna()
        
        if len(df) == 0:
            print("Error: No valid data points after dropping NaN values")
            return
        
        # Create figure
        fig = px.scatter(
            df, 
            x='Actual', 
            y='Predicted',
            color=abs(df['Error']),
            color_continuous_scale='Viridis',
            opacity=0.7,
            title='Actual vs Predicted APY',
            labels={'color': 'Absolute Error'}
        )
        
        # Add perfect prediction line
        min_val = min(df['Actual'].min(), df['Predicted'].min())
        max_val = max(df['Actual'].max(), df['Predicted'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            width=900,
            xaxis_title='Actual APY (%)',
            yaxis_title='Predicted APY (%)'
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                # For image formats like .png, .jpg, etc.
                image_path = save_path
                if save_path.endswith('.html'):
                    image_path = save_path.replace('.html', '.png')
                fig.write_image(image_path)
                
                # Also save an HTML version for interactivity
                html_path = image_path.replace('.png', '.html')
                if html_path != save_path:  # Don't write html if save_path is already html
                    fig.write_html(html_path)
        
        # Show
        fig.show()
        
    except Exception as e:
        print(f"Error in plot_predictions_vs_actual_plotly: {str(e)}")
        print("Continuing with the rest of the analysis...")

def plot_correlation_heatmap_plotly(df, target='apy', save_path=None):
    """
    Plot correlation heatmap to identify potential data leakage issues
    """
    try:
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        # Drop columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with 0 to allow correlation calculation
        numeric_df = numeric_df.fillna(0)
        
        # Check if we have enough numeric columns
        if numeric_df.shape[1] < 2:
            print("Warning: Not enough numeric columns for correlation heatmap")
            return
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Sort by correlation with target
        if target in corr.columns:
            target_corr = corr[target].abs().sort_values(ascending=False)
            sorted_cols = target_corr.index.tolist()
            corr = corr.loc[sorted_cols, sorted_cols]
        
        # Create figure
        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap',
            zmin=-1, zmax=1
        )
        
        # Highlight potential data leakage
        leak_features = ['apyBase', 'apyReward', 'apyPct7D', 'apyPct30D']
        annotations = []
        
        for feature in leak_features:
            if feature in corr.columns and target in corr.columns:
                row = corr.index.get_loc(feature)
                col = corr.columns.get_loc(target)
                corr_value = corr.iloc[row, col]
                
                if abs(corr_value) > 0.5:  # High correlation threshold
                    annotations.append(
                        dict(
                            x=col,
                            y=row,
                            xref='x',
                            yref='y',
                            text='Potential Leakage',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='black',
                            ax=0,
                            ay=-30
                        )
                    )
        
        fig.update_layout(
            annotations=annotations,
            height=800,
            width=900
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                # For image formats like .png, .jpg, etc.
                image_path = save_path
                if save_path.endswith('.html'):
                    image_path = save_path.replace('.html', '.png')
                fig.write_image(image_path)
                
                # Also save an HTML version for interactivity
                html_path = image_path.replace('.png', '.html')
                if html_path != save_path:  # Don't write html if save_path is already html
                    fig.write_html(html_path)
        
        # Show
        fig.show()
        
    except Exception as e:
        print(f"Error in plot_correlation_heatmap_plotly: {str(e)}")
        print("Continuing with the rest of the analysis...")

def create_apy_plotly_dashboard(df, filtered_df=None, save_prefix=None):
    """
    Create a comprehensive APY analysis dashboard using Plotly
    """
    plots_dir = create_plots_dir()
    
    try:
        # 1. APY Distribution
        fig1 = px.histogram(
            df,
            x='apy',
            nbins=50,
            opacity=0.7,
            title='APY Distribution',
            color_discrete_sequence=['blue']
        )
        
        fig1.add_vline(x=df['apy'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['apy'].mean():.2f}%")
        
        if filtered_df is not None:
            fig1.add_trace(
                go.Histogram(
                    x=filtered_df['apy'],
                    nbinsx=50,
                    opacity=0.5,
                    name='Filtered APY (3-150%)',
                    marker_color='green'
                )
            )
        
        # Save if prefix provided
        if save_prefix:
            save_path = os.path.join(plots_dir, f'{save_prefix}_apy_distribution_plotly.html')
            fig1.write_html(save_path)
        
        fig1.show()
        
        # 2. APY vs TVL Scatter
        # Create a copy of the dataframe and calculate percentage of total TVL
        plot_df = df.copy()
        
        # Handle possible NaN values in tvlUsd
        plot_df['tvlUsd'] = plot_df['tvlUsd'].fillna(0)
        
        # Calculate percentage of total TVL
        total_tvl = plot_df['tvlUsd'].sum()
        if total_tvl > 0:  # Prevent division by zero
            plot_df['percentage_of_total_tvl'] = (plot_df['tvlUsd'] / total_tvl) * 100
        else:
            plot_df['percentage_of_total_tvl'] = 0
            
        # Limit to top projects by TVL to avoid overloading the plot
        # Get the top 20 projects by TVL
        top_projects = plot_df.groupby('project')['tvlUsd'].sum().nlargest(20).index.tolist()
        plot_df_filtered = plot_df[plot_df['project'].isin(top_projects)]
        
        fig2 = px.scatter(
            plot_df_filtered,  # Use the filtered dataframe with top projects
            x='tvlUsd',
            y='apy',
            color='project',
            size='percentage_of_total_tvl',
            hover_data=['chain', 'symbol'],
            log_x=True,
            title='APY vs TVL by Top 20 Projects',
            opacity=0.7
        )
        
        # Save if prefix provided
        if save_prefix:
            save_path = os.path.join(plots_dir, f'{save_prefix}_apy_vs_tvl_plotly.html')
            fig2.write_html(save_path)
        
        fig2.show()
        
        # 3. Correlation heatmap
        plot_correlation_heatmap_plotly(
            df, 
            target='apy',
            save_path=os.path.join(plots_dir, f'{save_prefix}_correlation_heatmap_plotly.html') if save_prefix else None
        )
    except Exception as e:
        print(f"Error in create_apy_plotly_dashboard: {str(e)}")
        print("Continuing with the rest of the analysis...")

# Add alias for the function name used in main_training_improved.py
plot_prediction_vs_actual = plot_predictions_vs_actual

def plot_residuals(y_test, y_pred, save_path=None):
    """
    Plot residuals (prediction errors) analysis
    """
    residuals = y_pred - y_test
    
    plt.figure(figsize=(15, 10))
    
    # Residuals vs predicted values
    plt.subplot(2, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Residuals distribution
    plt.subplot(2, 3, 2)
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # Q-Q plot for normality check
    plt.subplot(2, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs actual values
    plt.subplot(2, 3, 4)
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Absolute residuals vs predicted
    plt.subplot(2, 3, 5)
    plt.scatter(y_pred, np.abs(residuals), alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('|Residuals|')
    plt.title('Absolute Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Box plot of residuals
    plt.subplot(2, 3, 6)
    plt.boxplot(residuals)
    plt.ylabel('Residuals')
    plt.title('Residuals Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_prediction_intervals(y_test, y_pred, pred_std, confidence=0.95, save_path=None):
    """
    Plot prediction intervals for ensemble models
    """
    # Calculate confidence intervals
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Create confidence intervals
    lower_bound = y_pred - z_score * pred_std
    upper_bound = y_pred + z_score * pred_std
    
    # Sort by predicted values for better visualization
    sorted_indices = np.argsort(y_pred)
    y_test_sorted = y_test.iloc[sorted_indices] if hasattr(y_test, 'iloc') else y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    lower_sorted = lower_bound[sorted_indices]
    upper_sorted = upper_bound[sorted_indices]
    pred_std_sorted = pred_std[sorted_indices]
    
    plt.figure(figsize=(15, 10))
    
    # Main prediction intervals plot
    plt.subplot(2, 2, 1)
    # Plot confidence interval
    plt.fill_between(range(len(y_pred_sorted)), lower_sorted, upper_sorted, 
                     alpha=0.3, color='blue', label=f'{confidence*100}% Confidence Interval')
    
    # Plot predictions and actual values
    plt.plot(y_pred_sorted, 'b-', label='Predicted', linewidth=1)
    plt.scatter(range(len(y_test_sorted)), y_test_sorted, alpha=0.6, color='red', 
                s=20, label='Actual', zorder=3)
    
    plt.xlabel('Sample Index (sorted by prediction)')
    plt.ylabel('APY (%)')
    plt.title(f'Prediction Intervals ({confidence*100}% Confidence)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction uncertainty vs predicted value
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, pred_std, alpha=0.6)
    plt.xlabel('Predicted APY (%)')
    plt.ylabel('Prediction Std')
    plt.title('Prediction Uncertainty vs Predicted Value')
    plt.grid(True, alpha=0.3)
    
    # Coverage analysis
    plt.subplot(2, 2, 3)
    coverage = ((y_test >= lower_bound) & (y_test <= upper_bound))
    coverage_rate = coverage.mean() if hasattr(coverage, 'mean') else np.mean(coverage)
    
    # Plot coverage by prediction bins
    n_bins = 10
    pred_bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_coverage = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i+1])
        if np.sum(mask) > 0:
            bin_cov = coverage[mask].mean() if hasattr(coverage[mask], 'mean') else np.mean(coverage[mask])
            bin_coverage.append(bin_cov)
            bin_centers.append((pred_bins[i] + pred_bins[i+1]) / 2)
    
    plt.bar(bin_centers, bin_coverage, alpha=0.7, width=(pred_bins[1] - pred_bins[0]) * 0.8)
    plt.axhline(y=confidence, color='red', linestyle='--', 
                label=f'Target Coverage ({confidence*100}%)')
    plt.axhline(y=coverage_rate, color='green', linestyle='-', 
                label=f'Actual Coverage ({coverage_rate*100:.1f}%)')
    plt.xlabel('Predicted APY Bins')
    plt.ylabel('Coverage Rate')
    plt.title('Coverage Rate by Prediction Bins')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction uncertainty distribution
    plt.subplot(2, 2, 4)
    sns.histplot(pred_std, bins=30, kde=True)
    plt.axvline(np.mean(pred_std), color='red', linestyle='--', 
                label=f'Mean Std: {np.mean(pred_std):.3f}')
    plt.xlabel('Prediction Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print coverage statistics
    print(f"\nPrediction Interval Analysis:")
    print(f"Target coverage: {confidence*100:.1f}%")
    print(f"Actual coverage: {coverage_rate*100:.1f}%")
    print(f"Mean prediction uncertainty: {np.mean(pred_std):.4f}")
    print(f"Max prediction uncertainty: {np.max(pred_std):.4f}")
    print(f"Min prediction uncertainty: {np.min(pred_std):.4f}") 
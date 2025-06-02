#!/usr/bin/env python3
"""
Stock Portfolio Recommendations Module

This module provides investment recommendations and portfolio analysis:
1. Sector-based portfolio segmentation
2. Risk-adjusted performance analysis
3. Investment strategies (Conservative/Balanced/Aggressive)
4. Stock scoring and ranking system
5. Portfolio optimization recommendations
6. Performance visualization and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_processing_improved import improved_preprocess_pipeline

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def analyze_stock_performance(df):
    """Analyze historical performance of stocks"""
    print("📈 ANALYZING STOCK PERFORMANCE...")
    
    # Calculate performance metrics for each stock
    stock_metrics = []
    
    for symbol in df['symbol'].unique():
        stock_data = df[df['symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if len(stock_data) > 20:  # Ensure sufficient data
            # Basic metrics
            avg_return = stock_data['return'].mean()
            volatility = stock_data['return'].std()
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Price performance
            first_price = stock_data['price'].iloc[0]
            last_price = stock_data['price'].iloc[-1]
            total_return = (last_price - first_price) / first_price
            
            # Risk metrics
            max_drawdown = calculate_max_drawdown(stock_data['price'])
            downside_volatility = calculate_downside_volatility(stock_data['return'])
            
            # Get sector and industry
            sector = stock_data['sector'].iloc[0]
            industry = stock_data['industry'].iloc[0]
            
            stock_metrics.append({
                'symbol': symbol,
                'sector': sector,
                'industry': industry,
                'avg_daily_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'downside_volatility': downside_volatility,
                'trading_days': len(stock_data),
                'current_price': last_price
            })
    
    performance_df = pd.DataFrame(stock_metrics)
    
    print(f"  ✅ Performance analyzed for {len(performance_df)} stocks")
    return performance_df

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_downside_volatility(returns, target_return=0):
    """Calculate downside volatility"""
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return 0
    return downside_returns.std()

def segment_stocks_by_sector(performance_df):
    """Segment stocks by sector and analyze performance"""
    print("🏢 SEGMENTING STOCKS BY SECTOR...")
    
    if performance_df.empty:
        print("  ❌ No performance data available for sector analysis")
        return {}
    
    # Check if required columns exist
    required_cols = ['sector', 'avg_daily_return', 'volatility', 'sharpe_ratio', 'total_return', 'max_drawdown', 'symbol']
    missing_cols = [col for col in required_cols if col not in performance_df.columns]
    if missing_cols:
        print(f"  ❌ Missing required columns: {missing_cols}")
        return {}
    
    sector_analysis = {}
    sectors = performance_df['sector'].dropna().unique()
    
    if len(sectors) == 0:
        print("  ❌ No sectors found in the data")
        return {}
    
    print(f"  📊 Analyzing {len(sectors)} sectors...")
    
    for sector in sectors:
        sector_stocks = performance_df[performance_df['sector'] == sector].copy()
        
        # Skip sectors with insufficient data
        if len(sector_stocks) < 2:
            print(f"  ⚠️ Skipping {sector}: insufficient stocks ({len(sector_stocks)})")
            continue
        
        # Check for valid sharpe ratios
        valid_sharpe = sector_stocks['sharpe_ratio'].dropna()
        if len(valid_sharpe) == 0:
            print(f"  ⚠️ Skipping {sector}: no valid Sharpe ratios")
            continue
        
        # Find top stock safely
        try:
            best_stock_idx = sector_stocks['sharpe_ratio'].idxmax()
            top_stock = sector_stocks.loc[best_stock_idx, 'symbol']
        except (ValueError, KeyError):
            # Fallback to first stock if idxmax fails
            top_stock = sector_stocks['symbol'].iloc[0] if len(sector_stocks) > 0 else 'Unknown'
        
        sector_analysis[sector] = {
            'count': len(sector_stocks),
            'avg_return': sector_stocks['avg_daily_return'].mean(),
            'avg_volatility': sector_stocks['volatility'].mean(),
            'avg_sharpe': sector_stocks['sharpe_ratio'].mean(),
            'median_total_return': sector_stocks['total_return'].median(),
            'top_stock': top_stock,
            'avg_max_drawdown': sector_stocks['max_drawdown'].mean(),
            'stocks': sector_stocks['symbol'].tolist()
        }
    
    if not sector_analysis:
        print("  ❌ No valid sectors found after filtering")
        return {}
    
    print(f"  ✅ Analyzed {len(sector_analysis)} sectors")
    
    # Print sector summary
    print(f"\n🏢 SECTOR PERFORMANCE SUMMARY:")
    print("-" * 100)
    print(f"{'Sector':<25} {'Count':<6} {'Avg Return':<12} {'Volatility':<12} {'Sharpe':<8} {'Total Return':<12}")
    print("-" * 100)
    
    for sector, metrics in sorted(sector_analysis.items(), key=lambda x: x[1]['avg_sharpe'], reverse=True):
        print(f"{sector[:24]:<25} {metrics['count']:<6} {metrics['avg_return']*100:<11.2f}% "
              f"{metrics['avg_volatility']*100:<11.2f}% {metrics['avg_sharpe']:<7.3f} "
              f"{metrics['median_total_return']*100:<11.1f}%")
    
    return sector_analysis

def create_investment_strategies(performance_df, sector_analysis):
    """Create investment strategies based on risk profiles"""
    print("💼 CREATING INVESTMENT STRATEGIES...")
    
    if performance_df.empty:
        print("  ❌ No performance data available for strategy creation")
        return {}
    
    if not sector_analysis:
        print("  ❌ No sector analysis available for strategy creation")
        return {}
    
    # Define strategies
    strategies = {
        'Conservative': {
            'description': 'Low risk, stable returns, focus on dividends and utilities',
            'target_sectors': ['Utilities', 'Consumer Staples', 'Real Estate'],
            'risk_threshold': 0.15,  # Max volatility
            'min_sharpe': 0.3,
            'allocation': 'Equal weight top performers'
        },
        'Balanced': {
            'description': 'Moderate risk, diversified across sectors',
            'target_sectors': ['Technology', 'Health Care', 'Financials', 'Industrials'],
            'risk_threshold': 0.25,
            'min_sharpe': 0.2,
            'allocation': 'Risk-adjusted weight'
        },
        'Aggressive': {
            'description': 'High risk, high growth potential',
            'target_sectors': ['Technology', 'Consumer Discretionary', 'Communication Services'],
            'risk_threshold': 0.40,
            'min_sharpe': 0.1,
            'allocation': 'Performance-weighted'
        }
    }
    
    strategy_recommendations = {}
    
    for strategy_name, strategy_config in strategies.items():
        print(f"\n📊 Building {strategy_name} Strategy:")
        
        try:
            # Filter stocks based on strategy criteria
            filtered_stocks = performance_df[
                (performance_df['volatility'] <= strategy_config['risk_threshold']) &
                (performance_df['sharpe_ratio'] >= strategy_config['min_sharpe'])
            ].copy()
            
            if len(filtered_stocks) == 0:
                print(f"  ⚠️ No stocks meet {strategy_name} criteria, relaxing constraints...")
                # Relax constraints if no stocks meet criteria
                filtered_stocks = performance_df[
                    performance_df['volatility'] <= strategy_config['risk_threshold'] * 1.5
                ].copy()
                
                if len(filtered_stocks) == 0:
                    print(f"  ❌ Still no stocks available for {strategy_name}, using top performers...")
                    filtered_stocks = performance_df.nlargest(20, 'sharpe_ratio')
            
            # Prefer target sectors if they exist in our data
            available_sectors = set(performance_df['sector'].dropna().unique())
            target_sectors_available = [s for s in strategy_config['target_sectors'] if s in available_sectors]
            
            if target_sectors_available:
                preferred_stocks = filtered_stocks[
                    filtered_stocks['sector'].isin(target_sectors_available)
                ]
            else:
                preferred_stocks = filtered_stocks
            
            if len(preferred_stocks) < 5:  # Ensure minimum diversification
                print(f"  ⚠️ Insufficient preferred stocks ({len(preferred_stocks)}), expanding selection...")
                preferred_stocks = filtered_stocks.nlargest(min(15, len(filtered_stocks)), 'sharpe_ratio')
            
            if len(preferred_stocks) == 0:
                print(f"  ❌ No stocks available for {strategy_name} strategy")
                continue
            
            # Score stocks
            preferred_stocks = score_stocks(preferred_stocks)
            
            # Select top stocks based on strategy
            if strategy_name == 'Conservative':
                selected_stocks = preferred_stocks.nsmallest(min(15, len(preferred_stocks)), 'volatility').nlargest(min(10, len(preferred_stocks)), 'composite_score')
            elif strategy_name == 'Balanced':
                selected_stocks = preferred_stocks.nlargest(min(15, len(preferred_stocks)), 'composite_score')
            else:  # Aggressive
                selected_stocks = preferred_stocks.nlargest(min(20, len(preferred_stocks)), 'total_return').nlargest(min(12, len(preferred_stocks)), 'composite_score')
            
            if len(selected_stocks) == 0:
                print(f"  ❌ No stocks selected for {strategy_name} strategy")
                continue
            
            strategy_recommendations[strategy_name] = {
                'config': strategy_config,
                'stocks': selected_stocks,
                'expected_return': selected_stocks['avg_daily_return'].mean(),
                'expected_volatility': selected_stocks['volatility'].mean(),
                'expected_sharpe': selected_stocks['sharpe_ratio'].mean(),
                'sector_diversification': selected_stocks['sector'].nunique(),
                'count': len(selected_stocks)
            }
            
            print(f"  ✅ Selected {len(selected_stocks)} stocks")
            print(f"  ✅ Expected daily return: {selected_stocks['avg_daily_return'].mean()*100:.3f}%")
            print(f"  ✅ Expected volatility: {selected_stocks['volatility'].mean()*100:.2f}%")
            print(f"  ✅ Sector diversification: {selected_stocks['sector'].nunique()} sectors")
            
        except Exception as e:
            print(f"  ❌ Error creating {strategy_name} strategy: {e}")
            continue
    
    if not strategy_recommendations:
        print("  ❌ No strategies could be created successfully")
    
    return strategy_recommendations

def score_stocks(stocks_df):
    """Create composite scoring for stocks"""
    if stocks_df.empty:
        print("  ⚠️ No stocks to score")
        return stocks_df
    
    # Normalize metrics (0-1 scale)
    stocks_df = stocks_df.copy()
    
    # Helper function to safely normalize
    def safe_normalize(series, higher_is_better=True):
        if series.isna().all() or series.std() == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        
        min_val = series.min()
        max_val = series.max()
        
        if min_val == max_val:
            return pd.Series([0.5] * len(series), index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        
        if not higher_is_better:
            normalized = 1 - normalized
            
        return normalized.fillna(0.5)
    
    # Higher is better
    stocks_df['return_score'] = safe_normalize(stocks_df['avg_daily_return'], higher_is_better=True)
    stocks_df['sharpe_score'] = safe_normalize(stocks_df['sharpe_ratio'], higher_is_better=True)
    stocks_df['total_return_score'] = safe_normalize(stocks_df['total_return'], higher_is_better=True)
    
    # Lower is better (invert)
    stocks_df['volatility_score'] = safe_normalize(stocks_df['volatility'], higher_is_better=False)
    stocks_df['drawdown_score'] = safe_normalize(stocks_df['max_drawdown'], higher_is_better=False)
    
    # Composite score
    stocks_df['composite_score'] = (
        stocks_df['return_score'] * 0.25 +
        stocks_df['sharpe_score'] * 0.30 +
        stocks_df['total_return_score'] * 0.20 +
        stocks_df['volatility_score'] * 0.15 +
        stocks_df['drawdown_score'] * 0.10
    )
    
    return stocks_df

def create_portfolio_visualizations(strategy_recommendations, sector_analysis, performance_df):
    """Create comprehensive portfolio visualization"""
    print(f"\n📊 Creating portfolio visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('../results/plots', exist_ok=True)
    
    # Figure 1: Strategy Comparison and Sector Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stock Portfolio Analysis & Strategy Recommendations', fontsize=16, fontweight='bold')
    
    # Strategy Expected Returns vs Risk
    strategy_names = []
    expected_returns = []
    expected_volatilities = []
    
    for name, strategy in strategy_recommendations.items():
        strategy_names.append(name)
        expected_returns.append(strategy['expected_return'] * 252 * 100)  # Annualized %
        expected_volatilities.append(strategy['expected_volatility'] * np.sqrt(252) * 100)  # Annualized %
    
    colors = ['green', 'blue', 'red']
    axes[0,0].scatter(expected_volatilities, expected_returns, c=colors, s=200, alpha=0.7)
    
    for i, name in enumerate(strategy_names):
        axes[0,0].annotate(name, (expected_volatilities[i], expected_returns[i]), 
                          xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    axes[0,0].set_xlabel('Expected Volatility (% annually)')
    axes[0,0].set_ylabel('Expected Return (% annually)')
    axes[0,0].set_title('Risk-Return Profile of Strategies')
    axes[0,0].grid(True, alpha=0.3)
    
    # Sector Performance
    sector_returns = [metrics['avg_return'] * 252 * 100 for metrics in sector_analysis.values()]
    sector_names = list(sector_analysis.keys())
    
    # Sort for better visualization
    sorted_sectors = sorted(zip(sector_names, sector_returns), key=lambda x: x[1], reverse=True)
    sector_names_sorted = [x[0] for x in sorted_sectors]
    sector_returns_sorted = [x[1] for x in sorted_sectors]
    
    axes[0,1].barh(range(len(sector_names_sorted)), sector_returns_sorted, color='lightblue', alpha=0.8)
    axes[0,1].set_yticks(range(len(sector_names_sorted)))
    axes[0,1].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in sector_names_sorted])
    axes[0,1].set_xlabel('Average Annual Return (%)')
    axes[0,1].set_title('Sector Performance Ranking')
    axes[0,1].grid(True, alpha=0.3)
    
    # Strategy Stock Count and Diversification
    strategy_counts = [strategy['count'] for strategy in strategy_recommendations.values()]
    strategy_diversification = [strategy['sector_diversification'] for strategy in strategy_recommendations.values()]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    axes[1,0].bar(x - width/2, strategy_counts, width, label='Number of Stocks', color='lightgreen', alpha=0.8)
    axes[1,0].bar(x + width/2, strategy_diversification, width, label='Sector Diversification', color='lightcoral', alpha=0.8)
    
    axes[1,0].set_xlabel('Strategy')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Strategy Composition')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(strategy_names)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Top Stocks Overall (by composite score)
    top_stocks = score_stocks(performance_df).nlargest(15, 'composite_score')
    
    axes[1,1].barh(range(len(top_stocks)), top_stocks['composite_score'], color='gold', alpha=0.8)
    axes[1,1].set_yticks(range(len(top_stocks)))
    axes[1,1].set_yticklabels(top_stocks['symbol'])
    axes[1,1].set_xlabel('Composite Score')
    axes[1,1].set_title('Top 15 Stocks (Overall Ranking)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/portfolio_allocation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Detailed Strategy Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Strategy Analysis & Top Stock Recommendations', fontsize=16, fontweight='bold')
    
    # Risk-Return Scatter for all stocks colored by sector
    sample_size = min(200, len(performance_df))
    sample_df = performance_df.sample(n=sample_size)
    
    sectors = sample_df['sector'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
    
    for i, sector in enumerate(sectors):
        sector_data = sample_df[sample_df['sector'] == sector]
        axes[0,0].scatter(sector_data['volatility'] * np.sqrt(252) * 100, 
                         sector_data['avg_daily_return'] * 252 * 100,
                         c=[colors[i]], label=sector[:15], alpha=0.6, s=20)
    
    axes[0,0].set_xlabel('Volatility (% annually)')
    axes[0,0].set_ylabel('Return (% annually)')
    axes[0,0].set_title('Stock Risk-Return by Sector')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Sharpe Ratio Distribution by Strategy
    for i, (strategy_name, strategy) in enumerate(strategy_recommendations.items()):
        strategy_stocks = strategy['stocks']
        axes[0,1].hist(strategy_stocks['sharpe_ratio'], bins=15, alpha=0.6, 
                      label=strategy_name, color=colors[i % len(colors)])
    
    axes[0,1].set_xlabel('Sharpe Ratio')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Sharpe Ratio Distribution by Strategy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Sector Allocation for Balanced Strategy
    balanced_stocks = strategy_recommendations['Balanced']['stocks']
    sector_allocation = balanced_stocks['sector'].value_counts()
    
    axes[1,0].pie(sector_allocation.values, labels=sector_allocation.index, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Balanced Strategy - Sector Allocation')
    
    # Performance Comparison: Actual vs Strategy Expected
    strategy_metrics = []
    for name, strategy in strategy_recommendations.items():
        strategy_metrics.append({
            'Strategy': name,
            'Expected_Return': strategy['expected_return'] * 252 * 100,
            'Expected_Volatility': strategy['expected_volatility'] * np.sqrt(252) * 100,
            'Expected_Sharpe': strategy['expected_sharpe']
        })
    
    strategy_df = pd.DataFrame(strategy_metrics)
    
    x = np.arange(len(strategy_df))
    width = 0.25
    
    axes[1,1].bar(x - width, strategy_df['Expected_Return'], width, label='Expected Return (%)', alpha=0.8)
    axes[1,1].bar(x, strategy_df['Expected_Volatility'], width, label='Expected Volatility (%)', alpha=0.8)
    axes[1,1].bar(x + width, strategy_df['Expected_Sharpe'] * 10, width, label='Sharpe Ratio (×10)', alpha=0.8)
    
    axes[1,1].set_xlabel('Strategy')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Strategy Performance Metrics')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(strategy_df['Strategy'])
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/top_protocols_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Portfolio visualizations saved to ../results/plots/")

def generate_portfolio_recommendations_report(strategy_recommendations, sector_analysis, performance_df):
    """Generate comprehensive portfolio recommendations report"""
    print(f"\n📋 GENERATING PORTFOLIO RECOMMENDATIONS REPORT...")
    
    # Create detailed recommendations DataFrame
    all_recommendations = []
    
    for strategy_name, strategy in strategy_recommendations.items():
        strategy_stocks = strategy['stocks'].copy()
        strategy_stocks['strategy'] = strategy_name
        strategy_stocks['rank_in_strategy'] = range(1, len(strategy_stocks) + 1)
        all_recommendations.append(strategy_stocks)
    
    recommendations_df = pd.concat(all_recommendations, ignore_index=True)
    
    # Add investment sizing recommendations
    recommendations_df['recommended_weight'] = calculate_recommended_weights(recommendations_df)
    
    # Save detailed recommendations
    recommendations_df.to_csv('../results/reports/portfolio_recommendations.csv', index=False)
    
    # Create strategy summary
    strategy_summary = []
    for name, strategy in strategy_recommendations.items():
        strategy_summary.append({
            'Strategy': name,
            'Description': strategy['config']['description'],
            'Stock_Count': strategy['count'],
            'Sector_Diversification': strategy['sector_diversification'],
            'Expected_Annual_Return': f"{strategy['expected_return'] * 252 * 100:.2f}%",
            'Expected_Annual_Volatility': f"{strategy['expected_volatility'] * np.sqrt(252) * 100:.2f}%",
            'Expected_Sharpe_Ratio': f"{strategy['expected_sharpe']:.3f}",
            'Risk_Level': get_risk_level(strategy['expected_volatility'] * np.sqrt(252))
        })
    
    strategy_summary_df = pd.DataFrame(strategy_summary)
    strategy_summary_df.to_csv('../results/reports/strategy_summary.csv', index=False)
    
    print("✅ Portfolio recommendations saved to ../results/reports/portfolio_recommendations.csv")
    print("✅ Strategy summary saved to ../results/reports/strategy_summary.csv")
    
    return recommendations_df, strategy_summary_df

def calculate_recommended_weights(recommendations_df):
    """Calculate recommended portfolio weights based on composite scores"""
    weights = []
    
    for strategy in recommendations_df['strategy'].unique():
        strategy_stocks = recommendations_df[recommendations_df['strategy'] == strategy]
        
        # Weight based on composite score (higher score = higher weight)
        raw_weights = strategy_stocks['composite_score']
        normalized_weights = raw_weights / raw_weights.sum()
        
        weights.extend(normalized_weights.tolist())
    
    return weights

def get_risk_level(annual_volatility):
    """Categorize risk level based on volatility"""
    if annual_volatility < 0.15:
        return "Low"
    elif annual_volatility < 0.25:
        return "Medium"
    elif annual_volatility < 0.35:
        return "High"
    else:
        return "Very High"

def main_portfolio_analysis():
    """Main function for portfolio analysis"""
    print("="*80)
    print("💼 STOCK PORTFOLIO ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    # Step 1: Load data
    print("\n📊 STEP 1: DATA LOADING")
    X, y, feature_cols, df = improved_preprocess_pipeline()
    
    if df is None:
        print("❌ Failed to load data")
        return None, None, None
    
    # Step 2: Analyze stock performance
    print("\n📈 STEP 2: STOCK PERFORMANCE ANALYSIS")
    performance_df = analyze_stock_performance(df)
    
    # Step 3: Sector segmentation
    print("\n🏢 STEP 3: SECTOR ANALYSIS")
    sector_analysis = segment_stocks_by_sector(performance_df)
    
    # Step 4: Create investment strategies
    print("\n💼 STEP 4: INVESTMENT STRATEGY CREATION")
    strategy_recommendations = create_investment_strategies(performance_df, sector_analysis)
    
    # Step 5: Create visualizations
    print("\n📊 STEP 5: PORTFOLIO VISUALIZATIONS")
    create_portfolio_visualizations(strategy_recommendations, sector_analysis, performance_df)
    
    # Step 6: Generate reports
    print("\n📋 STEP 6: GENERATING REPORTS")
    recommendations_df, strategy_summary_df = generate_portfolio_recommendations_report(
        strategy_recommendations, sector_analysis, performance_df)
    
    print("\n" + "="*80)
    print("✅ PORTFOLIO ANALYSIS COMPLETE!")
    print("="*80)
    
    return recommendations_df, sector_analysis, strategy_recommendations

if __name__ == "__main__":
    # Run the portfolio analysis
    recommendations, sector_analysis, strategies = main_portfolio_analysis()
    if recommendations is not None:
        print(f"\n🎉 Generated recommendations for {len(recommendations)} stock positions")
        print(f"💼 Created {len(strategies)} investment strategies")
        print(f"🏢 Analyzed {len(sector_analysis)} sectors") 
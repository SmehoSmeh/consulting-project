#!/usr/bin/env python3
"""
Portfolio Recommendations: Stablecoin vs Non-Stablecoin Analysis

This script provides comprehensive portfolio recommendations based on:
1. Stablecoin vs Non-stablecoin protocol classification
2. Risk-adjusted return analysis
3. Diversification strategies
4. Investment allocation recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from data_processing_improved import load_data, improved_preprocess_pipeline

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def classify_stablecoin_protocols(df):
    """Classify protocols into stablecoin and non-stablecoin categories"""
    
    # Define stablecoin indicators
    stablecoin_symbols = [
        'USDT', 'USDC', 'DAI', 'BUSD', 'USDD', 'FRAX', 'TUSD', 'USDP', 'LUSD',
        'USDN', 'MIM', 'FEI', 'OUSD', 'USTC', 'GUSD', 'SUSD', 'DOLA', 'ALUSD'
    ]
    
    # Create stablecoin classification
    df['is_stablecoin_protocol'] = False
    df['risk_category'] = 'High-Risk'
    df['expected_apy_range'] = '5-50%'
    
    # Check if protocol involves stablecoins
    if 'symbol' in df.columns:
        for symbol in stablecoin_symbols:
            mask = df['symbol'].str.contains(symbol, case=False, na=False)
            df.loc[mask, 'is_stablecoin_protocol'] = True
            df.loc[mask, 'risk_category'] = 'Low-Risk'
            df.loc[mask, 'expected_apy_range'] = '1-15%'
    
    # Additional classification based on project names
    stablecoin_projects = ['compound', 'aave', 'maker', 'curve']
    if 'project' in df.columns:
        for project in stablecoin_projects:
            mask = df['project'].str.contains(project, case=False, na=False)
            # Only classify as stablecoin if APY is reasonable for stablecoins
            reasonable_apy = df['apy'] <= 20
            df.loc[mask & reasonable_apy, 'is_stablecoin_protocol'] = True
            df.loc[mask & reasonable_apy, 'risk_category'] = 'Low-Risk'
            df.loc[mask & reasonable_apy, 'expected_apy_range'] = '1-15%'
    
    return df

def analyze_portfolio_segments(df):
    """Analyze stablecoin vs non-stablecoin segments"""
    
    print("="*80)
    print("💼 PORTFOLIO SEGMENTATION ANALYSIS")
    print("="*80)
    
    # Classify protocols
    df = classify_stablecoin_protocols(df)
    
    # Split into segments
    stablecoin_protocols = df[df['is_stablecoin_protocol'] == True].copy()
    non_stablecoin_protocols = df[df['is_stablecoin_protocol'] == False].copy()
    
    print(f"\n📊 PORTFOLIO BREAKDOWN:")
    print(f"  Total protocols analyzed: {len(df)}")
    print(f"  Stablecoin protocols: {len(stablecoin_protocols)} ({len(stablecoin_protocols)/len(df)*100:.1f}%)")
    print(f"  Non-stablecoin protocols: {len(non_stablecoin_protocols)} ({len(non_stablecoin_protocols)/len(df)*100:.1f}%)")
    
    # Analyze each segment
    segments = {
        'Stablecoin': stablecoin_protocols,
        'Non-Stablecoin': non_stablecoin_protocols
    }
    
    segment_analysis = {}
    
    for segment_name, segment_df in segments.items():
        if len(segment_df) > 0:
            analysis = {
                'count': len(segment_df),
                'avg_apy': segment_df['apy'].mean(),
                'median_apy': segment_df['apy'].median(),
                'std_apy': segment_df['apy'].std(),
                'min_apy': segment_df['apy'].min(),
                'max_apy': segment_df['apy'].max(),
                'total_tvl': segment_df['tvlUsd'].sum(),
                'avg_tvl': segment_df['tvlUsd'].mean(),
                'total_volume': segment_df['volumeUsd1d'].sum(),
                'avg_volume': segment_df['volumeUsd1d'].mean(),
                'protocols': segment_df
            }
            segment_analysis[segment_name] = analysis
            
            print(f"\n💰 {segment_name.upper()} PROTOCOLS ANALYSIS:")
            print(f"  Protocol count: {analysis['count']}")
            print(f"  APY range: {analysis['min_apy']:.2f}% - {analysis['max_apy']:.2f}%")
            print(f"  Average APY: {analysis['avg_apy']:.2f}% (±{analysis['std_apy']:.2f}%)")
            print(f"  Median APY: {analysis['median_apy']:.2f}%")
            print(f"  Total TVL: ${analysis['total_tvl']:,.0f}")
            print(f"  Average TVL: ${analysis['avg_tvl']:,.0f}")
            print(f"  Total Volume: ${analysis['total_volume']:,.0f}")
            print(f"  Risk Level: {'Low' if segment_name == 'Stablecoin' else 'High'}")
    
    return segment_analysis

def generate_top_protocols_by_segment(segment_analysis):
    """Generate top protocol recommendations for each segment"""
    
    print(f"\n🏆 TOP PROTOCOL RECOMMENDATIONS BY SEGMENT")
    print("="*80)
    
    recommendations = {}
    
    for segment_name, analysis in segment_analysis.items():
        protocols_df = analysis['protocols']
        
        if len(protocols_df) > 0:
            # Calculate risk-adjusted scores
            protocols_df = protocols_df.copy()
            protocols_df['apy_tvl_score'] = protocols_df['apy'] * np.log(protocols_df['tvlUsd'])
            protocols_df['volume_efficiency'] = protocols_df['volumeUsd1d'] / protocols_df['tvlUsd']
            protocols_df['risk_adjusted_score'] = (
                protocols_df['apy'] * 0.4 +
                np.log10(protocols_df['tvlUsd']) * 0.3 +
                np.log10(protocols_df['volumeUsd1d']) * 0.2 +
                protocols_df['volume_efficiency'] * 100 * 0.1
            )
            
            # Sort by risk-adjusted score
            top_protocols = protocols_df.nlargest(min(10, len(protocols_df)), 'risk_adjusted_score')
            
            print(f"\n🎯 TOP {min(10, len(protocols_df))} {segment_name.upper()} PROTOCOLS:")
            print("-" * 100)
            print(f"{'Rank':<4} {'Project':<15} {'Symbol':<15} {'Chain':<10} {'APY':<8} {'TVL (M)':<10} {'Volume (M)':<12} {'Score':<8}")
            print("-" * 100)
            
            for i, (_, protocol) in enumerate(top_protocols.iterrows()):
                project = protocol.get('project', 'Unknown')[:14]
                symbol = protocol.get('symbol', 'Unknown')[:14]
                chain = protocol.get('chain', 'Unknown')[:9]
                apy = protocol['apy']
                tvl_m = protocol['tvlUsd'] / 1e6
                volume_m = protocol['volumeUsd1d'] / 1e6
                score = protocol['risk_adjusted_score']
                
                print(f"{i+1:<4} {project:<15} {symbol:<15} {chain:<10} {apy:<8.2f}% ${tvl_m:<9.1f} ${volume_m:<11.1f} {score:<8.2f}")
            
            recommendations[segment_name] = {
                'top_protocols': top_protocols,
                'segment_summary': {
                    'recommended_allocation': 60 if segment_name == 'Stablecoin' else 40,
                    'risk_level': 'Low' if segment_name == 'Stablecoin' else 'High',
                    'expected_apy_range': '3-13%' if segment_name == 'Stablecoin' else '5-15%',
                    'min_investment': '$100,000' if segment_name == 'Stablecoin' else '$50,000'
                }
            }
    
    return recommendations

def create_portfolio_allocation_strategies(recommendations):
    """Create different portfolio allocation strategies"""
    
    print(f"\n📊 PORTFOLIO ALLOCATION STRATEGIES")
    print("="*80)
    
    strategies = {
        'Conservative': {
            'description': 'Low-risk, stable returns focus',
            'stablecoin_allocation': 80,
            'non_stablecoin_allocation': 20,
            'expected_return': '4-8%',
            'risk_level': 'Low',
            'min_investment': '$500,000'
        },
        'Balanced': {
            'description': 'Balanced risk-return profile',
            'stablecoin_allocation': 60,
            'non_stablecoin_allocation': 40,
            'expected_return': '6-11%',
            'risk_level': 'Medium',
            'min_investment': '$250,000'
        },
        'Aggressive': {
            'description': 'Higher risk, higher potential returns',
            'stablecoin_allocation': 30,
            'non_stablecoin_allocation': 70,
            'expected_return': '8-14%',
            'risk_level': 'High',
            'min_investment': '$100,000'
        }
    }
    
    for strategy_name, strategy in strategies.items():
        print(f"\n🎯 {strategy_name.upper()} STRATEGY:")
        print(f"  Description: {strategy['description']}")
        print(f"  Stablecoin allocation: {strategy['stablecoin_allocation']}%")
        print(f"  Non-stablecoin allocation: {strategy['non_stablecoin_allocation']}%")
        print(f"  Expected annual return: {strategy['expected_return']}")
        print(f"  Risk level: {strategy['risk_level']}")
        print(f"  Minimum investment: {strategy['min_investment']}")
    
    return strategies

def calculate_portfolio_metrics(recommendations, investment_amount=1000000):
    """Calculate detailed portfolio metrics for each strategy"""
    
    print(f"\n📈 PORTFOLIO PERFORMANCE PROJECTIONS")
    print(f"    (Based on ${investment_amount:,.0f} total investment)")
    print("="*80)
    
    if 'Stablecoin' in recommendations and 'Non-Stablecoin' in recommendations:
        stablecoin_protocols = recommendations['Stablecoin']['top_protocols'].head(5)
        non_stablecoin_protocols = recommendations['Non-Stablecoin']['top_protocols'].head(5)
        
        strategies = {
            'Conservative': {'stable': 0.8, 'non_stable': 0.2},
            'Balanced': {'stable': 0.6, 'non_stable': 0.4},
            'Aggressive': {'stable': 0.3, 'non_stable': 0.7}
        }
        
        for strategy_name, allocation in strategies.items():
            stable_investment = investment_amount * allocation['stable']
            non_stable_investment = investment_amount * allocation['non_stable']
            
            # Calculate weighted average APY
            stable_apy = stablecoin_protocols['apy'].mean()
            non_stable_apy = non_stablecoin_protocols['apy'].mean()
            
            weighted_apy = (stable_apy * allocation['stable'] + 
                           non_stable_apy * allocation['non_stable'])
            
            annual_return = investment_amount * (weighted_apy / 100)
            
            print(f"\n💼 {strategy_name.upper()} PORTFOLIO:")
            print(f"  Stablecoin investment: ${stable_investment:,.0f} ({allocation['stable']*100:.0f}%)")
            print(f"  Non-stablecoin investment: ${non_stable_investment:,.0f} ({allocation['non_stable']*100:.0f}%)")
            print(f"  Expected weighted APY: {weighted_apy:.2f}%")
            print(f"  Projected annual return: ${annual_return:,.0f}")
            print(f"  Monthly return: ${annual_return/12:,.0f}")

def create_portfolio_visualizations(recommendations, segment_analysis):
    """Create comprehensive portfolio visualization dashboard"""
    
    print(f"\n📊 Creating portfolio analysis visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('../results/plots', exist_ok=True)
    
    # Figure 1: Portfolio Allocation and Performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DeFi Portfolio: Stablecoin vs Non-Stablecoin Analysis', fontsize=16, fontweight='bold')
    
    # Segment distribution pie chart
    if len(segment_analysis) >= 2:
        segment_names = list(segment_analysis.keys())
        segment_counts = [segment_analysis[name]['count'] for name in segment_names]
        segment_tvl = [segment_analysis[name]['total_tvl'] for name in segment_names]
        
        axes[0,0].pie(segment_counts, labels=segment_names, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Protocol Distribution by Type')
        
        # TVL distribution
        axes[0,1].pie(segment_tvl, labels=segment_names, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('TVL Distribution by Type')
    
    # APY comparison
    if 'Stablecoin' in segment_analysis and 'Non-Stablecoin' in segment_analysis:
        stable_apys = segment_analysis['Stablecoin']['protocols']['apy']
        non_stable_apys = segment_analysis['Non-Stablecoin']['protocols']['apy']
        
        axes[1,0].hist([stable_apys, non_stable_apys], bins=10, alpha=0.7, 
                      label=['Stablecoin', 'Non-Stablecoin'], color=['blue', 'orange'])
        axes[1,0].set_xlabel('APY (%)')
        axes[1,0].set_ylabel('Number of Protocols')
        axes[1,0].set_title('APY Distribution Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Risk-Return scatter
        all_protocols = []
        colors = []
        for segment_name, analysis in segment_analysis.items():
            protocols = analysis['protocols']
            all_protocols.extend(protocols[['apy', 'tvlUsd']].values.tolist())
            colors.extend(['blue' if segment_name == 'Stablecoin' else 'orange'] * len(protocols))
        
        if all_protocols:
            all_protocols = np.array(all_protocols)
            scatter = axes[1,1].scatter(all_protocols[:, 1], all_protocols[:, 0], 
                                      c=colors, alpha=0.7, s=50)
            axes[1,1].set_xscale('log')
            axes[1,1].set_xlabel('TVL (USD, log scale)')
            axes[1,1].set_ylabel('APY (%)')
            axes[1,1].set_title('Risk-Return Profile')
            axes[1,1].grid(True, alpha=0.3)
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', label='Stablecoin'),
                             Patch(facecolor='orange', label='Non-Stablecoin')]
            axes[1,1].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('../results/plots/portfolio_allocation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Top Protocols Dashboard
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Top Recommended Protocols by Category', fontsize=16, fontweight='bold')
    
    y_pos = 0
    colors = ['skyblue', 'lightcoral']
    
    for i, (segment_name, rec) in enumerate(recommendations.items()):
        top_protocols = rec['top_protocols'].head(8)
        
        protocol_names = [f"{row.get('project', 'Unknown')[:10]}-{row.get('symbol', '')[:8]}" 
                         for _, row in top_protocols.iterrows()]
        apy_values = top_protocols['apy'].values
        
        y_positions = range(y_pos, y_pos + len(protocol_names))
        
        axes[i].barh(y_positions, apy_values, color=colors[i], alpha=0.8)
        axes[i].set_yticks(y_positions)
        axes[i].set_yticklabels(protocol_names)
        axes[i].set_xlabel('APY (%)')
        axes[i].set_title(f'Top {segment_name} Protocols by APY')
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels
        for j, (y_pos_val, apy) in enumerate(zip(y_positions, apy_values)):
            axes[i].text(apy + 0.1, y_pos_val, f'{apy:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/plots/top_protocols_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Portfolio visualizations saved to ../results/plots/")

def generate_final_recommendations(recommendations, segment_analysis):
    """Generate final actionable portfolio recommendations"""
    
    print(f"\n🎯 FINAL PORTFOLIO RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n📋 EXECUTIVE SUMMARY:")
    
    if 'Stablecoin' in segment_analysis and 'Non-Stablecoin' in segment_analysis:
        stable_count = segment_analysis['Stablecoin']['count']
        stable_avg_apy = segment_analysis['Stablecoin']['avg_apy']
        non_stable_count = segment_analysis['Non-Stablecoin']['count']
        non_stable_avg_apy = segment_analysis['Non-Stablecoin']['avg_apy']
        
        print(f"  • Analyzed {stable_count + non_stable_count} high-quality DeFi protocols")
        print(f"  • Stablecoin protocols: {stable_count} (avg APY: {stable_avg_apy:.1f}%)")
        print(f"  • Non-stablecoin protocols: {non_stable_count} (avg APY: {non_stable_avg_apy:.1f}%)")
        print(f"  • All protocols meet institutional criteria: TVL ≥ $10M, Volume ≥ $500K")
    
    print(f"\n🏆 RECOMMENDED PORTFOLIO ALLOCATION:")
    
    # Balanced strategy recommendation
    print(f"\n💼 BALANCED STRATEGY (RECOMMENDED):")
    print(f"  Allocation: 60% Stablecoin + 40% Non-Stablecoin")
    print(f"  Risk Level: Medium")
    print(f"  Expected APY: 6-11%")
    print(f"  Minimum Investment: $250,000")
    
    if 'Stablecoin' in recommendations:
        stable_top = recommendations['Stablecoin']['top_protocols'].head(3)
        print(f"\n  📈 TOP 3 STABLECOIN PROTOCOLS (60% allocation):")
        for i, (_, protocol) in enumerate(stable_top.iterrows()):
            project = protocol.get('project', 'Unknown')
            symbol = protocol.get('symbol', 'Unknown')
            apy = protocol['apy']
            tvl_m = protocol['tvlUsd'] / 1e6
            print(f"    {i+1}. {project} ({symbol}) - {apy:.2f}% APY, ${tvl_m:.0f}M TVL")
    
    if 'Non-Stablecoin' in recommendations:
        non_stable_top = recommendations['Non-Stablecoin']['top_protocols'].head(3)
        print(f"\n  🚀 TOP 3 NON-STABLECOIN PROTOCOLS (40% allocation):")
        for i, (_, protocol) in enumerate(non_stable_top.iterrows()):
            project = protocol.get('project', 'Unknown')
            symbol = protocol.get('symbol', 'Unknown')
            apy = protocol['apy']
            tvl_m = protocol['tvlUsd'] / 1e6
            print(f"    {i+1}. {project} ({symbol}) - {apy:.2f}% APY, ${tvl_m:.0f}M TVL")
    
    print(f"\n⚠️ RISK CONSIDERATIONS:")
    print(f"  • Stablecoin protocols: Lower risk, stable returns, less volatility")
    print(f"  • Non-stablecoin protocols: Higher risk, potential for higher returns")
    print(f"  • Diversification across chains reduces single-point-of-failure risk")
    print(f"  • Regular rebalancing recommended (monthly)")
    print(f"  • Monitor TVL and volume metrics for liquidity changes")
    
    print(f"\n📊 ALTERNATIVE STRATEGIES:")
    print(f"  • Conservative (80/20): Lower risk, 4-8% expected return")
    print(f"  • Aggressive (30/70): Higher risk, 8-14% expected return")
    print(f"  • Custom allocation based on risk tolerance and investment goals")
    
    # Save recommendations to CSV
    if recommendations:
        all_recommendations = []
        for segment_name, rec in recommendations.items():
            top_protocols = rec['top_protocols'].head(5)
            for _, protocol in top_protocols.iterrows():
                all_recommendations.append({
                    'Segment': segment_name,
                    'Project': protocol.get('project', 'Unknown'),
                    'Symbol': protocol.get('symbol', 'Unknown'),
                    'Chain': protocol.get('chain', 'Unknown'),
                    'APY_Percent': protocol['apy'],
                    'TVL_USD': protocol['tvlUsd'],
                    'Volume_USD': protocol['volumeUsd1d'],
                    'Risk_Level': 'Low' if segment_name == 'Stablecoin' else 'High',
                    'Recommended_Allocation': '60%' if segment_name == 'Stablecoin' else '40%'
                })
        
        recommendations_df = pd.DataFrame(all_recommendations)
        recommendations_df.to_csv('../results/reports/portfolio_recommendations.csv', index=False)
        print(f"\n💾 Detailed recommendations saved to: ../results/reports/portfolio_recommendations.csv")

def main_portfolio_analysis():
    """Main function for comprehensive portfolio analysis"""
    
    print("="*80)
    print("💼 COMPREHENSIVE PORTFOLIO ANALYSIS")
    print("   Stablecoin vs Non-Stablecoin DeFi Investment Strategy")
    print("="*80)
    
    # Load and process data
    data_path = '../data/sample_defi_data_small.json'
    
    print(f"\n📊 Step 1: Loading and processing DeFi data...")
    df_raw = load_data(data_path)
    if df_raw is None:
        print("Failed to load data")
        return None
    
    # Apply preprocessing
    result = improved_preprocess_pipeline(
        df_raw, target='apy', min_apy=1, max_apy=150, 
        min_tvl=10000000, min_volume=500000
    )
    
    if result[0] is None:
        print("No data remaining after preprocessing")
        return None
    
    X, y, numeric_features, categorical_features, binary_features, processed_df = result
    
    print(f"\n🔧 Step 2: Analyzing portfolio segments...")
    segment_analysis = analyze_portfolio_segments(processed_df)
    
    print(f"\n🏆 Step 3: Generating top protocol recommendations...")
    recommendations = generate_top_protocols_by_segment(segment_analysis)
    
    print(f"\n📊 Step 4: Creating allocation strategies...")
    strategies = create_portfolio_allocation_strategies(recommendations)
    
    print(f"\n📈 Step 5: Calculating portfolio metrics...")
    calculate_portfolio_metrics(recommendations, investment_amount=1000000)
    
    print(f"\n📊 Step 6: Creating visualizations...")
    create_portfolio_visualizations(recommendations, segment_analysis)
    
    print(f"\n🎯 Step 7: Generating final recommendations...")
    generate_final_recommendations(recommendations, segment_analysis)
    
    print(f"\n✅ PORTFOLIO ANALYSIS COMPLETED!")
    print(f"📁 Generated files:")
    print(f"   • ../results/plots/portfolio_allocation_analysis.png")
    print(f"   • ../results/plots/top_protocols_dashboard.png")
    print(f"   • ../results/reports/portfolio_recommendations.csv")
    
    return recommendations, segment_analysis, strategies

if __name__ == "__main__":
    results = main_portfolio_analysis() 
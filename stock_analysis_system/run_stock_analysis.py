#!/usr/bin/env python3
"""
Stock Analysis System - Unified Launcher

This script orchestrates the complete stock price prediction and portfolio analysis pipeline:
1. Tree-based machine learning models (XGBoost, LightGBM, CatBoost, Random Forest)
2. Sector-based portfolio segmentation
3. Investment recommendations and allocation strategies
4. Comprehensive visualization and reporting

Directory Structure:
├── core/                           # Core analysis modules
│   ├── financial_models_analysis.py
│   ├── portfolio_recommendations.py
│   └── data_processing_improved.py
├── results/                       # All outputs organized here
│   ├── plots/                     # Visualization outputs
│   ├── models/                    # Trained model artifacts
│   └── reports/                   # CSV reports and summaries
└── archive/                       # Backup and legacy files

Usage:
    python run_stock_analysis.py [options]
    
Options:
    --full-analysis     Run complete analysis (tree models + portfolio)
    --tree-models-only  Run only tree-based model analysis
    --portfolio-only    Run only portfolio analysis
    --investment-amount Amount to analyze (default: 1000000)
"""

import sys
import os
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add core directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def print_banner():
    """Print system banner"""
    print("="*100)
    print("🚀 STOCK ANALYSIS SYSTEM - COMPREHENSIVE PRICE PREDICTION & PORTFOLIO OPTIMIZATION")
    print("="*100)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objective: Tree-based ML models + Sector analysis + Investment strategy")
    print(f"📁 Results Directory: results/ (plots, models, reports)")
    print("="*100)

def check_dependencies():
    """Check if all required packages are available"""
    print("\n🔍 CHECKING SYSTEM DEPENDENCIES...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'
    ]
    
    optional_packages = [
        'xgboost', 'lightgbm', 'catboost'
    ]
    
    missing_packages = []
    available_packages = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            available_packages.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - NOT FOUND")
    
    # Check optional packages
    optional_available = []
    for package in optional_packages:
        try:
            __import__(package)
            optional_available.append(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️ {package} (optional) - not available")
    
    if missing_packages:
        print(f"\n⚠️ WARNING: Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print(f"\n✅ ALL REQUIRED DEPENDENCIES SATISFIED ({len(available_packages)}/{len(required_packages)})")
    print(f"✅ OPTIONAL ML LIBRARIES AVAILABLE: {len(optional_available)}/{len(optional_packages)}")
    return True

def check_data_availability():
    """Check if required data files exist"""
    print("\n📊 CHECKING DATA AVAILABILITY...")
    
    # Check for stock data files
    data_paths = [
        os.path.join("..", "hse-portfolio-stocks", "data", "raw", "profile.parquet"),
        os.path.join("..", "hse-portfolio-stocks", "data", "raw", "prices.parquet"),
        os.path.join("..", "hse-portfolio-stocks", "data", "processed", "master.parquet")
    ]
    
    missing_files = []
    available_files = []
    
    for path in data_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            available_files.append(path)
            print(f"  ✅ {path} ({file_size:.1f} MB)")
        else:
            missing_files.append(path)
            print(f"  ❌ {path} - NOT FOUND")
    
    if missing_files:
        print(f"\n⚠️ WARNING: Missing data files: {missing_files}")
        print("Please ensure the stock data files are in the hse-portfolio-stocks directory")
        return False
    
    print(f"\n✅ ALL DATA FILES AVAILABLE ({len(available_files)}/{len(data_paths)})")
    return True

def create_results_structure():
    """Ensure results directory structure exists"""
    print("\n📁 SETTING UP RESULTS DIRECTORY STRUCTURE...")
    
    directories = [
        "results",
        "results/plots",
        "results/models", 
        "results/reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ Results structure ready")

def run_tree_models_analysis(quick_mode=False):
    """Execute tree-based models analysis"""
    print("\n" + "="*80)
    if quick_mode:
        print("🌳 PHASE 1: TREE-BASED MACHINE LEARNING ANALYSIS (QUICK MODE)")
    else:
        print("🌳 PHASE 1: TREE-BASED MACHINE LEARNING ANALYSIS")
    print("="*80)
    
    try:
        # Change to core directory to run the analysis
        original_dir = os.getcwd()
        os.chdir('core')
        
        # Import and run tree models analysis
        from financial_models_analysis import main_tree_analysis
        
        start_time = time.time()
        results, models, processed_df = main_tree_analysis(quick_mode=quick_mode)
        execution_time = time.time() - start_time
        
        print(f"\n✅ Tree models analysis completed in {execution_time:.1f} seconds")
        
        # Return to original directory
        os.chdir(original_dir)
        
        return results, models, processed_df
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"❌ Error in tree models analysis: {e}")
        return None, None, None

def run_portfolio_analysis(investment_amount=1000000):
    """Execute portfolio analysis and recommendations"""
    print("\n" + "="*80)
    print("💼 PHASE 2: PORTFOLIO ANALYSIS & INVESTMENT RECOMMENDATIONS")
    print("="*80)
    
    try:
        # Change to core directory to run the analysis
        original_dir = os.getcwd()
        os.chdir('core')
        
        # Import and run portfolio analysis
        from portfolio_recommendations import main_portfolio_analysis
        
        start_time = time.time()
        recommendations, sector_analysis, strategies = main_portfolio_analysis()
        execution_time = time.time() - start_time
        
        print(f"\n✅ Portfolio analysis completed in {execution_time:.1f} seconds")
        
        # Return to original directory
        os.chdir(original_dir)
        
        return recommendations, sector_analysis, strategies
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"❌ Error in portfolio analysis: {e}")
        return None, None, None

def generate_executive_summary(tree_results, portfolio_results):
    """Generate executive summary of all analyses"""
    print("\n" + "="*80)
    print("📋 EXECUTIVE SUMMARY - COMPREHENSIVE STOCK ANALYSIS RESULTS")
    print("="*80)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("📊 STOCK ANALYSIS SYSTEM - EXECUTIVE SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Tree Models Summary
    if tree_results and len(tree_results) >= 3 and tree_results[0] is not None:
        try:
            # tree_results is a tuple (results_list, models, df)
            # results_list is a list of tuples (model_name, model_result)
            results_list, models, df = tree_results
            
            if results_list and len(results_list) > 0:
                # The list is already sorted by RMSE, so the first entry is the best
                best_model_name, best_result = results_list[0]
                best_rmse = best_result['rmse_cv']
                best_directional = best_result['directional_accuracy']
                
                summary_lines.append("🤖 MACHINE LEARNING MODELS PERFORMANCE:")
                summary_lines.append(f"  Best Model: {best_model_name.replace('_', ' ')}")
                summary_lines.append(f"  Prediction RMSE: {best_rmse:.6f}")
                summary_lines.append(f"  Directional Accuracy: {best_directional:.1%}")
                summary_lines.append(f"  Model Quality: {'Excellent' if best_directional > 0.55 else 'Good' if best_directional > 0.52 else 'Moderate'}")
                summary_lines.append("")
            else:
                summary_lines.append("🤖 MACHINE LEARNING MODELS PERFORMANCE:")
                summary_lines.append("  ⚠️ No successful model results available")
                summary_lines.append("")
        except Exception as e:
            print(f"  ⚠️ Error processing tree results: {e}")
            summary_lines.append("🤖 MACHINE LEARNING MODELS PERFORMANCE:")
            summary_lines.append("  ⚠️ Error processing model results")
            summary_lines.append("")
    
    # Portfolio Summary
    if portfolio_results and len(portfolio_results) >= 2 and portfolio_results[1]:
        try:
            sector_analysis = portfolio_results[1]
            if sector_analysis and len(sector_analysis) > 0:
                total_sectors = len(sector_analysis)
                
                # Get top performing sector
                top_sector = max(sector_analysis.items(), key=lambda x: x[1]['avg_sharpe'])
                
                summary_lines.append("📈 STOCK PORTFOLIO RECOMMENDATIONS:")
                summary_lines.append(f"  Analyzed Sectors: {total_sectors} market sectors")
                summary_lines.append(f"  Top Performing Sector: {top_sector[0]} (Sharpe: {top_sector[1]['avg_sharpe']:.3f})")
                summary_lines.append(f"  Investment Strategies: 3 (Conservative/Balanced/Aggressive)")
                summary_lines.append("")
            else:
                summary_lines.append("📈 STOCK PORTFOLIO RECOMMENDATIONS:")
                summary_lines.append("  ⚠️ No sector analysis results available")
                summary_lines.append("")
        except Exception as e:
            print(f"  ⚠️ Error processing portfolio results: {e}")
            summary_lines.append("📈 STOCK PORTFOLIO RECOMMENDATIONS:")
            summary_lines.append("  ⚠️ Error processing portfolio results")
            summary_lines.append("")
    
    # Strategy Summary
    if portfolio_results and len(portfolio_results) >= 3 and portfolio_results[2]:
        try:
            strategies = portfolio_results[2]
            if strategies and len(strategies) > 0:
                summary_lines.append("💼 INVESTMENT STRATEGY PROJECTIONS (Annual):")
                for strategy_name, strategy in strategies.items():
                    annual_return = strategy['expected_return'] * 252 * 100
                    annual_vol = strategy['expected_volatility'] * np.sqrt(252) * 100
                    summary_lines.append(f"  {strategy_name} Strategy: {annual_return:.1f}% return, {annual_vol:.1f}% volatility")
                summary_lines.append("")
            else:
                summary_lines.append("💼 INVESTMENT STRATEGY PROJECTIONS:")
                summary_lines.append("  ⚠️ No strategy results available")
                summary_lines.append("")
        except Exception as e:
            print(f"  ⚠️ Error processing strategy results: {e}")
            summary_lines.append("💼 INVESTMENT STRATEGY PROJECTIONS:")
            summary_lines.append("  ⚠️ Error processing strategy results")
            summary_lines.append("")
    
    # Generated Files
    summary_lines.append("📁 GENERATED DELIVERABLES:")
    summary_lines.append("  📊 Visualizations:")
    summary_lines.append("    • results/plots/tree_models_raw_data_overview.png")
    summary_lines.append("    • results/plots/tree_models_performance_comparison.png")
    summary_lines.append("    • results/plots/tree_models_feature_importance.png")
    summary_lines.append("    • results/plots/portfolio_allocation_analysis.png")
    summary_lines.append("    • results/plots/top_protocols_dashboard.png")
    summary_lines.append("")
    summary_lines.append("  📈 Reports:")
    summary_lines.append("    • results/reports/tree_models_comparison.csv")
    summary_lines.append("    • results/reports/tree_models_feature_importance.csv")
    summary_lines.append("    • results/reports/portfolio_recommendations.csv")
    summary_lines.append("    • results/reports/strategy_summary.csv")
    summary_lines.append("")
    summary_lines.append("⚠️ NEXT STEPS:")
    summary_lines.append("  1. Review top stock recommendations in portfolio_recommendations.csv")
    summary_lines.append("  2. Choose investment strategy based on risk tolerance")
    summary_lines.append("  3. Implement sector diversification as recommended")
    summary_lines.append("  4. Monitor performance and rebalance quarterly")
    summary_lines.append("="*80)
    
    # Print summary
    for line in summary_lines:
        print(line)
    
    # Save summary to file
    with open('results/reports/executive_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n💾 Executive summary saved to: results/reports/executive_summary.txt")

def main():
    """Main function to orchestrate the entire analysis"""
    parser = argparse.ArgumentParser(description='Stock Analysis System')
    parser.add_argument('--full-analysis', action='store_true', 
                       help='Run complete analysis (default)')
    parser.add_argument('--tree-models-only', action='store_true',
                       help='Run only tree-based model analysis')
    parser.add_argument('--portfolio-only', action='store_true',
                       help='Run only portfolio analysis')
    parser.add_argument('--quick-mode', action='store_true',
                       help='Run in quick mode (faster, reduced accuracy)')
    parser.add_argument('--investment-amount', type=float, default=1000000,
                       help='Investment amount to analyze (default: 1000000)')
    
    args = parser.parse_args()
    
    # Default to full analysis if no specific option chosen
    if not any([args.tree_models_only, args.portfolio_only]):
        args.full_analysis = True
    
    # Print banner
    print_banner()
    
    if args.quick_mode:
        print("⚡ QUICK MODE ENABLED - Optimized for speed over accuracy")
        print("  • Reduced model complexity")
        print("  • Smaller datasets for training")
        print("  • GPU acceleration where available")
        print("  • Faster cross-validation")
    
    # Pre-flight checks
    if not check_dependencies():
        print("\n❌ System dependencies not satisfied. Please install missing packages.")
        return False
    
    if not check_data_availability():
        print("\n❌ Required data files not found. Please check data directory.")
        return False
    
    # Setup results structure
    create_results_structure()
    
    # Track overall execution time
    overall_start = time.time()
    
    # Initialize results
    tree_results = None
    portfolio_results = None
    
    # Execute analyses based on arguments
    if args.full_analysis or args.tree_models_only:
        tree_results = run_tree_models_analysis(quick_mode=args.quick_mode)
        if tree_results[0] is None:
            print("\n❌ Tree models analysis failed. Stopping execution.")
            return False
    
    if args.full_analysis or args.portfolio_only:
        portfolio_results = run_portfolio_analysis(args.investment_amount)
        if portfolio_results[0] is None:
            print("\n❌ Portfolio analysis failed. Stopping execution.")
            return False
    
    # Generate executive summary
    if args.full_analysis:
        # Import numpy for summary generation
        import numpy as np
        generate_executive_summary(tree_results, portfolio_results)
    
    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    if args.quick_mode:
        print("🎉 QUICK STOCK ANALYSIS COMPLETE!")
    else:
        print("🎉 STOCK ANALYSIS COMPLETE!")
    print("="*80)
    print(f"⏱️ Total Execution Time: {total_time:.1f} seconds")
    print(f"📁 All results saved to: results/ directory")
    print(f"📊 Ready for investment decision making!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
# 🚀 Stock Analysis System
## Comprehensive Tree-Based Price Prediction & Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-purple.svg)](https://catboost.ai/)

**Advanced machine learning system for stock price prediction and sector-based portfolio allocation using tree-based ensemble models.**

---

## 🎯 **System Overview**

This system implements a comprehensive stock analysis pipeline that:

1. **🌳 Tree-Based ML Models**: XGBoost, LightGBM, CatBoost, Random Forest for price prediction
2. **📈 Technical Analysis**: RSI, SMA, momentum, volatility features
3. **🏢 Sector Segmentation**: Technology, Healthcare, Finance, etc. classification
4. **💼 Investment Strategies**: Conservative/Balanced/Aggressive allocation frameworks
5. **📊 Performance Analysis**: Risk-adjusted metrics with directional accuracy
6. **📁 Organized Results**: Professional reporting with executive summaries

### **Key Performance Metrics**
- **Prediction Target**: Next-day stock returns
- **Data Coverage**: 500+ stocks across 11 major sectors
- **Time Period**: 2015-2025 historical data (10+ years)
- **Features**: 100+ technical and fundamental indicators
- **Directional Accuracy**: Target >55% for trading signals

---

## 📁 **Project Structure**

```
stock_analysis_system/
├── 🚀 run_stock_analysis.py        # UNIFIED LAUNCHER - Start here!
├── 📖 README.md                    # This documentation
├── 🔧 core/                        # Core analysis modules
│   ├── financial_models_analysis.py # Tree-based ML models
│   ├── portfolio_recommendations.py # Portfolio analysis
│   └── data_processing_improved.py  # Data preprocessing
├── 📈 results/                     # ALL OUTPUTS GO HERE
│   ├── plots/                      # Visualization outputs
│   ├── models/                     # Trained model artifacts  
│   └── reports/                    # CSV reports & summaries
├── 📦 archive/                     # Backup and legacy files
└── 📋 requirements.txt             # Python dependencies
```

---

## ⚡ **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Complete Analysis**
```bash
python run_stock_analysis.py
```

### **3. View Results**
Results are automatically organized in the `results/` directory:
- **📊 Visualizations**: `results/plots/`
- **📈 Reports**: `results/reports/`
- **🤖 Models**: `results/models/`

---

## 🎮 **Usage Options**

### **Full Analysis (Recommended)**
```bash
python run_stock_analysis.py --full-analysis
```
*Runs complete pipeline: Tree models + Portfolio analysis + Executive summary*

### **Quick Mode (Fast Development)**
```bash
python run_stock_analysis.py --quick-mode
```
*Optimized for speed: Reduced model complexity, GPU acceleration, faster training*

### **Tree Models Only**
```bash
python run_stock_analysis.py --tree-models-only
```
*Focuses on machine learning model training and stock price prediction*

### **Tree Models Only (Quick)**
```bash
python run_stock_analysis.py --tree-models-only --quick-mode
```
*Fast model training with reduced complexity for development/testing*

### **Portfolio Analysis Only**
```bash
python run_stock_analysis.py --portfolio-only
```
*Focuses on investment recommendations and sector allocation strategies*

### **Custom Investment Amount**
```bash
python run_stock_analysis.py --investment-amount 5000000
```
*Analyzes portfolio performance for $5M investment*

### **Speed Optimization Options**
- **`--quick-mode`**: Faster training with reduced accuracy
  - Random Forest: 20-100 trees (vs 200)
  - XGBoost/LightGBM: 50-200 trees (vs 200)  
  - GPU acceleration when available
  - 3-fold CV (vs 5-fold)
  - Data sampling for large datasets
  - Skip heavy visualizations

---

## �� **Generated Outputs**

### **📈 Visualizations** (`results/plots/`)
1. **`tree_models_raw_data_overview.png`** - Stock market data exploration
2. **`tree_models_performance_comparison.png`** - Model accuracy comparison
3. **`tree_models_feature_importance.png`** - Feature importance analysis
4. **`portfolio_allocation_analysis.png`** - Sector allocation charts
5. **`top_protocols_dashboard.png`** - Top stock recommendations

### **📊 Reports** (`results/reports/`)
1. **`tree_models_comparison.csv`** - Model performance metrics
2. **`tree_models_feature_importance.csv`** - Feature rankings
3. **`portfolio_recommendations.csv`** - **🏆 MAIN INVESTMENT GUIDE**
4. **`strategy_summary.csv`** - Investment strategy comparison
5. **`executive_summary.txt`** - Complete analysis summary

### **🤖 Models** (`results/models/`)
- Trained XGBoost, LightGBM, CatBoost, Random Forest models (`.pkl` files)

---

## 🏆 **Investment Recommendations**

### **💼 Investment Strategies**

#### **🛡️ Conservative Strategy**
- **Target Sectors**: Utilities, Consumer Staples, Real Estate
- **Risk Level**: Low (volatility < 15%)
- **Expected Return**: 8-12% annually
- **Portfolio Size**: 10-15 stocks

#### **⚖️ Balanced Strategy (Recommended)**
- **Target Sectors**: Technology, Healthcare, Financials, Industrials
- **Risk Level**: Medium (volatility 15-25%)
- **Expected Return**: 10-16% annually
- **Portfolio Size**: 15-20 stocks

#### **🚀 Aggressive Strategy**
- **Target Sectors**: Technology, Consumer Discretionary, Communication Services
- **Risk Level**: High (volatility 25-40%)
- **Expected Return**: 12-20% annually
- **Portfolio Size**: 12-18 stocks

### **🎯 Top Sectors by Performance**
1. **Technology** - High growth, innovation-driven
2. **Healthcare** - Defensive with growth potential
3. **Financials** - Economic cycle leverage
4. **Consumer Discretionary** - Economic strength indicator
5. **Industrials** - Infrastructure and manufacturing

---

## 🔧 **Technical Implementation**

### **🌳 Machine Learning Models**
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Microsoft's high-performance gradient boosting
- **CatBoost**: Yandex's categorical feature optimization
- **Random Forest**: Scikit-learn ensemble with bagging

### **📊 Data Processing Pipeline**
1. **Raw Data Loading**: Stock prices, returns, sector/industry metadata
2. **Feature Engineering**: Technical indicators (RSI, SMA, momentum)
3. **Target Creation**: Forward-looking returns (1-day, 5-day, 20-day)
4. **Model Training**: Time series cross-validation
5. **Portfolio Analysis**: Risk-adjusted scoring and sector allocation

### **🎯 Key Features**
- **Technical Indicators**: RSI, SMA, price relatives, volatility measures
- **Sector Features**: Dummy variables and sector-level aggregations
- **Time Series Validation**: Proper temporal splitting for financial data
- **Risk Metrics**: Sharpe ratio, maximum drawdown, downside volatility
- **Directional Accuracy**: Critical for trading signal quality

---

## 📋 **System Requirements**

### **Python Packages**
```
pandas >= 1.3.0          # Data manipulation
numpy >= 1.21.0          # Numerical computing
matplotlib >= 3.4.0      # Plotting
seaborn >= 0.11.0        # Statistical visualization
scikit-learn >= 1.0.0    # Machine learning
xgboost >= 1.5.0         # Gradient boosting
lightgbm >= 3.3.0        # LightGBM
catboost >= 1.0.0        # CatBoost
pyarrow >= 5.0.0         # Parquet support
```

### **Hardware Recommendations**
- **RAM**: 8GB+ (for full dataset processing)
- **CPU**: Multi-core processor (for ensemble training)
- **Storage**: 2GB free space (for models and results)

### **Data Requirements**
- **Stock Data**: Located in `../hse-portfolio-stocks/data/`
- **Format**: Parquet files (profile, prices, master)
- **Coverage**: 500+ stocks, 2015-2025 period

---

## 🚀 **Advanced Usage**

### **Custom Feature Engineering**
Modify technical indicators in `core/data_processing_improved.py`:
```python
# Add custom technical indicators
symbol_data['custom_indicator'] = calculate_custom_indicator(symbol_data['price'])
symbol_data['sector_momentum'] = calculate_sector_momentum(symbol_data, sector_data)
```

### **Model Customization**
Adjust hyperparameters in `core/financial_models_analysis.py`:
```python
models['XGBoost'] = xgb.XGBRegressor(
    n_estimators=300,        # Increase for better accuracy
    max_depth=8,             # Adjust complexity
    learning_rate=0.05,      # Fine-tune learning
    subsample=0.8,           # Prevent overfitting
)
```

### **Strategy Customization**
Modify investment strategies in `core/portfolio_recommendations.py`:
```python
strategies = {
    'Ultra_Conservative': {
        'target_sectors': ['Utilities', 'Consumer Staples'],
        'risk_threshold': 0.10,
        'min_sharpe': 0.5
    },
    'Growth_Focused': {
        'target_sectors': ['Technology', 'Biotech'],
        'risk_threshold': 0.50,
        'min_sharpe': 0.0
    }
}
```

---

## ⚠️ **Risk Considerations**

### **Model Limitations**
- **Market Regime Changes**: Models trained on historical data may not capture regime shifts
- **Volatility Clustering**: Financial markets exhibit periods of high/low volatility
- **Behavioral Factors**: Models may not capture sentiment and behavioral patterns

### **Investment Risks**
- **Market Risk**: Systematic risk affecting entire market
- **Sector Risk**: Concentration in specific sectors
- **Liquidity Risk**: Ability to exit positions quickly
- **Model Risk**: Prediction errors and overfitting

### **Recommended Practices**
1. **Diversification**: Spread investments across sectors and strategies
2. **Position Sizing**: Limit individual stock positions to 5-10%
3. **Regular Rebalancing**: Quarterly portfolio review and adjustment
4. **Risk Management**: Set stop-losses and maximum drawdown limits
5. **Performance Monitoring**: Track actual vs predicted performance

---

## 📞 **Support & Troubleshooting**

### **Common Issues**

#### **Data Loading Errors**
```bash
# Check data file paths
ls ../hse-portfolio-stocks/data/raw/
ls ../hse-portfolio-stocks/data/processed/
```

#### **Memory Issues**
```python
# Reduce data size in data_processing_improved.py
# Sample fewer stocks or shorter time periods
sample_stocks = df['symbol'].unique()[:100]  # Use first 100 stocks
```

#### **Model Training Errors**
```bash
# Install optional ML libraries
pip install xgboost lightgbm catboost
```

### **Performance Optimization**
- **Parallel Processing**: Set `n_jobs=-1` in model parameters
- **Data Sampling**: Use representative subsets for faster development
- **Feature Selection**: Remove low-importance features

---

## 🎯 **Next Steps**

### **For Immediate Use**
1. ✅ Run the complete analysis: `python run_stock_analysis.py`
2. ✅ Review investment recommendations in `results/reports/portfolio_recommendations.csv`
3. ✅ Choose strategy based on risk tolerance
4. ✅ Implement sector diversification as recommended

### **For Production Deployment**
1. 🔄 Set up automated data feeds (Yahoo Finance, Alpha Vantage)
2. 🔄 Implement real-time prediction pipeline
3. 🔄 Add automated portfolio rebalancing
4. 🔄 Create web dashboard for monitoring

### **For Research Extension**
1. 📚 Add alternative data sources (news, social media)
2. 📚 Implement ensemble of multiple timeframes
3. 📚 Add options and derivatives modeling
4. 📚 Develop custom risk models

---

## 📊 **Performance Benchmarks**

| Metric | Target | Status |
|--------|--------|--------|
| Directional Accuracy | >55% | 🟡 In Progress |
| Sharpe Ratio | >1.0 | 🟢 Achievable |
| Max Drawdown | <20% | 🟢 Controlled |
| Processing Time | <5 minutes | 🟢 Fast |
| Memory Usage | <4GB | 🟢 Efficient |

---

## 🔄 **Version History**

- **v1.0** - Initial release with basic tree models
- **v1.1** - Added sector analysis and portfolio optimization
- **v1.2** - Enhanced feature engineering and risk metrics
- **Current** - Comprehensive system with multiple strategies

---

**🎉 Ready to transform your stock investment strategy with institutional-grade analysis!**

*For questions, issues, or contributions, please review the generated reports and executive summary in the `results/` directory.* 
# Cryptocurrency Volatility Prediction

A comprehensive machine learning project for predicting cryptocurrency market volatility using advanced ensemble methods and feature engineering techniques.

## 📊 Project Overview

This project develops and evaluates machine learning models to predict cryptocurrency volatility, achieving exceptional predictive performance with XGBoost explaining **60.7% of volatility variance**. The model provides valuable insights for risk management and trading applications in cryptocurrency markets.

## 🎯 Key Achievements

- **Model Performance**: XGBoost achieved R² = 0.607, explaining 60.7% of volatility variance
- **Feature Engineering**: Comprehensive preprocessing pipeline handling outliers and multicollinearity
- **Dimensionality Optimization**: PCA analysis confirming optimal feature reduction to 2-4 components
- **Model Robustness**: Minimal overfitting (0.10) indicating strong generalization capability

## 🚀 Features

- **Advanced Data Preprocessing**: Outlier detection, multicollinearity resolution, and feature scaling
- **Comprehensive EDA**: Univariate, bivariate, and multivariate analysis with visualizations
- **Feature Engineering**: Creation of volatility indicators, temporal features, and market metrics
- **Multiple ML Models**: Comparison of 8 different algorithms including linear and ensemble methods
- **Model Validation**: Rigorous performance evaluation with cross-validation and residual analysis
- **Production Ready**: Trained model saved for deployment with prediction pipeline

## 📈 Technical Highlights

### Data Processing
- **Dataset**: 72,946 daily records across 56+ cryptocurrencies (2013-2022)
- **Feature Reduction**: Eliminated multicollinearity (VIF analysis)
- **Missing Data**: Handled using advanced imputation techniques
- **Outlier Treatment**: IQR-based capping for robust modeling

### Model Performance
| Model | Test R² | Test RMSE | Test MAE | Overfitting |
|-------|---------|-----------|----------|-------------|
| **XGBoost** | **0.6073** | **0.7504** | **0.5623** | **0.1017** |
| Gradient Boosting | 0.5867 | 0.7699 | 0.5733 | 0.0026 |
| Random Forest | 0.5820 | 0.7743 | 0.5833 | 0.3592 |
| Linear Regression | 0.5491 | 0.8041 | 0.5919 | -0.0022 |

### Key Features for Prediction
1. **volatility_14d** (70.8% importance) - Historical 14-day volatility
2. **return_pct** (4.4% importance) - Daily percentage returns
3. **gk_volatility** (4.1% importance) - Garman-Klass volatility estimator
4. **quarter** (3.2% importance) - Seasonal patterns
5. **crypto_encoded** (2.8% importance) - Asset-specific characteristics

## 🛠️ Technologies Used

- **Python 3.10+**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: statsmodels
- **Model Validation**: cross-validation, hyperparameter tuning

## 📁 Project Structure

```
crypto-volatility-prediction/
├── data/
│   ├── raw/                    # Original dataset
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Final datasets for modeling
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── feature_selection_and_classification.ipynb
├── src/crypto_pred/
│   ├── dataset.py            # Data loading and preprocessing
│   ├── features.py           # Feature engineering functions
│   ├── modeling/
│   │   ├── train.py         # Model training pipeline
│   │   └── predict.py       # Prediction functions
│   └── plots.py             # Visualization utilities
├── models/                   # Trained model artifacts
├── reports/                  # Generated analysis reports
├── requirements.txt          # Dependencies
├── Makefile                 # Build automation
└── README.md
```

## 🔧 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/happii2k/crypto-volatility-prediction.git
cd crypto-volatility-prediction
```

2. **Create virtual environment**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install package in development mode**
```bash
pip install -e .
```

## 📊 Usage

### Data Analysis
```python
# Run exploratory data analysis
jupyter notebook notebooks/EDA.ipynb

# Feature engineering and model training
jupyter notebook notebooks/feature_selection_and_classification.ipynb
```

### Model Training
```python
from crypto_pred.modeling.train import train_volatility_model
from crypto_pred.features import create_features

# Load and preprocess data
df = load_data()
df_processed = create_features(df)

# Train model
model = train_volatility_model(df_processed)
```

### Making Predictions
```python
from crypto_pred.modeling.predict import predict_volatility

# Predict volatility for new data
predictions = predict_volatility(model, new_data)
```

## 🎯 Business Applications

### Risk Management
- **Dynamic Position Sizing**: Adjust exposure based on predicted volatility
- **Portfolio Optimization**: Optimize risk-return profiles using volatility forecasts
- **Hedging Strategies**: Implement proactive hedging based on volatility predictions

### Trading Strategies
- **Market Timing**: Optimize entry/exit timing using volatility predictions
- **Volatility Breakout**: Enhanced breakout strategies with forward-looking volatility
- **Options Trading**: Improved volatility forecasting for derivatives pricing

## 📈 Model Insights

### Volatility Drivers
- **Historical Volatility**: Strong persistence in cryptocurrency volatility patterns
- **Market Structure**: Volume and market cap significantly influence volatility
- **Temporal Patterns**: Systematic calendar effects in crypto markets
- **Asset-Specific Effects**: Different cryptocurrencies exhibit distinct volatility characteristics

### Performance Validation
- **Residual Analysis**: Randomly distributed residuals with no systematic bias
- **Distribution Alignment**: Strong agreement between predicted and actual volatility distributions
- **Cross-Validation**: Consistent performance across different time periods and market conditions

## 🔮 Future Enhancements

- **Deep Learning**: LSTM networks for temporal pattern recognition
- **Alternative Data**: Integration of sentiment analysis and network activity metrics
- **Real-time Deployment**: Streaming prediction pipeline for live trading
- **Multi-asset Models**: Transfer learning across different cryptocurrency markets
- **Advanced Features**: Technical indicators and macroeconomic variables

## 📊 Results & Visualizations

The project includes comprehensive visualizations:
- **Correlation heatmaps** for feature relationships
- **Actual vs Predicted** scatter plots with R² metrics
- **Residual analysis** plots for model validation
- **Feature importance** rankings
- **Time series** predictions with confidence intervals



---

⭐ **If you find this project useful, please consider giving it a star!**

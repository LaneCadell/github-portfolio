# Quantile Return Forecasting Pipeline

A sophisticated end-to-end machine learning system for predicting 3-month forward stock returns across multiple quantiles using LightGBM quantile regression, with integration of market-based probabilities from derivatives and prediction markets.

## Overview

This pipeline predicts the **20th, 50th (median), and 80th percentile** percentage returns for stocks with market cap > $20 billion over a 3-month (63 trading day) horizon, then reconstructs absolute price targets from these predictions.

### Why Predict Returns, Not Prices?

**Mathematical Advantages:**

1. **Stationarity**: Stock prices are non-stationary unit root processes (I(1)), violating standard regression assumptions. Returns (differences in log prices) are typically stationary (I(0)), enabling valid inference and more stable model convergence.

2. **Homoscedasticity**: Price volatility scales with price levels, creating heteroscedastic errors. Log-returns normalize this variance across price ranges, improving model efficiency.

3. **Scale Invariance**: Return-based models generalize across stocks of different absolute price levels. A $5 move on a $100 stock equals a $50 move on a $1,000 stock—but both represent 5% returns.

4. **Statistical Properties**: Returns exhibit better statistical properties for regression:
   - Reduced serial correlation vs. prices
   - Closer approximation to normality
   - Better suited for time-series modeling

5. **Risk Metrics**: Returns directly correspond to investor-relevant concepts:
   - Portfolio-level risk aggregation
   - Information Ratio and Sharpe Ratio calculations
   - Intuitive alpha measurement

## Architecture

### Modular Structure

```
Quantile-Return-Forecasting/
├── data_loader.py          # Data ingestion from APIs
├── features.py             # Feature engineering & preprocessing
├── model.py                # LightGBM quantile regression & backtesting
├── main.py                 # Orchestration, CLI, & Streamlit UI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Data Pipeline Flow

```
APIs (Alpha Vantage, FRED, Kalshi)
            ↓
    [DataLoader] → Raw OHLCV, macro, sentiment data
            ↓
[FeatureEngineer] → Technical indicators, macro lags, stationary features
            ↓
    [QuantileModel] → LightGBM quantile regression training
            ↓
[BacktestEngine] → Walk-forward validation & performance metrics
            ↓
   Price Forecasts (20th, 50th, 80th %ile)
```

## Module Documentation

### 1. **data_loader.py**

Integrates three financial data sources:

#### Alpha Vantage
- **Daily OHLCV Data**: `fetch_alpha_vantage_daily(ticker)` returns adjusted close prices and volume
- **Fundamentals**: `fetch_alpha_vantage_fundamentals(ticker)` retrieves market cap for universe filtering

#### FRED (Federal Reserve Economic Data)
- **Macro Series**: Treasury yields (DGS10), inflation (CPIAUCSL), GDP growth (A191RL1Q225SBEA)
- **Time Alignment**: Automatic forward-fill for misaligned frequencies

#### Kalshi (Prediction Markets)
- **Event Odds**: Interest rate cut probabilities, S&P 500 inclusion odds
- **Sentiment Feature**: Implied probabilities converted to 0-1 daily features

**Key Methods:**
- `filter_by_market_cap()`: Ensures universe contains only $20B+ stocks
- `align_dataframes()`: Handles multi-source frequency misalignment

---

### 2. **features.py**

Comprehensive feature engineering for quantile regression:

#### Target Variable
```python
R_{t → t+63} = (Price_{t+63} - Price_t) / Price_t × 100
```
- 63 trading days ≈ 3 months
- Expressed as percentage for interpretability

#### Technical Indicators
- **RSI (14)**: Momentum oscillator (0-100)
- **MACD**: Trend-following momentum (MACD, Signal, Histogram)
- **Bollinger Bands**: Volatility-adjusted support/resistance
- **Momentum (20)**: Short-term price momentum
- **Volume Ratio**: Current volume vs. 20-day SMA

#### Macro Features
- **Treasury Yield Lags**: 10-2 curve, 5-year, 30-year spreads
- **Inflation (CPI)**: Real vs. nominal yield impacts
- **Growth Proxy**: GDP growth rates

#### Sentiment Features
- **Kalshi Odds**: Prediction market implied probabilities
- All features lag-adjusted to ensure no lookahead bias

#### Stationarity Checks
- **ADF Test**: Verifies I(0) property of engineered features
- **Log-Returns**: Primary input uses log differences for stability

**Key Methods:**
- `engineer_features()`: End-to-end feature generation
- `check_stationarity()`: Augmented Dickey-Fuller test for each feature
- `prepare_train_test_split()`: Walk-forward temporal split

---

### 3. **model.py**

Quantile regression using LightGBM with comprehensive backtesting:

#### QuantileModel
```python
model = QuantileModel(quantiles=[0.2, 0.5, 0.8])
model.train(X_train, y_train)
predictions = model.predict(X_test)  # Returns Dict[quantile → predictions]
```

**Configuration:**
- **Objective**: `quantile` for pinball loss
- **Alpha Parameter**: One model per quantile (0.2, 0.5, 0.8)
- **Loss Function**: Pinball loss minimizes asymmetric errors
- **Hyperparameters**: Learning rate, max depth, min data in leaf (tunable via Optuna)

#### BacktestEngine

Simulates **walk-forward validation** to avoid data leakage:

1. **Training Window**: 504 trading days (2 years)
2. **Test Window**: 63 trading days (3 months)
3. **Step Forward**: Move test window by 63 days, retrain model

**Evaluation Metrics:**

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Directional Accuracy (DA)** | % periods where sign(pred) = sign(actual) | 50%+ indicates directional skill |
| **Information Coefficient (IC)** | Spearman rank correlation | 0.05-0.10 is alpha-generating |
| **MAE (Returns %)** | Mean absolute error in %-points | Lower is better |
| **MAPE (Returns %)** | Mean absolute percentage error | Normalized error scale |
| **MAE/MAPE (Prices)** | Errors reconstructed to price levels | Dollar-based performance |

---

### 4. **main.py**

Orchestration layer with dual interfaces:

#### Command-Line Interface (CLI)

```bash
python main.py \
  --alpha-vantage-key YOUR_KEY \
  --fred-key YOUR_KEY \
  --ticker AAPL \
  --mode forecast
```

**Modes:**
- `forecast`: Generate 3-month price predictions for a single ticker
- `backtest`: Run full walk-forward validation and report metrics

#### Streamlit Web Interface

```bash
streamlit run main.py streamlit
```

**Features:**
- Interactive ticker input
- Real-time API credential handling
- "Fan chart" visualization (20th, 50th, 80th %ile prices)
- Price forecast display with return percentages

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Obtain API Keys

1. **Alpha Vantage**: https://www.alphavantage.co/api/ (free tier: 5 calls/min)
2. **FRED**: https://fredaccount.stlouisfed.org/login/ (free)
3. **Kalshi** (optional): https://kalshi.com/api

### Example 1: Generate Forecast

```python
from main import QuantileForecastingPipeline

pipeline = QuantileForecastingPipeline(
    alpha_vantage_key="YOUR_KEY",
    fred_api_key="YOUR_KEY",
)

forecast = pipeline.forecast_prices("MSFT")
# Output:
# {
#     "ticker": "MSFT",
#     "current_price": 350.25,
#     "forecast_date": "2026-08-23",
#     "predictions": {
#         "quantile_0.2": {"return_pct": -5.2, "forecasted_price": 331.95},
#         "quantile_0.5": {"return_pct": 8.3, "forecasted_price": 379.35},
#         "quantile_0.8": {"return_pct": 22.1, "forecasted_price": 427.50},
#     }
# }
```

### Example 2: Run Backtest

```python
pipeline = QuantileForecastingPipeline(
    alpha_vantage_key="YOUR_KEY",
    fred_api_key="YOUR_KEY",
)

data_dict = pipeline.load_and_prepare_data("GOOGL")
features = pipeline.engineer_features(data_dict)
pipeline.train_model(features)
results = pipeline.run_backtest(features, data_dict["price_data"]["Close"])

print(results.head())
#    fold  quantile        date  directional_accuracy  information_coefficient
# 0     0       0.2  2023-04-28                0.619                     0.087
# 1     0       0.5  2023-04-28                0.667                     0.102
# 2     0       0.8  2023-04-28                0.571                     0.064
```

---

## Interpretation & Validation

### Understanding the Quantile Predictions

The model outputs a **distribution of returns**, not a single point estimate:

- **20th Percentile** ($331.95): Pessimistic scenario—20% chance of lower returns
- **50th Percentile ($379.35)** (Median): Most likely outcome
- **80th Percentile** ($427.50): Optimistic scenario—20% chance of higher returns

The **spread** between 20th and 80th %iles indicates:
- **Wide spread**: High uncertainty / volatility
- **Narrow spread**: High conviction / stability

### Key Performance Indicators

1. **Directional Accuracy > 55%**: 
   - Baseline is 50% (random coin flip)
   - >55% indicates consistent directional signal
   - Used in risk management to filter predictions

2. **Information Coefficient > 0.05**:
   - IC range: -1.0 (perfect inverse) to +1.0 (perfect)
   - IC = 0.05 × Annual factor (~0.15-0.20 annually) is alpha-generating
   - Measures correlation of rankings (robust to scale errors)

3. **MAPE < 5% on Returns**:
   - <5% MAPE indicates ±5% forecast accuracy
   - Translates to ±$17.50 on $350 stock

---

## Data Leakage Prevention

**Critical Design Decisions:**

1. **Features at time T**: Use only data available at T
2. **Target at time T+63**: Strictly forward-looking (not available at T)
3. **No Walk-Ahead**: Never train on data after test period
4. **Macro Lags**: All macro features lagged 1-5 days to prevent future peeking

---

## Future Enhancements

- [ ] Hyperparameter tuning with Optuna (minimize pinball loss)
- [ ] Cross-asset correlations (SPX, bonds, VIX)
- [ ] Regime detection (bull/bear/sideways markets)
- [ ] Ensemble methods (combine with mean reversion, momentum)
- [ ] Kalshi sentiment weighting (event probability impact on returns)
- [ ] Production deployment with scheduled predictions
- [ ] Multi-horizon forecasts (1-month, 6-month, 12-month)

---

## References

1. **Quantile Regression**: Koenker & Bassett (1978) — "Regression Quantiles"
2. **LightGBM**: Microsoft's gradient boosting with built-in quantile loss
3. **Walk-Forward Validation**: Parmesano & Marks (2007) — preventing look-ahead bias
4. **Information Coefficient**: Grinold & Kahn (2000) — "Active Portfolio Management"
5. **Stationary Testing**: Augmented Dickey-Fuller test (Said & Dickey, 1984)

---

## License

MIT License — Feel free to use for educational and research purposes.

## Contact

Questions or improvements? Open an issue or reach out.

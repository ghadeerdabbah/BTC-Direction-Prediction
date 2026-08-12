# Bitcoin Price Direction Prediction Using LSTM and XGBoost

## Final Project – AI & Innovation in Capital Markets

This project predicts whether Bitcoin's (BTC) closing price will rise or fall **7 trading days ahead**, using two machine learning approaches:

1. **XGBoost** – Gradient-boosted decision tree classifier
2. **LSTM + GRU** – Deep learning sequential model

The project includes a full pipeline: data collection → feature engineering → EDA → modeling → evaluation → **trading backtest** with exposure-matched benchmarking.

## Results Summary

| Metric | XGBoost | LSTM |
|--------|---------|------|
| Test Accuracy | 52.8% | 40.0% |
| ROC-AUC | 0.561 | < 0.5 |
| Best Naive Baseline | 56.8% | 56.8% |

Neither model beats the best naive baseline (always predict Down = 56.8%). This is **not a bug** — the notebook includes a full regime-shift diagnostic proving that the training period (bull market) and test period (bear market) are fundamentally different regimes, causing momentum features to reverse their predictive sign.

**Trading Backtest:** XGBoost achieved positive alpha vs buy-and-hold by holding only ~40% average exposure during a declining market. However, the exposure-matched benchmark reveals this is beta reduction (sitting in cash), not genuine timing skill. The LSTM strategy performed worse than both benchmarks.

## Project Structure

```
├── AIPredictor.ipynb                    # Main notebook (run in Google Colab)
├── BTC_Prediction_Project_Summary.pdf   # 5-page project summary
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

## How to Run

1. Open `AIPredictor.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run All**
3. The notebook downloads data automatically via Yahoo Finance
4. Full execution takes approximately 5–10 minutes

No API keys or manual data downloads are required.

## Features Used (17 features)

**Target Asset:** BTC-USD (Bitcoin)

**Explanatory Assets:**
- ETH-USD (Ethereum)
- ^GSPC (S&P 500 Index)
- GLD (Gold ETF)
- DX-Y.NYB (US Dollar Index)

**Technical Indicators (BTC):**
- RSI (14-day Relative Strength Index)
- MACD histogram normalised by price (scale-invariant)
- Bollinger Band Width (20-day)
- Distance from 20-day MA (%)

**Returns:**
- Daily returns for all 5 assets

**Momentum (stationary replacement for raw price levels):**
- 5-day momentum for all 5 assets
- 10-day and 20-day momentum for BTC

**Volatility:**
- 20-day rolling standard deviation of BTC returns

> All features are stationary by construction — no raw price levels are used as model inputs.

## Pipeline Steps

1. **Data Collection** – 6+ years of daily data (2020-01-01 to 2026-07-15) from Yahoo Finance via `yfinance`. Merged dataset: 1,640 rows across 5 assets.
2. **Feature Engineering** – 17 stationary features (1,600 usable rows after indicator warm-up and target construction)
3. **Exploratory Data Analysis** – Price chart, return distribution, correlation heatmap on model features, outlier detection (6.2% of days), target balance (46.1% Down / 53.9% Up)
4. **Data Preparation** – `StandardScaler` fitted on training data only; chronological train/test split (pre-2026 / 2026); purge of last 7 training rows to prevent label leakage. Train: 1,468 rows, Test: 125 rows.
5. **XGBoost Training** – 500 estimators, max depth 4, learning rate 0.03, class weight balancing, early stopping on chronological validation slice (220 rows)
6. **LSTM+GRU Training** – 30-day sequences, 64→32→16 architecture (30,945 parameters) with dropout 0.2–0.3, early stopping with `restore_best_weights`, balanced class weights
7. **Model Comparison** – Accuracy, ROC-AUC, precision/recall/F1, comparison against three naive baselines
8. **Regime-Shift Diagnostic** – Label verification, AUC by split, feature correlation sign-flip analysis, rolling correlation plot
9. **Trading Backtest** – Strategy vs buy-and-hold AND exposure-matched benchmark, 0.1% fees, Sharpe ratio, max drawdown, sanity controls

## Key Findings

- **XGBoost** achieved 52.8% accuracy and 0.561 ROC-AUC — above chance but below the best naive baseline (56.8%)
- **LSTM** achieved only 40.0% accuracy — systematically inverted predictions due to concept drift
- **Top features by importance:** Bollinger Band Width, BTC 20-day momentum, BTC 5-day momentum, MACD (normalised), BTC vs MA20 distance
- **Regime shift confirmed:** many features reversed their correlation sign between training (bull) and test (bear) periods
- **LSTM overfitting visible:** training loss continued decreasing while validation loss diverged sharply after ~5 epochs

## Methodological Highlights

| Decision | What we did | Why it matters |
|---|---|---|
| Feature stationarity | Returns, momentum, ratios — no raw prices | Prevents train/test distribution mismatch |
| Scaler fitting | Train only | Prevents test-period mean/variance leakage |
| Split boundary | Purged last 7 training rows | Labels resolving in the test window leak |
| Model selection | Early stopping on validation, not test | Using the test set is leakage |
| Backtest benchmark | Exposure-matched AND buy-and-hold | Buy-and-hold alone rewards sitting in cash |
| Sanity checks | Always-buy ≈ buy-and-hold, never-buy = 0% | Validates the backtest engine itself |

## Technologies

| Component | Tool |
|-----------|------|
| Data sourcing | `yfinance` |
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Technical indicators | `ta` |
| Deep learning | `tensorflow` / `keras` (LSTM + GRU) |
| Tree-based model | `xgboost` |
| Evaluation & scaling | `scikit-learn` |

## Requirements

```
yfinance
pandas
numpy
matplotlib
seaborn
ta
tensorflow
xgboost
scikit-learn
```

## Limitations

- Single chronological split — only one market regime tested out-of-sample
- 125-day test set gives ±4.4pp standard error on accuracy
- No alternative data (sentiment, on-chain, Google Trends)
- Backtest assumes execution at close with no slippage or market impact
- Not investment advice

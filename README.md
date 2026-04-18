# SP500 Quantitative Factor Pipeline

A production-grade daily quantitative equity strategy targeting the S&P 500 universe. The pipeline covers data ingestion, factor engineering, dynamic signal weighting, ML-based scoring, and a realistic long/short backtest engine — all in a single end-to-end run.

---

## Architecture Overview

```
main.py
├── Stage 1   Data ingestion      data_loader.py
├── Stage 2   Factor engineering  features.py
├── Stage 3   ML feature matrix   alpha_models.py
├── Stage 4   ICIR composite      features.synthesize_dynamic()
├── Stage 5   Alphalens tearsheet simulator.prepare_alphalens_data()
├── Stage 6   Linear backtest     simulator.run_realistic_backtest()
├── Stage 7   Walk-forward ML     alpha_models.run_ml_scoring()
├── Stage 8   ML backtest         simulator.run_realistic_backtest()
└── Stage 9   Report generation   visualization.py
```

---

## Module Summary

| File | Role |
|---|---|
| `src/config.py` | All constants: tickers, dates, backtest params, ICIR and ML hyperparameters |
| `src/data_loader.py` | Alpaca OHLCV download, VIX/SPY/macro fetch, news sentiment via Alpaca News API |
| `src/features.py` | 11-factor library (momentum, volatility, HLOC-specific) + dynamic ICIR synthesis |
| `src/alpha_models.py` | Walk-forward ML engine (LightGBM + Ridge ensemble), feature matrix construction |
| `src/simulator.py` | Variable-beta 130/30 backtest with VIX deleveraging and regime switching |
| `src/visualization.py` | All chart rendering functions → PNG files in `reports/` |
| `src/utils.py` | Shared logger (stdout + `reports/run.log`) |
| `src/test_data_integrity.py` | 5-point look-ahead bias test suite |

---

## Factor Library

### Classic Factors
| Factor | Description |
|---|---|
| `Momentum_252` | 12-month price momentum, skipping the last month |
| `Momentum_63`  | 3-month momentum |
| `MeanRevert_5` | 5-day short-term reversal |
| `LowVol_63`    | Low annualised volatility (63-day) |
| `AmihudIlliq`  | Amihud illiquidity ratio (negative: prefer liquid stocks) |
| `TurnoverRate` | Log turnover rate |

### HLOC Factors (exploit Open, High, Low)
| Factor | Formula | Intuition |
|---|---|---|
| `OvernightGapRev` | `−(Open/Close_prev − 1)` | Fade extreme overnight gaps |
| `IntradayMom`     | `Close/Open − 1`          | Institutional buying within the session |
| `ParkinsonVol_21/63` | `√(ln(H/L)² / 4ln2)` | More efficient volatility estimate than close-to-close |
| `CLV_21`          | `((C−L)−(H−C)) / (H−L)` | Closing location within daily range (buying pressure) |

All factors pass through `_cs_clean()`: cross-sectional MAD winsorisation → z-score → optional EWMA smoothing.

---

## Signal Weighting: Dynamic ICIR Synthesis

Each factor's weight is proportional to its rolling Information Ratio (IC / IC_std).  
Three lookback windows (63 / 126 / 252 days) are blended at weights (0.2 / 0.3 / 0.5) to stabilise weight allocation across regime changes.

**IC sign-consistency protection**: for directionally unstable factors (`LowVol_63`, `ParkinsonVol_*`), if the monthly IC has been negative for 3 consecutive months, the weight is zeroed until the signal recovers. The IC series is shifted by `forward_days` before weight computation to ensure no look-ahead.

---

## ML Scoring Engine

Walk-forward cross-validation with a strict embargo gap:

```
|←—— train (1.5 yr) ——→|← embargo (10d) →|← test (~63d) →|
```

- **Target**: binary label — top 10% cross-sectional 10-day forward return
- **Models**: LightGBM (0.7 weight) + L2-regularised logistic regression (0.3 weight)
- **Features**: 11 HLOC factors + 3 market-level + 4 macro + news sentiment + sector code
- **Retraining**: every 63 trading days (~1 quarter)

---

## Backtest Engine

**Strategy**: sector-neutral long/short equity
- Select top/bottom 10% within each sector by factor score
- Weight positions by inverse realised volatility (21-day)
- Smooth target weights over the holding period (rolling mean) to reduce churn
- Turnover band: only trade when deviation exceeds 0.5%
- Hard per-stock cap: ±10% of NAV

**Risk controls**:
- VIX deleveraging: linearly reduce gross exposure as VIX rises above 25
- Bull/bear regime: long target 1.30× (bull) / 0.80× (bear) based on price vs 120-day SMA
- Macro winter: simultaneous bear trend + rate shock → cap both legs at 0.50×
- Transaction cost: 5 bps per unit of one-way turnover

---

## Backtest Results (Sep 2020 – Mar 2026)

### Linear ICIR Strategy

| Metric | Value |
|---|---|
| Total Return | 66.69% |
| CAGR | 9.63% |
| Annualised Volatility | 11.18% |
| Sharpe Ratio | 0.861 |
| Sortino Ratio | 1.210 |
| Calmar Ratio | 0.465 |
| Max Drawdown | −20.73% |
| Max DD Duration | 630 days |
| Win Rate (daily) | 53.96% |
| Avg Daily Turnover | 3.31% |
| Annual Turnover | 8.35× |
| Avg Gross Exposure | 1.616× |

### ML Ensemble Strategy

| Metric | Value |
|---|---|
| Total Return | 287.33% |
| CAGR | 27.60% |
| Annualised Volatility | 16.27% |
| Sharpe Ratio | 1.697 |
| Sortino Ratio | 2.524 |
| Calmar Ratio | 1.771 |
| Max Drawdown | −15.59% |
| Max DD Duration | 405 days |
| Win Rate (daily) | 56.83% |
| Avg Daily Turnover | 6.95% |
| Annual Turnover | 17.52× |
| Avg Gross Exposure | 1.654× |

### Alphalens Factor Evaluation (composite signal)

| Horizon | Ann. Alpha | IC Mean | Risk-Adjusted IC |
|---|---|---|---|
| 1D | 3.8% | 0.004 | 0.030 |
| 5D | 3.2% | 0.005 | 0.046 |
| 10D | 2.6% | 0.007 | 0.061 |
| 21D | 3.2% | 0.014 | 0.123 |

The spread between top and bottom quintile is ~2.4 bps/day at 1D, rising to ~2.2 bps at 21D, with near-zero beta across all horizons.

### Walk-Forward ML OOS IC

27 quarterly folds (Jul 2019 – Mar 2026). Mean OOS ensemble IC: **+0.216**.  
IC was consistently positive across all folds (range: +0.142 to +0.285), demonstrating robust out-of-sample predictive power with no sign of decay over the 6-year period.

---

## Reports Generated

| File | Contents |
|---|---|
| `performance_dashboard.png` | Equity curve, rolling Sharpe, drawdown, factor IC heatmap |
| `factor_ic_summary.png` | Mean IC and ICIR bar chart per factor |
| `icir_weight_history.png` | Dynamic ICIR weights over time (line + stacked area) |
| `ml_diagnostics.png` | Per-fold OOS IC, cumulative IC, LightGBM feature importance |
| `strategy_comparison.png` | Linear vs ML equity curves and metrics side-by-side |
| `monthly_returns_heatmap.png` | Calendar heatmap of monthly net returns |
| `cost_turnover_analysis.png` | Gross/net return decomposition, turnover and cost time-series |
| `drawdown_analysis.png` | Top-5 drawdown periods table + underwater curve |
| `regime_conditional_returns.png` | Returns split by VIX regime and market trend |
| `long_short_attribution.png` | Long-leg vs short-leg daily P&L |
| `reports/plots/` | Alphalens tearsheet: quantile returns, IC time-series, turnover |

---

## Quick Start

```bash
pip install -r requirements.txt

# Full run (linear + ML)
python main.py

# Linear ICIR only (faster)
python main.py --linear-only

# Force re-download all data
python main.py --force-refresh

# Skip Alphalens tearsheet
python main.py --no-alphalens

# Data integrity tests
python -m src.test_data_integrity
```

**Data is cached** in `data/raw/` after the first run. Subsequent runs load from disk unless `--force-refresh` is passed.

---

## Requirements

- Python 3.10+
- Alpaca Markets API key (paper or live) — set in `data_loader.py`
- See `requirements.txt` for full dependency list

Key dependencies: `alpaca-trade-api`, `yfinance`, `lightgbm`, `scikit-learn`, `alphalens-reloaded`, `pandas`, `numpy`, `scipy`, `matplotlib`

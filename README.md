# Dynamic ORB — Intelligent Opening Range Breakout System

> **Status:** Prototype under active development. Not production-ready. Strategy is experimental.

---

## Overview

Dynamic ORB is an intraday algorithmic trading system built for NQ (Nasdaq 100 Futures) targeting prop-firm environments with end-of-day drawdown rules (Topstep, MFF, Take Profit Trader, etc.). The system is designed to answer a single question on each trading day:

**"Which opening range timeframe gives the highest-probability breakout today — and is the move real or a fakeout?"**

It does this by combining classical ORB mechanics with order flow analysis, machine learning proxies built from institutional-grade MBO (Market-by-Order) tick data, local LLM reasoning, and walk-forward backtesting.

**Current backtest result:** `$56,011` profit on a 2-year 1-minute OHLCV sample set | Sharpe: `0.84`

---

## Core Concept

Standard ORB strategies pick a fixed opening range (e.g., "always use the first 5 minutes"). This system instead treats the opening range duration as a *learned variable*. It trains on historical data to determine which timeframe — from 1 minute to 1 hour — is statistically best-suited for the current session's characteristics.

On top of that, it integrates two custom order flow indicators derived from L3 MBO data to filter entries, distinguishing genuine institutional breakouts from retail fakeouts.

---

## Architecture Overview

```
Module.1  → Data ingestion (16-year OHLCV + 8-month L3 MBO)
    │
Module.2a → Indicators (Volume Profile/POC + Delta Imbalance + 1.25x ATR)
    │
Module.2b → Proxy models (XGBoost — maps OHLCV features → MBO ground truth)
    │
Module.3  → ORB strategy variants (1m, 3m, 5m, 15m, 30m, 1h openings)
    │
Module.4  → Time series & Monte Carlo testing per variant
    │
Module.5  → Walk-forward rolling window backtest + variant weighting
    |
Module.6  → Implement weights into a NN
    │
Module.7  → Local LLM (llama3.2:1b via Ollama) reasoning layer
    │
Module.8  → Final signal: session ORB recommendation + entry/exit logic
```

---

## Modules

### Module 1 — Data Ingestion

Loads and preprocesses two data sources:

**OHLCV (16 years, 2010–2026):**
Aggregated 1-minute bars for NQ futures stored as CSV. Parses UTC timestamps, filters to front-month outright contracts (excluding spreads like `NQU5-NQZ5`), and casts to memory-efficient float32/int32 dtypes.

**MBO / L3 Tick Data (~8 months):**
Raw DataBento `.dbn.zst` files organized by date. Contains every individual order book event. The system extracts `action == 'T'` (matched trades) and classifies aggressor side: `'B'` = buyer lifted the ask (bullish), `'A'` = seller hit the bid (bearish). Prices are already in real NQ point units — no conversion needed.

The ~8-month MBO overlap with the full OHLCV history is what enables the proxy model approach described in Module 2b.

---

### Module 2a — Indicators

#### Volume Profile with POC

Built from L3 MBO trade data. Every executed trade is bucketed to the nearest NQ tick (`0.25` points). The resulting distribution reveals price acceptance and rejection:

- **POC (Point of Control):** price level with the highest traded volume — the market's "fair value" anchor for the session
- **HVNs (High Volume Nodes):** levels at or above the 70th percentile — zones of acceptance, likely to act as support/resistance
- **LVNs (Low Volume Nodes):** levels at or below the 30th percentile — zones of rejection, price moves quickly through these
- **Profile Skew:** `volume above POC / volume below POC` — measures directional bias

Visualized as a horizontal histogram overlaid on the 1-minute candlestick chart. POC is drawn as an orange dashed line. HVNs are teal, LVNs are dim gray.

#### Delta Imbalance (Volume Delta)

Per-minute order flow imbalance computed from MBO trades:

- **Delta:** `buy_volume − sell_volume` per minute
- **Delta Ratio:** `|delta| / total_volume` — measures one-sidedness (0–100%)
- **Buy/Sell Ratio:** `buy_vol / sell_vol` — directional dominance

**Signal Logic:**
```
EXECUTE LONG  →  Buy/Sell Ratio > 300%  AND  |Δ|/Volume > 15%
IGNORE        →  Buy/Sell Ratio < 300%  OR   |Δ|/Volume < 10%
```

This filters out fakeouts where price breaks a level but order flow does not confirm it. A move above a range high with a `buy_sell_ratio` of 1.1 and a `delta_ratio` of 5% is a fakeout candidate. A move with a ratio of 4.2 and delta conviction of 22% is institutional.

#### 1.25x ATR — Risk Management

A 1.25× ATR multiplier determines the initial stop-loss distance from entry. A 60/40 trailing stop is applied once the trade moves in favor — locking in 60% of unrealized gains while allowing 40% room for continuation.

---

### Module 2b — Statistical Proxy Models

**The problem:** The volume profile and delta imbalance indicators require MBO data, which is expensive and only covers ~8 months. The OHLCV history goes back 16 years.

**The solution:** Train XGBoost regression models during the overlap period (OHLCV features → MBO ground truth), then apply those models to generate *proxy* versions of the indicators for the full 16-year dataset.

#### Proxy 1 — Volume Profile POC (Daily Resolution)

| Cellblock | Purpose |
|-----------|---------|
| CB 1 | Build daily OHLCV feature matrix: range, returns, volatility, volume momentum |
| CB 2 | Join OHLCV features with true MBO POC for the overlap period |
| CB 3 | Train XGBoost on first 80% of overlap dates (chronological split, never random); validate on last 20% |
| CB 4 | Apply model to all 16 years → `proxy_poc` column added to OHLCV dataframe |
| CB 5 | Sanity check: overlay `proxy_poc` vs true POC on validation days |

Target variables: `mbo_poc`, `mbo_value_area_high`, `mbo_value_area_low`, `mbo_profile_skew`

#### Proxy 2 — Delta Imbalance (Minute Resolution)

| Cellblock | Purpose |
|-----------|---------|
| CB 6 | Vectorized per-minute feature extraction using `.rolling()` — eliminates O(N²) loop |
| CB 7 | Join per-minute OHLCV features with true MBO delta targets for overlap period |
| CB 8 | Train XGBoost regression for `delta_ratio` and `buy_sell_ratio` (80/20 chronological) |
| CB 9 | Batch vectorized prediction across full 16-year history: `proxy_delta_ratio`, `proxy_buy_sell_ratio` |
| CB 10 | Sanity check: overlay true vs proxy delta on validation days; mark flagged signals |

All feature engineering uses `.shift(1)` to prevent look-ahead bias. Processing is done day-by-day to avoid memory overload on 500k+ minute bars.

---

### Module 3 — ORB Strategy Variants *(planned)*

Multiple ORB configurations are defined and compared:

- **Variant A:** 1-minute opening range
- **Variant B:** 3-minute opening range
- **Variant C:** 5-minute opening range
- **Variant D:** 15-minute opening range
- **Variant E:** 30-minute opening range
- **Variant F:** 60-minute opening range

Each variant gets a performance-based weight in Module 5. The session recommendation engine uses these weights to select the best variant for the current day's characteristics.

---

### Module 4 — Time Series Testing & Monte Carlo Simulation *(planned)*

Evaluates each ORB variant's edge with statistical rigor:

- **Walk-forward time series testing** to confirm edge persists out-of-sample
- **Monte Carlo simulation** to estimate the distribution of possible outcomes and drawdown risk
- Results feed directly into the variant weighting system

---

### Module 5 — Walk-Forward Backtest *(planned)*

A rolling window backtest that moves forward through time in fixed steps, retraining the variant selector at each step. This prevents overfitting to any single market regime and validates that the system generalizes.

---

### Module 6 — Neural Network implementation *(experimental)*

Will be used to weigh each strategy as well as each individual time series forecast per startegy for determing which ORB variant has the highest probability of profiting on the given day.

---

### Module 7 — Local LLM Reasoning *(experimental)*

Uses `llama3.2:1b` running locally via **Ollama** for interpretive reasoning on top of the quantitative signals. The LLM layer is not a decision-maker — it acts as a reasoning filter that can contextualize signal conflicts and provide explanatory output for why a particular ORB variant was recommended on a given session.

This module is explicitly experimental. The quantitative pipeline functions independently without it.

---

## Data Requirements

| Data Type | Format | Coverage | Source |
|-----------|--------|----------|--------|
| NQ 1-minute OHLCV | CSV (Databento) | 2010-06-06 → 2026-01-23 | Databento |
| NQ L3 MBO | `.dbn.zst` (Databento) | ~8 months (2025-2026) | Databento |

MBO files are organized by date in folders and indexed by a `files_by_date` dictionary at runtime. The system automatically resolves the overlapping date range between both datasets.

---

## Dependencies

```python
# Data & numerics
pandas, numpy, scipy

# Market data
databento          # MBO .dbn.zst file reading
yfinance           # supplemental price data

# Machine learning
xgboost            # POC and delta proxy models
scikit-learn       # metrics, preprocessing
torch              # neural network components (Module 3+)
statsmodels        # time series analysis

# Technical analysis
ta                 # ATR and other indicators

# Visualization
matplotlib, seaborn

# LLM (local)
ollama             # llama3.2:1b runtime (requires Ollama installed)

# Utilities
pathlib, glob, gc, logging, collections
```

---

## File Structure

```
project/
├── strategy.ipynb       # Main notebook (all modules)
├── prompting.md         # Module.2b specification and design notes
├── README.md            # This file
└── data/
    ├── historical/
    │   └── NQ/
    │       ├── 1m OHLCV data/
    │       │   └── NQ-2010.06.06-2026.01.23-ohlcv-1m.csv
    │       └── MBO data/
    │           ├── 2025.07.28-2025.10.08/
    │           ├── 2025.10.10-2025.12-17/
    │           ├── 2025.12.18-2026.03.08/
    │           └── 2026.03.09-2026.05.19/
```

---

## Key Design Decisions

**Chronological splits only.** No random train/test splits anywhere in the system. All models split by date: first 80% of dates for training, last 20% for validation. This is non-negotiable for time series.

**Vectorized feature engineering.** All per-minute feature extraction uses pandas `.rolling()` with `.shift(1)` rather than per-minute Python loops. This reduces a 30+ minute operation to under one second on 500k+ bars.

**Proxy model philosophy.** The proxy approach assumes that OHLCV bar shape carries latent information about the underlying order flow that produced it. The XGBoost models learn this mapping during the overlap period and apply it to the full history. The proxy will not be as accurate as true MBO data, but it provides a statistically grounded approximation that is far better than no signal at all.

**Front-month filtering.** All analysis filters to `^NQ[A-Z]\d$` (outright front-month contracts) and excludes calendar spreads. On days where multiple contracts trade, the highest-volume contract is selected.

---

## Prop-Firm Compatibility

Designed for accounts with end-of-day drawdown limits. The 1.25x ATR initial stop ensures position sizing remains within typical prop-firm daily loss thresholds when combined with standard lot sizing. The 60/40 trailing stop structure is designed to protect overnight gains while avoiding premature exits during intraday continuation.

---

## What Is Not Yet Built

- Module 3 (ORB variant strategy logic)
- Module 4 (Monte Carlo and time series validation)
- Module 5 (rolling walk-forward backtest)
- module 6 (NN implementation)
- Module 7 (LLM integration — Ollama)
- Module 8 (final signal aggregation and session recommendation engine)
- Automated variant weighting system
- Live execution interface

---

## Disclaimer

This system is a research prototype. Nothing here constitutes financial advice. Past backtest results do not guarantee future performance. Futures trading involves substantial risk of loss. All strategy development and testing should be conducted in simulation environments before any live deployment.


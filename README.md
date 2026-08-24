# DynamicORB

### Machine Learning–Driven Opening Range Breakout Research System

DynamicORB is an experimental quantitative research framework designed to investigate whether machine learning can improve **Opening Range Breakout (ORB)** strategies by adapting signals to changing market conditions.

The system combines historical market data, feature engineering, machine-learning models, time-series validation, and quantitative backtesting to study how market structure and intraday behavior influence ORB performance.

> **Status:** Active Research / Experimental
> **Domain:** Machine Learning · Quantitative Finance · Time-Series Modeling · Algorithmic Trading
> **Primary Language:** Python

---

## Overview

Traditional Opening Range Breakout strategies use a fixed set of rules to determine whether a market breakout is likely to continue.

DynamicORB explores a different question:

> **Can machine learning identify market conditions in which an ORB setup is more or less likely to succeed?**

Rather than treating every breakout equally, the project is designed to build a pipeline capable of analyzing market context, generating predictive features, estimating breakout outcomes, and evaluating strategy performance under realistic chronological conditions.

The long-term objective is to develop an adaptive research framework capable of dynamically adjusting ORB signals based on learned market behavior.

---

# Research Objectives

DynamicORB investigates several questions:

* Can market microstructure features improve ORB predictions?
* Can machine learning identify favorable ORB market regimes?
* Can models generalize to unseen market periods?
* How does model performance change across different volatility environments?
* Can adaptive signals outperform static ORB rules after transaction costs and slippage?
* How robust are the results to different market regimes?
* Can model predictions be converted into actionable trading signals?

The project prioritizes **out-of-sample performance and robustness** over maximizing in-sample results.

---

# System Architecture

The intended research pipeline is:

```text
Historical Market Data
        │
        ▼
Data Cleaning & Normalization
        │
        ▼
Feature Engineering
        │
        ├── Price Features
        ├── Volume Features
        ├── Volatility Features
        ├── Market Structure
        └── Order-Flow / MBO Features
        │
        ▼
Machine Learning Models
        │
        ├── Baseline Models
        ├── XGBoost
        └── Neural Networks
        │
        ▼
Time-Series Validation
        │
        ├── Chronological Splits
        ├── Walk-Forward Testing
        └── Monte Carlo Analysis
        │
        ▼
Signal Generation
        │
        ▼
Quantitative Backtesting
        │
        ├── Returns
        ├── Sharpe Ratio
        ├── Drawdown
        ├── Win Rate
        └── Risk Metrics
        │
        ▼
Research Results
```

---

# Data

DynamicORB is designed to work with historical intraday market data, including:

* OHLCV data
* Minute-level market data
* Market-by-order (MBO) / Level-3 data
* Volume information
* Intraday price structure

The system is designed around chronological financial data where the temporal relationship between observations is critical.

---

# Machine Learning Approach

The project investigates multiple model classes rather than relying on a single algorithm.

### Baseline Models

Simple statistical and machine-learning baselines provide reference points for determining whether more complex models actually provide additional predictive value.

### XGBoost

Gradient-boosted decision trees are used to investigate nonlinear relationships between engineered market features and ORB outcomes.

### Neural Networks

Future iterations explore neural-network architectures for learning more complex representations of intraday market behavior.

The goal is not to use the most complicated model possible.

The goal is to determine:

> **Whether additional model complexity produces meaningful out-of-sample improvement.**

---

# Feature Engineering

Potential feature categories include:

### Price

* Opening price
* High / low relationships
* Opening range size
* Breakout distance
* Price momentum
* Previous-session structure

### Volume

* Relative volume
* Volume acceleration
* Opening volume
* Intraday volume distribution

### Volatility

* Realized volatility
* Opening-range volatility
* ATR-style measurements
* Volatility regime

### Market Structure

* Breakout direction
* Range expansion
* Previous highs/lows
* Intraday trend structure

### Market Microstructure

Where available, MBO/L3 data can be used to investigate:

* Order-flow imbalance
* Bid/ask dynamics
* Market depth
* Liquidity conditions
* Aggressive buying/selling behavior

---

# Preventing Look-Ahead Bias

Financial machine-learning systems are particularly vulnerable to **look-ahead bias**.

DynamicORB therefore prioritizes strict chronological processing.

The system avoids randomly shuffling time-series observations during model evaluation.

Where appropriate, features are shifted so that information from the future cannot enter a prediction made at an earlier point in time.

For example:

```python
df["feature_lagged"] = df["feature"].shift(1)
```

The objective is to ensure that:

```text
Information available at time T
            ↓
Prediction made at time T
            ↓
Future outcome at T + n
```

rather than:

```text
Future information
        ↓
Prediction
        ↓
Historical outcome
```

---

# Validation Strategy

Random train/test splitting can produce misleading results in financial time series.

DynamicORB instead emphasizes chronological evaluation.

Conceptually:

```text
TRAIN
████████████████

VALIDATION
                ████████

TEST
                        ████████
```

Future iterations are designed to incorporate rolling walk-forward validation:

```text
Train ────────► Test
   Train ────────► Test
      Train ────────► Test
         Train ────────► Test
```

This provides a more realistic estimate of how a model might behave when exposed to previously unseen market conditions.

---

# Backtesting

Model predictions are ultimately evaluated in the context of a trading strategy.

Key metrics include:

| Metric            | Purpose                         |
| ----------------- | ------------------------------- |
| Total Return      | Overall strategy performance    |
| CAGR              | Annualized growth               |
| Sharpe Ratio      | Risk-adjusted return            |
| Sortino Ratio     | Downside-risk-adjusted return   |
| Maximum Drawdown  | Worst peak-to-trough decline    |
| Win Rate          | Percentage of profitable trades |
| Profit Factor     | Gross profit / gross loss       |
| Average Trade     | Expected trade outcome          |
| Turnover          | Trading activity                |
| Transaction Costs | Realistic execution impact      |

A model is not considered successful simply because it achieves high classification accuracy.

The objective is to determine whether predictions translate into **robust risk-adjusted trading performance**.

---

# Experimental Design

DynamicORB follows a research-first methodology.

Models should be compared against progressively stronger baselines:

```text
Static ORB
     │
     ▼
Simple Statistical Model
     │
     ▼
Traditional Machine Learning
     │
     ▼
Gradient Boosting
     │
     ▼
Neural Network
     │
     ▼
Adaptive / Ensemble Approach
```

Each increase in complexity should justify itself through measurable out-of-sample improvement.

---

# Current Development Status

| Component                    | Status            |
| ---------------------------- | ----------------- |
| Historical data ingestion    | 🟢 Implemented    |
| Data preprocessing           | 🟢 Implemented    |
| Feature engineering          | 🟢 Implemented    |
| Chronological data splitting | 🟢 Implemented    |
| XGBoost modeling             | 🟢 Implemented    |
| ORB research framework       | 🟡 Experimental   |
| Advanced ORB variants        | 🟡 In Development |
| Walk-forward validation      | 🟡 In Development |
| Monte Carlo analysis         | 🟡 Planned        |
| Neural-network modeling      | 🟡 In Development |
| LLM research layer           | 🔵 Experimental   |
| Adaptive signal aggregation  | 🔵 Planned        |
| Live execution               | 🔵 Future Work    |

**Legend**

* 🟢 Implemented
* 🟡 In Development
* 🔵 Planned / Experimental

---

# Important Research Disclaimer

This repository is a **research project**, not a production trading system or financial product.

Machine-learning models can overfit historical data and produce results that fail to generalize to future market conditions.

Backtested performance does not guarantee future performance.

Particular attention is given to:

* Overfitting
* Look-ahead bias
* Data leakage
* Survivorship bias
* Regime changes
* Transaction costs
* Slippage
* Execution assumptions

The purpose of DynamicORB is to investigate these problems scientifically rather than make claims of guaranteed profitability.

---

# Technology Stack

### Programming

* Python

### Data & Scientific Computing

* NumPy
* pandas
* SciPy
* Matplotlib

### Machine Learning

* scikit-learn
* XGBoost
* PyTorch

### Quantitative Research

* Time-series analysis
* Feature engineering
* Backtesting
* Walk-forward validation
* Monte Carlo simulation
* Market microstructure analysis

### AI Infrastructure

* Local LLM experimentation
* Ollama

---

# Project Structure

```text
DynamicORB/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   └── research/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── validation/
│   ├── backtesting/
│   └── signals/
│
├── tests/
│
├── experiments/
│
├── configs/
│
├── README.md
├── pyproject.toml
└── .gitignore
```

*The structure will evolve as the research framework develops.*

---

# Research Roadmap

### Phase 1 — Data

* [x] Historical market-data ingestion
* [x] Data preprocessing
* [x] Initial feature engineering
* [x] Chronological splitting

### Phase 2 — Modeling

* [x] Establish baseline
* [x] XGBoost model
* [ ] Feature selection
* [ ] Hyperparameter optimization
* [ ] Neural-network models

### Phase 3 — Validation

* [ ] Rolling walk-forward testing
* [ ] Monte Carlo analysis
* [ ] Regime-based evaluation
* [ ] Out-of-sample evaluation
* [ ] Statistical significance testing

### Phase 4 — Strategy

* [ ] Multiple ORB variants
* [ ] Adaptive signal weighting
* [ ] Transaction-cost modeling
* [ ] Slippage modeling
* [ ] Risk management framework

### Phase 5 — Production Research

* [ ] FastAPI inference service
* [ ] Docker containerization
* [ ] Model versioning
* [ ] Experiment tracking
* [ ] Monitoring
* [ ] Paper-trading interface

---

# Research Philosophy

DynamicORB is built around a simple principle:

> **A more complicated model is not automatically a better model.**

A strategy that performs exceptionally well on historical data but fails under walk-forward testing is not considered a successful result.

The research process therefore prioritizes:

**Data integrity → Reproducibility → Validation → Robustness → Performance**

rather than:

**Complexity → Optimization → Backtest → Claim**

---

# Future Direction

The long-term objective is to develop DynamicORB into a modular research platform capable of evaluating adaptive trading strategies across different market regimes.

Potential future research includes:

* Multi-model ensembles
* Regime classification
* Reinforcement learning
* Transformer-based time-series models
* Advanced order-flow modeling
* Dynamic position sizing
* Probabilistic signal generation
* Online learning
* Model monitoring
* Paper trading
* Real-time inference

These features will only be considered successful if they demonstrate measurable improvements under rigorous out-of-sample evaluation.

---

# Author

**Andrew Kozak**

Machine Learning · Quantitative Finance · AI Systems · Software Engineering

GitHub: [@andrewjkdev](https://github.com/andrewjkdev)

---

## Status

**Experimental Research — Not for Live Trading**

DynamicORB is continuously evolving as new experiments, validation methods, and models are developed.

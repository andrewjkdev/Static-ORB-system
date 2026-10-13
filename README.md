# Static ORB System

### Opening Range Breakout research on US ETFs: from a published strategy to a validated, tradeable edge

Static ORB System is a quantitative research project that set out to answer one question:

> **Can an Opening Range Breakout (ORB) strategy be improved, and validated out of sample, using market context, machine learning and news data?**

It started with the published ORB strategy of Zarattini, Barbon & Aziz (2024). The paper's version didn't hold up on ETFs, so the project searched 13,440 variations under a strict anti-overfitting protocol. That search found one configuration that held up from 2017 to 2026 on data it had never seen. This README explains the research, the exact winning strategy, why it works, and how to trade it.

> **Status:** Research complete for the ETF phase · **Domain:** Quantitative finance · Time-series validation · **Language:** Python
> **Not financial advice. Backtested results do not guarantee future performance.**

---

## Table of contents
1. [Results at a glance](#1-results-at-a-glance)
2. [The winning strategy: exact rules](#2-the-winning-strategy-exact-rules)
3. [Why it works](#3-why-it-works)
4. [How to trade it, step by step](#4-how-to-trade-it-step-by-step)
5. [Risk management & monitoring](#5-risk-management--monitoring)
6. [The research, start to finish](#6-the-research-start-to-finish)
7. [Methodology & anti-overfitting safeguards](#7-methodology--anti-overfitting-safeguards)
8. [Data](#8-data)
9. [Reproducing the research](#9-reproducing-the-research)
10. [Limitations & open questions](#10-limitations--open-questions)
11. [Project structure](#11-project-structure)

---

## 1. Results at a glance

**Live / out-of-sample period: Jan 2017 – Sep 2026.** The strategy was selected using only data up to 2016. Sizing is 1 % risk per trade with a 4× gross-leverage cap and whole shares, after commissions and slippage.

| | Winning ORB | SPY buy & hold |
|---|---|---|
| CAGR | **21.3 %** | 14.7 % |
| Sharpe ratio | **1.33** | 0.84 |
| Max drawdown | **−11.9 %** | −33.9 % |
| Worst calendar year | −3.3 % | −22.8 % |
| Volatility | 15.4 % | 18.3 % |
| Correlation to SPY | **−0.05** | 1.00 |
| Mean result per trade | +0.255 R (t = 5.26) | — |

**During market crashes:**

| SPY crash | SPY | Winning ORB |
|---|---|---|
| GFC 2007–09 | −54.8 % | **+121.2 %** |
| Q4 2018 | −18.5 % | +6.2 % |
| COVID 2020 | −33.5 % | +0.4 % |
| 2022 bear market | −28.4 % | **+38.4 %** |
| 2025 tariff shock | −18.4 % | −3.4 % |

**What is and isn't statistically proven:**
- ✅ **The edge is real.** Live t-statistics are 4–5 per trade, per day and per daily return, all above the |t| > 3 bar recommended for strategies found by searching.
- ✅ **It's uncorrelated with the stock market** (correlation −0.05, p = 0.008).
- ✅ **No evidence of decay** between the in-sample and live periods (p = 0.41).
- ⚠️ **It doesn't provably beat SPY.** The Sharpe difference has p = 0.29 and the return difference p = 0.49. Ten years of data can't separate the two statistically.

---

## 2. The winning strategy: exact rules

**Name:** 15-minute gap-confirmed ORB on index ETFs (internal config #9323)

### Universe
| Item | Rule |
|---|---|
| Instruments | **SPY, QQQ, IWM, DIA**, traded independently, at most one trade per ETF per day |
| Session | US regular hours, 09:30–16:00 ET, 1-minute bars |
| ATR | 14-day Wilder ATR of daily bars, **using prior days only** |

### Entry
| # | Rule |
|---|---|
| 1 | **Opening range (OR):** high and low of the first **15 minutes** (bars 09:30–09:44). |
| 2 | **OR direction:** bullish if the 09:44 bar closes above the 09:30 open, bearish if below. Equal → **no trade**. |
| 3 | **Gap confirmation:** today's open vs. yesterday's close must point the **same way** as the OR candle. |
| 4 | **Gap size:** \|open − previous close\| ≥ **0.3 × ATR**. |
| 5 | **Narrow range:** OR high − OR low ≤ **0.5 × ATR**. |
| 6 | **Trigger:** long = OR high + **0.02 × ATR** (round up to the cent). Short = OR low − 0.02 × ATR (round down). |
| 7 | **Entry:** the first 1-minute bar from 09:45 that **closes beyond the trigger**. Enter at that close. |
| 8 | **Entry window:** no new entries after the **11:30** bar. |
| 9 | Only the direction set by rules 2–3. No re-entry the same day. |

### Exit
| Exit | Rule |
|---|---|
| Initial stop | **OR midpoint**, rounded away from entry. **1R** = entry − stop (per share). |
| Breakeven | Once price has traded **+2R** beyond entry (on completed bars), move the stop to entry. |
| Profit target | **None.** Winners run until the stop or the time exit. |
| Time exit | Flat at the **close of the 14:00 bar** (≈ 14:01 ET). Never held overnight. |

### Position sizing
```
shares = floor( min( risk% × equity / R_per_share ,  available_notional / entry_price ) )
available_notional = 4 × start-of-day equity − notional of positions already open
```
- **risk% = 1 %** of start-of-day equity in the realistic version (0.25 % was used during the search).
- The **4× gross-leverage cap** (US intraday margin) often binds. Actual risk averaged **0.62 %** per trade.

### Trade statistics (2003–2026, 2,844 trades)
| Metric | Value |
|---|---|
| Win rate | 42.2 % |
| Average win / loss | +1.79 R / −0.92 R |
| Mean per trade | +0.227 R (t = 7.46) |
| Exits | 53 % stop, 47 % time exit |

---

## 3. Why it works

This strategy is a **bet that today will be a trend day.**

- **Trend days** (SPY moves ≥ 0.7 ATR from open to close): trades average **+1.09 R**.
- **Range days:** trades average **−0.43 R**.

Many small losses on quiet days are paid for by a few large wins on strong days. That's why the win rate is only 42 % while the average win is almost twice the average loss. Each rule is designed to tilt the odds toward catching trend days:

| Rule | Rationale (hypothesis consistent with the data) |
|---|---|
| Gap and opening candle agree | The overnight move (news, positioning) is being *confirmed* by the open, not faded. That points to a persistent order imbalance. |
| Narrow opening range | Compression before expansion. It also keeps the stop small relative to the potential move. |
| Bar-close confirmation + small buffer | Filters out wick-only false breakouts. |
| Stop at OR midpoint | Gives the breakout room to breathe; a return to mid-range means the thesis has failed. |
| Breakeven at +2R, no target | Protects winners without capping the rare large trend-day gains that produce the edge. |
| Exit by 14:00 | Captures the morning-to-midday trend and avoids the afternoon chop and FOMC-statement minute. |

**When it does best and worst** (mean R per trade):
- **Best:** high volatility (SPY 20-day vol 20–30 %: +0.37 R; > 30 %: +0.49 R), SPY below its 200-day average (+0.37 R), large gaps > 1 ATR (+0.41 R), SPY & QQQ (+0.26 R each).
- **Worst:** calm rising markets (vol < 12 %: +0.17 R, the most common regime), late entries 10:30–11:30 (+0.10 R), wider ranges 0.35–0.5 ATR (+0.10 R), DIA (+0.17 R).
- **Long and short are equal** (+0.23 R vs +0.22 R), so it isn't simply riding the bull market.

---

## 4. How to trade it, step by step

> **Paper-trade first** for at least 2–3 months, and compare your fills with the backtest's assumptions.

### Before the open (by 09:25 ET)
1. For each of SPY, QQQ, IWM, DIA, compute **ATR(14)** from completed daily bars and note **yesterday's close**.
2. Check the economic calendar. *Optional, unconfirmed:* skip the **day before an FOMC statement** (see [§6.9](#69-news--high-impact-events)).
3. Confirm account equity and that you're above the **$25,000 PDT minimum** with a cushion.

### 09:30 – 09:45: build the opening range
4. Record the **09:30 open**, then track the high and low through the **09:44** bar.
5. At 09:45 compute for each ETF:
   - direction = sign(09:44 close − 09:30 open)
   - gap = today's 09:30 open − yesterday's close → must have the **same sign** as direction, and \|gap\| ≥ 0.3 × ATR
   - OR width = OR high − OR low → must be ≤ 0.5 × ATR
6. If any check fails, **no trade in that ETF today**. Otherwise compute:
   - **trigger** = OR high + 0.02 × ATR (long) or OR low − 0.02 × ATR (short)
   - **stop** = (OR high + OR low) / 2
   - **R per share** = |trigger − stop| (recomputed at actual entry)

### 09:45 – 11:30: entry
7. Watch **1-minute closes**. On the first bar that closes beyond the trigger, **enter immediately** (marketable limit or market order).
8. Size: `shares = floor(min(0.01 × equity / R_per_share, remaining_4x_notional / price))`.
9. Immediately place a **stop order at the OR midpoint**.
10. If no close beyond the trigger by the **11:30** bar → no trade today.

### While in the trade
11. When price has reached **entry + 2R** (long) or entry − 2R (short) on a completed bar, **move the stop to entry**.
12. Don't add a profit target and don't move the stop otherwise.

### Exit
13. If still open, **close at the 14:00 bar's close** (market order ≈ 14:00–14:01 ET).
14. **FOMC days:** statements are released at 14:00 ET, the same minute as the exit. Consider exiting at **13:59** on FOMC days to avoid announcement-minute fills. This is a practical precaution, not a tested rule.

### Broker & costs assumed in the backtest
- Commission **$0.0035/share/side** (IBKR Pro tiered; verify current), plus **1 tick ($0.01) slippage** on every market/stop fill.
- Intraday margin up to **4×** (Reg-T day-trading buying power). Pattern-day-trader rules apply below $25,000.

---

## 5. Risk management & monitoring

### Built into the strategy
- **Hard stop** on every trade (OR midpoint) and **breakeven** at +2R.
- **No overnight risk:** flat by 14:01 every day.
- **Fixed-fractional sizing:** 1 % of start-of-day equity per trade, **4× gross-leverage cap**, at most 4 concurrent positions.

### Risk controls that were tested and *didn't* help
A daily loss limit, a drawdown throttle and an equity-curve filter changed results little, or cut returns more than risk. Capping trades per day and lowering risk per trade reduced drawdowns roughly in proportion to returns.

### What to expect (Monte Carlo, 10,000 block-bootstrapped paths from 2017–2026)
| Horizon | Median CAGR | 5th–95th pct CAGR | P(loss) | P(beat SPY) | Median max DD | 1-in-20 worst DD |
|---|---|---|---|---|---|---|
| 1 year | 20.1 % | −2.8 % to 56.4 % | 7.8 % | 57 % | −7.8 % | −13.9 % |
| 5 years | 20.9 % | 9.4 % to 35.3 % | 0.1 % | 70 % | −12.6 % | −20.4 % |
| 5 years, edge −0.10 R/trade | 11.6 % | 0.9 % to 24.9 % | 3.6 % | 40 % | −17.1 % | −29.1 % |

### Monitoring / kill-switch (suggested)
- **Track live R per trade.** The backtest expects about +0.25 R. If the rolling 100-trade mean falls below **+0.10 R**, the edge has decayed by more than the −0.10 R stress scenario. Stop and review.
- **Drawdown:** a drawdown deeper than **−20 %** is worse than 95 % of simulated 5-year paths. Stop and review.
- **Fill quality:** log the difference between your fills and the bar-close entry. More than 1 tick of slippage on average erodes the edge quickly.
- **PDT cushion:** about 85 % of simulated paths dip below their starting equity at some point. Starting exactly at $25,000 risks losing day-trading permission. **Start with a cushion (≥ $30k)** or trade a cash account.

---

## 6. The research, start to finish

### 6.1 Audit of the original notebook
The first version reported strong results that turned out to be artefacts. The audit found 12 defects:
- **Look-ahead bias:** trend filters used *today's* close, and several features and filters used same-day data.
- **Selection bias:** the configuration was chosen by out-of-sample Sharpe.
- **Inflated Sharpe:** Sharpe was computed over trade days only.
- **Optimistic fills:** targets filled on a touch and won same-bar ties.
- **A corrupted NQ futures series:** 157 contract months and spreads were mixed together.
- **An ETF data-source change in 2022** that silently broke volume filters.

All were fixed. The NQ futures data was later dropped, so the research is ETF-only with SPY as the benchmark.

### 6.2 Paper replication: Zarattini, Barbon & Aziz (2024)
*"A Profitable Day Trading Strategy for the U.S. Equity Market"* (SSRN 4729284): 5-minute ORB, 10 % ATR stop, close at end of day, "Stocks in Play" relative-volume filter. The paper reports a Sharpe of 2.81 on 7,000 US stocks.

| Paper period 2016–2023, on 11 ETFs | Sharpe |
|---|---|
| 5m ORB + Relative Volume, zero slippage (as in the paper) | −0.00 |
| Same with 1-tick slippage | **−0.76** |
| SPY buy & hold | 0.72 |

**Why it fails on ETFs:**
- **ETFs rarely become "in play".** Relative volume ≥ 3× occurs on about 1.5 % of ETF days, while the paper's edge comes from news-driven single stocks at 10–30×.
- **There are too few independent bets.** About 2 correlated ETF trades a day, versus about 20 unrelated stock trades in the paper.
- **A 10 % ATR stop on an ETF is 5–38 cents.** Commission and one tick of slippage then consume most of the edge.

### 6.3 Variation search: 13,440 configurations
The search varied opening-range length, direction rule, entry buffer and confirmation, entry cut-off, stop type and size, breakeven, trailing stop, target, time exit, and filters (relative volume, range width, gap, volatility regime, NR7, universe), plus risk-management overlays.
- **Selection used 2003–2015 only.** 2016+ was held out.
- **The rank correlation between in-sample and out-of-sample performance was 0.61.** Families of settings were persistently good or bad.
- **The parameter effects that held both in and out of sample:** wider stops (0.25–0.5 ATR or range-based) instead of tight ones, gap-confirmed direction, bar-close entries, no profit targets, and equity ETFs only. Relative-volume filters did **not** help.
- **Walk-forward re-selection** (pick the best on all prior years, trade the next year) produced a 2016+ Sharpe of 1.03. From 2017 it chose the winning configuration every year.

### 6.4 Analysis of the winner
The winner is a trend-day strategy (see [§3](#3-why-it-works)). The best and worst conditions are listed there.

### 6.5 Winner vs SPY
See [§1](#1-results-at-a-glance). A 70 % SPY / 30 % ORB blend kept 94 % of SPY's correlation and most of its crash drawdowns. The ORB is most valuable as a **separate, uncorrelated** return stream.

### 6.6 Statistical tests
- **One-sample t-tests,** run three ways: per trade, per day (same-day trades combined), and on daily returns with Newey-West standard errors.
- **Tests against SPY:** a paired return test and the Jobson–Korkie/Memmel Sharpe test.
- **An in-sample vs live comparison** with Welch and Mann–Whitney tests.

Results are in [§1](#1-results-at-a-glance).

### 6.7 Machine learning: a Random Forest to skip or double trades
- **Setup.** A walk-forward Random Forest using only information known at entry: gap, range width, stop distance, entry time, relative volume, volatility regime, SPY trend, ETF and day of week.
- **No predictive skill.** Out-of-sample AUC was **0.51**, where 0.5 is a coin flip.
- **Skipping low-probability trades** cut the live Sharpe from 1.33 to 0.99.
- **Doubling only high-probability trades** ranked at the **33rd percentile** of random doubling.
- **Conclusion:** at entry time, nothing reliably predicts which days will trend. **Trade every signal.**

### 6.8 Monte Carlo
Stationary block bootstrap (about 20-day blocks) of 2017–2026 daily returns, with SPY sampled on the same days, plus edge-decay stress tests. Results are in [§5](#5-risk-management--monitoring).

### 6.9 News & high-impact events
- **The calendar.** It covers 2003–2026 and was rebuilt from official sources, because Forex Factory blocks automated access:
  - FRED release dates for NFP, CPI, PPI, PCE, GDP, Retail Sales and JOLTS
  - federalreserve.gov for FOMC statements (197, including emergency statements)
  - a publication rule for ISM PMIs
- **FOMC days are trend days.** 36 % of them trend, versus 21 % of days with no news. The winner still breaks even there, because its 14:00 exit coincides with the statement.
- **The day before an FOMC statement** was the one significant effect: −0.40 R per trade (t = −5.0). It was consistent in-sample (−0.38 R) and live (−0.43 R). Pre-FOMC sessions are quiet, with only 17 % trend days.
- **The day after FOMC** was +0.53 R per trade, but not significant after correction.
- **News-specific strategy variants:** 1,729 configurations × 9 news contexts were tested with placebo-calendar bootstraps. No context had a credible variant of its own. A "news playbook" that switched configurations by news type did **worse** live (Sharpe 1.06 vs 1.33).
- **Conclusion:** trade through all news. **Skipping the day before FOMC** is the one candidate rule. Confirm it on new data before relying on it.

---

## 7. Methodology & anti-overfitting safeguards

| Safeguard | Implementation |
|---|---|
| No look-ahead | Every feature is shifted to be known before it's used. A self-test corrupts future data and asserts that past features don't change. |
| Conservative fills | The stop wins same-bar ties. Targets need a one-tick trade-through. Stop-order entries are checked for a stop on the entry bar. Gaps through stops fill at the open. Slippage is applied on every market/stop fill. |
| Realistic sizing | Whole shares, start-of-day equity, 4× gross-leverage cap across open positions. |
| Honest metrics | Daily mark-to-market over **every** session (zero-return days included). |
| Out-of-sample protocol | Selection on in-sample data only, with a single out-of-sample evaluation of the chosen configuration. |
| Selection-bias correction | Deflated Sharpe Ratio, rank correlation between in-sample and out-of-sample results, Bonferroni corrections. |
| Walk-forward | Yearly re-selection on prior data only, for both strategy selection and the ML models. |
| Placebo tests | News effects checked against random calendars. ML picks checked against random doubling. |
| Reality checks | Bootstrap confidence intervals, Monte Carlo stress tests, and paired t-tests vs baselines. |

---

## 8. Data

| Item | Detail |
|---|---|
| Instruments | 11 ETFs: CPER, DIA, GLD, IEF, IWM, QQQ, SHY, SLV, SPY, TLT, USO (1-minute OHLCV, dividend-adjusted) |
| Period | 2002-12-30 → 2026-09-21 (5,969 sessions) |
| Location | `/Users/andrew/main/trading/data/etfs/{SYMBOL}_clean_1min.csv` (not in the repo) |
| Known issue | From **2022-03-07** the files switch from `pitrading` (consolidated volume) to `iex` (≈ 2 % of volume, sparser bars). Volume features are computed per data source, and sparse sessions are excluded by a coverage check. |
| News | FRED API (free key), federalreserve.gov FOMC calendars, ISM publication rule. Cached in `data_cache/`. |

---

## 9. Reproducing the research

**Environment:** Python 3.13 (Anaconda) with `numpy`, `pandas`, `scipy`, `scikit-learn`, `plotly`, `pyarrow`, `requests`.

**Notebook order** (`research.ipynb`):
1. Libraries & configuration
2. Data import
3. Backtest engine (shared by every cell below it)
4. Paper replication (Zarattini et al.)
5. Variation search: 13,440 configurations (≈ 3 min)
6. Winning-strategy analysis
7. Winner vs SPY
8. t-tests
9. Random Forest skip/double experiments
10. Monte Carlo
11. News calendar
12. News-event analysis
13. News bootstrap (4 cells)

**FRED API key** (needed only for the news cells): get a free key at https://fred.stlouisfed.org/docs/api/api_key.html and store it **outside the repo**:
```bash
echo "YOUR_KEY_HERE" > ~/.fred_api_key && chmod 600 ~/.fred_api_key
```

`data_cache/` is regenerated automatically and should not be committed (add it to `.gitignore`).

---

## 10. Limitations & open questions
- **Selection bias.** The winner is the best of about 13,440 configurations. Only 2017–2026 is genuinely out-of-sample, and the search space was designed with knowledge of this dataset.
- **Execution.** The edge per trade is modest. Slippage worse than one tick, partial fills or slow bar-close entries would shrink it, and the Monte Carlo shows −0.10 R per trade pushes it below SPY on most paths.
- **Recent weakness.** Two of the three worst drawdowns (in R) were in 2023–2026, and 2025 was roughly flat. That could be a low-volatility regime or early decay, so monitor live R.
- **Data.** Post-2022 bars are IEX-only. Highs and lows printed on a single venue approximate the consolidated tape.
- **Leverage.** Results at 1 % risk depend on 4× intraday margin.
- **Open questions:**
  - Confirm the *skip pre-FOMC days* rule on new data.
  - Test an FOMC-day 13:59 exit.
  - Paper-trade to measure real fills.
  - Extend to a broad single-stock universe, where the original paper's "Stocks in Play" edge lives.

---

## 11. Project structure
```text
ORB-system/
├── research.ipynb            # the full research notebook (engine, experiments, results)
├── README.md                 # this file
├── artifacts/
│   └── ssrn-4729284.pdf      # Zarattini, Barbon & Aziz (2024)
└── data_cache/               # auto-generated parquet / CSV caches (do not commit)
```

---

## Research philosophy

> **A more complicated model is not automatically a better model.**

A strategy that performs exceptionally well on historical data but fails under walk-forward testing is not a successful result. This project's history illustrates that. The original notebook's impressive numbers came from look-ahead bias and selection bias. The paper's strategy didn't transfer to ETFs. The machine-learning and news-specific refinements made results *worse* out of sample. The one thing that survived every test was a simple rule set, validated on data it had never seen.

**Data integrity → Reproducibility → Validation → Robustness → Performance**, not **Complexity → Optimization → Backtest → Claim**.

---

## Disclaimer
This repository is a **research project**, not a trading system or financial product, and nothing here is investment advice. Backtested and simulated results have inherent limitations, and future performance may differ materially. Trading with leverage can result in losses exceeding the initial investment.

---

## Author
**Andrew Kozak**: Machine Learning · Quantitative Finance · AI Systems · Software Engineering
GitHub: [@andrewjkdev](https://github.com/andrewjkdev)

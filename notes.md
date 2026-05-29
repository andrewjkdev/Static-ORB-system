// TASK: We are going to build two statsitical proxies; a volume profile with POC proxy, and a delta imbalance proxy from the indicators in module.2a

// GOAL: have a proxy for volume profile POC, and delta imbalance that can be used to train a model to predict the volume profile and delta imbalance for a full 16 years of trading data.

// proxy 1: Volume Profile with POC proxy
// proxy 2: Delta Imbalance proxy
****


// Module.2b
proxy_1::{
    // TASK: use the 8‑month MBO data to learn a mapping from OHLCV shape → true POC, then apply that mapping to the full 16‑year history.
    
    // PREDICTIVE TARGETS:
        // From each trading day’s MBO data you will compute:
        // mbo_poc: the single price level with the highest traded volume (Point of Control)
        // mbo_value_area_high / mbo_value_area_low: boundaries of the 70% value area
        // mbo_profile_skew: (volume above POC) / (volume below POC) — a measure of bias
        // These become your y targets for the proxy model.

    // Cellblock.1: Build a daily OHLCV feature matrix (one row per day)
        // For each day in the overlapping period, compute aggregate features from the 1‑minute OHLCV bars. 
        // These describe the day’s price action, volume, and volatility.

    // Cellblock.2: Build the training dataset (overlap period)
        // Loop over every day present in both OHLCV and MBO.
        // For each day, extract the OHLCV features and the true MBO POC (computed exactly in Cellblock.1)
        // Store everything in a data frame. This is our training set.

    // Cellblock.3: Train the proxy model
        // We'll use XGBoost to predict mbo_poc from the OHLCV features. Split chronologically:
            // first 80% of dates for training
            // last 20% for out‑of‑sample validation
            // Never random split.

    // Cellblock.4: Apply to the full 16‑year dataset
        // Now loop over every day in the OHLCV history (2010-2026)
        // compute the same features
        // predict the proxy POC
        // store it back as a new column

    // Cellblock.5: Sanity check 
        // Plot the proxy POC against the actual POC on a few random days from the validation period to visually confirm it tracks the true POC within a reasonable band.
},

proxy_2::{
    // TASK: The proxy pipeline maps 1-minute OHLCV bar fearures → estimated delta ratio and buy-sell ratio
        // Apply the same  ratio "Ratio > 300% + Δ/Vol > 15%" logic to the full 16-year history.
        // The process is identical in spirit to the POC proxy, but at the minute level.

    // TARGET VARIABLES (per minute, from MBO)
        // For every minute in the overlapping period, you have the ground truth:
            // delta_ratio = |delta| / total_volume
            // buy_sell_ratio = buy_vol / sell_vol (capped to avoid infinity)
        // These are your regression targets. You can also derive a binary target: signal = (buy_sell_ratio > 3) & (delta_ratio > 0.15)

    // Cellblock.6: Build per‑minute OHLCV features
        // Each row in the training set is one minute.
        // The features come only from the 1‑minute OHLCV bar (plus rolling windows).
        // No MBO.

    // Cellblock.7: Build the training dataset (overlap period)
        // Loop over every overlapping day.
        // For each minute that has both OHLCV and MBO data, compute features and attach the targets
        // Store in a dataframe

    // Cellblock.8: Train the proxy model (XGBoost regression)
        // Split by date, not random.
        // Use 80% of dates for training, 20% for validation.
    
    // Cellblock.9: Apply proxy to the full 16-year OHLCV data with a vectorized batch prediction approach for better speed.
        // generate proxy_delta_ratio and proxy_buy_sell_ratio for every minute in the historical dataset.
        // Process day by day to avoid memory overload.
        // Precompute the per‑minute feature matrix for the entire 16‑year dataset (as a DataFrame) using a rolling window operation. 
        // Predict in one shot. This is far faster than per‑minute loops.

    // Cellblock.10: Sanity check
        // Plot a few days from the validation set: overlay true and proxy delta_ratio, and mark the derived signals.
        // The proxy should track the direction and spikes.
}
****


//EXTRA CONTEXT: Do a deep quality assurance check on the entire program and then go over every cellblock in module.2b to make sure the logic is sound and the code is clean and well documented.


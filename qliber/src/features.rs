use std::collections::VecDeque;

use polars::prelude::*;

use crate::logging::log_event;

fn to_f64_vec(series: &Series) -> PolarsResult<Vec<f64>> {
    let float_series = if series.dtype() != &DataType::Float64 {
        series.cast(&DataType::Float64)?
    } else {
        series.clone()
    };

    let chunked = float_series.f64().expect("series casted to f64");
    Ok(chunked.into_iter().map(|opt| opt.unwrap_or(0.0)).collect())
}

/// Compute daily percentage returns from a price column and append them to the DataFrame.
pub fn with_daily_returns(
    frame: &DataFrame,
    price_column: &str,
    output_column: &str,
) -> PolarsResult<DataFrame> {
    let prices = to_f64_vec(frame.column(price_column)?)?;
    if prices.is_empty() {
        return Ok(frame.clone());
    }

    let mut returns = Vec::with_capacity(prices.len());
    returns.push(0.0);
    for window in prices.windows(2) {
        let prev = window[0];
        let current = window[1];
        let pct = if prev.abs() < f64::EPSILON {
            0.0
        } else {
            (current / prev) - 1.0
        };
        returns.push(pct);
    }

    let mut enriched = frame.clone();
    enriched.with_column(Series::new(output_column, returns))?;

    log_event(
        file!(),
        "FeatureEngineering",
        "with_daily_returns",
        "features.returns",
        line!(),
        &format!("Computed daily returns for {price_column} -> {output_column}"),
        None,
        "none",
        "GET",
    );

    Ok(enriched)
}

/// Append a moving average column computed with a numerically stable rolling window.
pub fn with_moving_average(
    frame: &DataFrame,
    price_column: &str,
    window: usize,
    output_column: &str,
) -> PolarsResult<DataFrame> {
    assert!(window > 0, "window size must be positive");
    let prices = to_f64_vec(frame.column(price_column)?)?;
    if prices.is_empty() {
        return Ok(frame.clone());
    }

    let mut averages = Vec::with_capacity(prices.len());
    let mut sum = 0.0;

    for (idx, value) in prices.iter().enumerate() {
        sum += value;
        if idx >= window {
            sum -= prices[idx - window];
            averages.push(sum / window as f64);
        } else {
            averages.push(sum / (idx + 1) as f64);
        }
    }

    let mut enriched = frame.clone();
    enriched.with_column(Series::new(output_column, averages))?;

    log_event(
        file!(),
        "FeatureEngineering",
        "with_moving_average",
        "features.moving_average",
        line!(),
        &format!("Computed {window}-period moving average for {price_column} -> {output_column}"),
        None,
        "none",
        "GET",
    );

    Ok(enriched)
}

/// Append a rolling z-score normalization column.
pub fn with_z_score(
    frame: &DataFrame,
    column: &str,
    window: usize,
    output_column: &str,
) -> PolarsResult<DataFrame> {
    assert!(
        window > 1,
        "window size must exceed one to compute z-scores"
    );
    let values = to_f64_vec(frame.column(column)?)?;
    if values.is_empty() {
        return Ok(frame.clone());
    }

    let mut zscores = Vec::with_capacity(values.len());
    let mut window_values: VecDeque<f64> = VecDeque::with_capacity(window);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for value in values.iter() {
        window_values.push_back(*value);
        sum += value;
        sum_sq += value * value;

        if window_values.len() > window
            && let Some(old) = window_values.pop_front()
        {
            sum -= old;
            sum_sq -= old * old;
        }

        let len = window_values.len() as f64;
        let mean = sum / len;
        let variance = (sum_sq / len) - mean * mean;
        let variance = variance.max(0.0);
        let std = variance.sqrt();
        let z = if std > f64::EPSILON {
            (*value - mean) / std
        } else {
            0.0
        };
        zscores.push(z);
    }

    let mut enriched = frame.clone();
    enriched.with_column(Series::new(output_column, zscores))?;

    log_event(
        file!(),
        "FeatureEngineering",
        "with_z_score",
        "features.zscore",
        line!(),
        &format!("Computed {window}-period z-score for {column} -> {output_column}"),
        None,
        "none",
        "GET",
    );

    Ok(enriched)
}

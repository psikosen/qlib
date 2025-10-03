use std::str::FromStr;

use polars::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("unsupported frequency format: {0}")]
    UnsupportedFrequency(String),
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("indicator analysis requires column `{0}`")]
    MissingColumn(String),
    #[error("indicator analysis encountered zero total weight for {0:?}")]
    ZeroWeights(IndicatorMethod),
}

pub type MetricsResult<T> = Result<T, MetricsError>;

/// Supported evaluation frequencies mirroring Qlib's `Freq` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyUnit {
    Minute,
    Day,
    Week,
    Month,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisFrequency {
    count: u32,
    unit: FrequencyUnit,
}

impl AnalysisFrequency {
    pub fn new(count: u32, unit: FrequencyUnit) -> Self {
        let normalized_count = count.max(1);
        Self {
            count: normalized_count,
            unit,
        }
    }

    pub fn unit(&self) -> FrequencyUnit {
        self.unit
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn periods_per_year(&self) -> f64 {
        let scaler = match self.unit {
            FrequencyUnit::Minute => 240.0 * 238.0,
            FrequencyUnit::Day => 238.0,
            FrequencyUnit::Week => 50.0,
            FrequencyUnit::Month => 12.0,
        };
        scaler / self.count as f64
    }
}

impl FromStr for AnalysisFrequency {
    type Err = MetricsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim().to_lowercase();
        let digit_count = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();

        let (count_str, suffix) = trimmed.split_at(digit_count);
        if suffix.is_empty() {
            return Err(MetricsError::UnsupportedFrequency(trimmed));
        }

        let count: u32 = if count_str.is_empty() {
            1
        } else {
            count_str
                .parse()
                .map_err(|_| MetricsError::UnsupportedFrequency(trimmed.clone()))?
        };

        let unit = match suffix {
            "month" | "mon" => FrequencyUnit::Month,
            "week" | "w" => FrequencyUnit::Week,
            "day" | "d" => FrequencyUnit::Day,
            "minute" | "min" => FrequencyUnit::Minute,
            _ => return Err(MetricsError::UnsupportedFrequency(trimmed)),
        };

        Ok(Self::new(count, unit))
    }
}

impl TryFrom<&str> for AnalysisFrequency {
    type Error = MetricsError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_str(value)
    }
}

/// Indicator weighting strategies matching Qlib's `indicator_analysis` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorMethod {
    Mean,
    AmountWeighted,
    ValueWeighted,
}

/// Controls how returns are accumulated when computing performance statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationMode {
    /// Arithmetic accumulation that mirrors Qlib's `mode="sum"` risk analysis.
    Sum,
    /// Geometric accumulation equivalent to Qlib's `mode="product"` risk analysis.
    Product,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceMetrics {
    pub mean_return: f64,
    pub std_dev: f64,
    pub cumulative_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub information_ratio: f64,
    pub max_drawdown: f64,
}

impl PerformanceMetrics {
    pub fn evaluate(returns: &[f64], periods_per_year: f64) -> Self {
        Self::evaluate_with_mode(returns, periods_per_year, AccumulationMode::Product)
    }

    pub fn evaluate_with_mode(
        returns: &[f64],
        periods_per_year: f64,
        mode: AccumulationMode,
    ) -> Self {
        if returns.is_empty() {
            log_event(
                file!(),
                "PerformanceMetrics",
                "evaluate_with_mode",
                "metrics.evaluate",
                line!(),
                "Received empty returns series; returning zeroed metrics",
                None,
                "none",
                "GET",
            );
            return Self {
                mean_return: 0.0,
                std_dev: 0.0,
                cumulative_return: 0.0,
                annualized_return: 0.0,
                annualized_volatility: 0.0,
                sharpe_ratio: 0.0,
                information_ratio: 0.0,
                max_drawdown: 0.0,
            };
        }

        match mode {
            AccumulationMode::Sum => Self::from_sum_mode(returns, periods_per_year),
            AccumulationMode::Product => Self::from_product_mode(returns, periods_per_year),
        }
    }

    pub fn evaluate_with_frequency(
        returns: &[f64],
        frequency: AnalysisFrequency,
        mode: AccumulationMode,
    ) -> Self {
        let periods_per_year = frequency.periods_per_year();
        let metrics = Self::evaluate_with_mode(returns, periods_per_year, mode);

        log_event(
            file!(),
            "PerformanceMetrics",
            "evaluate_with_frequency",
            "metrics.evaluate",
            line!(),
            &format!(
                "Evaluated returns with {:?} frequency (count={}) using {:?} mode",
                frequency.unit(),
                frequency.count(),
                mode
            ),
            None,
            "none",
            "GET",
        );

        metrics
    }

    pub fn evaluate_with_frequency_str(
        returns: &[f64],
        freq: &str,
        mode: AccumulationMode,
    ) -> MetricsResult<Self> {
        let frequency = AnalysisFrequency::try_from(freq)?;
        let metrics = Ok(Self::evaluate_with_frequency(returns, frequency, mode));

        log_event(
            file!(),
            "PerformanceMetrics",
            "evaluate_with_frequency_str",
            "metrics.evaluate",
            line!(),
            &format!("Parsed frequency `{freq}` into {:?}", frequency.unit()),
            None,
            "none",
            "GET",
        );

        metrics
    }

    fn from_sum_mode(returns: &[f64], periods_per_year: f64) -> Self {
        let count = returns.len() as f64;
        let mean = returns.iter().copied().sum::<f64>() / count;
        let variance = sample_variance(returns, mean);
        let std_dev = variance.sqrt();

        let cumulative_return = returns.iter().copied().sum::<f64>();
        let annualized_return = mean * periods_per_year;
        let annualized_volatility = std_dev * periods_per_year.sqrt();

        let mut running_sum = 0.0;
        let mut running_peak = 0.0;
        let mut max_drawdown = 0.0;
        for value in returns {
            running_sum += value;
            if running_sum > running_peak {
                running_peak = running_sum;
            }
            let drawdown = running_sum - running_peak;
            if drawdown < max_drawdown {
                max_drawdown = drawdown;
            }
        }

        let scaling = periods_per_year.sqrt();
        let information_ratio = if std_dev > f64::EPSILON {
            (mean / std_dev) * scaling
        } else {
            0.0
        };

        log_event(
            file!(),
            "PerformanceMetrics",
            "from_sum_mode",
            "metrics.evaluate",
            line!(),
            "Computed arithmetic accumulation performance statistics",
            None,
            "none",
            "GET",
        );

        Self {
            mean_return: mean,
            std_dev,
            cumulative_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio: information_ratio,
            information_ratio,
            max_drawdown,
        }
    }

    fn from_product_mode(returns: &[f64], periods_per_year: f64) -> Self {
        let mut cumulative_curve = Vec::with_capacity(returns.len());
        let mut cumulative_product = 1.0;
        for value in returns {
            cumulative_product *= 1.0 + value;
            cumulative_curve.push(cumulative_product);
        }

        let final_value = *cumulative_curve.last().unwrap_or(&1.0);
        let count = returns.len() as f64;
        let cumulative_return = final_value - 1.0;
        let mean = if count > 0.0 {
            final_value.powf(1.0 / count) - 1.0
        } else {
            0.0
        };

        let log_returns: Vec<f64> = returns
            .iter()
            .filter_map(|r| {
                let base = 1.0 + r;
                if base.is_sign_positive() && base > f64::EPSILON {
                    Some(base.ln())
                } else {
                    log_event(
                        file!(),
                        "PerformanceMetrics",
                        "from_product_mode",
                        "metrics.evaluate",
                        line!(),
                        &format!(
                            "Encountered non-positive gross return {:.6}; excluding from log std calculation",
                            base
                        ),
                        None,
                        "none",
                        "GET",
                    );
                    None
                }
            })
            .collect();

        let std_dev = if log_returns.len() > 1 {
            let log_mean = log_returns.iter().copied().sum::<f64>() / log_returns.len() as f64;
            sample_variance(&log_returns, log_mean).sqrt()
        } else {
            0.0
        };

        let annualized_return = if count > 0.0 {
            (1.0 + cumulative_return).powf(periods_per_year / count) - 1.0
        } else {
            0.0
        };
        let annualized_volatility = std_dev * periods_per_year.sqrt();

        let mut max_drawdown = 0.0;
        let mut peak = cumulative_curve.first().copied().unwrap_or(1.0);
        for value in &cumulative_curve {
            if *value > peak {
                peak = *value;
            }
            let drawdown = (value / peak) - 1.0;
            if drawdown < max_drawdown {
                max_drawdown = drawdown;
            }
        }

        let scaling = periods_per_year.sqrt();
        let information_ratio = if std_dev > f64::EPSILON {
            (mean / std_dev) * scaling
        } else {
            0.0
        };

        log_event(
            file!(),
            "PerformanceMetrics",
            "from_product_mode",
            "metrics.evaluate",
            line!(),
            "Computed geometric accumulation performance statistics",
            None,
            "none",
            "GET",
        );

        Self {
            mean_return: mean,
            std_dev,
            cumulative_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio: information_ratio,
            information_ratio,
            max_drawdown,
        }
    }
}

pub fn indicator_analysis(frame: &DataFrame, method: IndicatorMethod) -> MetricsResult<DataFrame> {
    let count_weights = match require_column(frame, "count") {
        Ok(column) => column,
        Err(error) => {
            log_event(
                file!(),
                "PerformanceMetrics",
                "indicator_analysis",
                "metrics.indicator",
                line!(),
                "Missing required `count` column for indicator analysis",
                Some(&error.to_string()),
                "none",
                "GET",
            );
            return Err(error);
        }
    };

    let weights = match method {
        IndicatorMethod::Mean => count_weights.clone(),
        IndicatorMethod::AmountWeighted => match require_column(frame, "deal_amount") {
            Ok(column) => column,
            Err(error) => {
                log_event(
                    file!(),
                    "PerformanceMetrics",
                    "indicator_analysis",
                    "metrics.indicator",
                    line!(),
                    "Missing `deal_amount` column required for amount-weighted analysis",
                    Some(&error.to_string()),
                    "none",
                    "GET",
                );
                return Err(error);
            }
        },
        IndicatorMethod::ValueWeighted => match require_column(frame, "value") {
            Ok(column) => column,
            Err(error) => {
                log_event(
                    file!(),
                    "PerformanceMetrics",
                    "indicator_analysis",
                    "metrics.indicator",
                    line!(),
                    "Missing `value` column required for value-weighted analysis",
                    Some(&error.to_string()),
                    "none",
                    "GET",
                );
                return Err(error);
            }
        },
    };

    let ffr_values = require_column(frame, "ffr").inspect_err(|error| {
        log_event(
            file!(),
            "PerformanceMetrics",
            "indicator_analysis",
            "metrics.indicator",
            line!(),
            "Missing `ffr` column required for indicator analysis",
            Some(&error.to_string()),
            "none",
            "GET",
        );
    })?;
    let pa_values = require_column(frame, "pa").inspect_err(|error| {
        log_event(
            file!(),
            "PerformanceMetrics",
            "indicator_analysis",
            "metrics.indicator",
            line!(),
            "Missing `pa` column required for indicator analysis",
            Some(&error.to_string()),
            "none",
            "GET",
        );
    })?;
    let pos_values = require_column(frame, "pos").inspect_err(|error| {
        log_event(
            file!(),
            "PerformanceMetrics",
            "indicator_analysis",
            "metrics.indicator",
            line!(),
            "Missing `pos` column required for indicator analysis",
            Some(&error.to_string()),
            "none",
            "GET",
        );
    })?;

    let ffr = weighted_average(&ffr_values, &weights, method, "ffr")?;
    let pa = weighted_average(&pa_values, &weights, method, "pa")?;
    let pos = weighted_average(&pos_values, &count_weights, IndicatorMethod::Mean, "pos")?;

    let indicators = Series::new("indicator", &["ffr", "pa", "pos"]);
    let values = Series::new("value", &[ffr, pa, pos]);
    let result = DataFrame::new(vec![indicators, values])?;

    log_event(
        file!(),
        "PerformanceMetrics",
        "indicator_analysis",
        "metrics.indicator",
        line!(),
        &format!("Computed indicator analysis using {:?} weighting", method),
        None,
        "none",
        "GET",
    );

    Ok(result)
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let sum_squares = values
        .par_iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>();
    sum_squares / (values.len() as f64 - 1.0)
}

fn require_column(frame: &DataFrame, name: &str) -> MetricsResult<Float64Chunked> {
    let series = frame
        .column(name)
        .map_err(|_| MetricsError::MissingColumn(name.to_string()))?;
    series_to_f64(series).map_err(Into::into)
}

fn series_to_f64(series: &Series) -> PolarsResult<Float64Chunked> {
    if matches!(series.dtype(), DataType::Float64) {
        Ok(series.f64()?.clone())
    } else {
        let casted = series.cast(&DataType::Float64)?;
        Ok(casted.f64().expect("series cast to f64").clone())
    }
}

fn weighted_average(
    values: &Float64Chunked,
    weights: &Float64Chunked,
    method: IndicatorMethod,
    indicator: &str,
) -> MetricsResult<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (value, weight) in values.into_iter().zip(weights.into_iter()) {
        if let (Some(v), Some(w)) = (value, weight)
            && v.is_finite()
            && w.is_finite()
        {
            let weight_value = match method {
                IndicatorMethod::AmountWeighted | IndicatorMethod::ValueWeighted => w.abs(),
                IndicatorMethod::Mean => w,
            };

            if weight_value.is_finite() {
                numerator += v * weight_value;
                denominator += weight_value;
            }
        }
    }

    if denominator.abs() <= f64::EPSILON {
        log_event(
            file!(),
            "PerformanceMetrics",
            "weighted_average",
            "metrics.indicator",
            line!(),
            &format!(
                "Encountered zero total weight while computing {indicator} with {:?} weighting",
                method
            ),
            None,
            "none",
            "GET",
        );
        return Err(MetricsError::ZeroWeights(method));
    }

    Ok(numerator / denominator)
}

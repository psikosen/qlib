use rayon::prelude::*;

use crate::logging::log_event;

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

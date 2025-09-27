use rayon::prelude::*;

use crate::logging::log_event;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceMetrics {
    pub cumulative_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
}

impl PerformanceMetrics {
    pub fn evaluate(returns: &[f64], periods_per_year: f64) -> Self {
        if returns.is_empty() {
            log_event(
                file!(),
                "PerformanceMetrics",
                "evaluate",
                "metrics.evaluate",
                line!(),
                "Received empty returns series; returning zeroed metrics",
                None,
                "none",
                "GET",
            );
            return Self {
                cumulative_return: 0.0,
                annualized_return: 0.0,
                annualized_volatility: 0.0,
                sharpe_ratio: 0.0,
            };
        }

        let growth = returns
            .par_iter()
            .map(|r| 1.0 + r)
            .reduce(|| 1.0, |acc, value| acc * value);
        let cumulative_return = growth - 1.0;

        let count = returns.len() as f64;
        let mean = returns.par_iter().copied().sum::<f64>() / count;
        let variance = returns
            .par_iter()
            .map(|r| {
                let diff = r - mean;
                diff * diff
            })
            .sum::<f64>()
            / count.max(1.0);
        let std_dev = variance.sqrt();

        let annualized_return = if count > 0.0 {
            growth.powf(periods_per_year / count) - 1.0
        } else {
            0.0
        };
        let annualized_volatility = std_dev * periods_per_year.sqrt();
        let sharpe_ratio = if std_dev > f64::EPSILON {
            (mean * periods_per_year) / (std_dev * periods_per_year.sqrt())
        } else {
            0.0
        };

        log_event(
            file!(),
            "PerformanceMetrics",
            "evaluate",
            "metrics.evaluate",
            line!(),
            "Computed performance statistics",
            None,
            "none",
            "GET",
        );

        Self {
            cumulative_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
        }
    }
}

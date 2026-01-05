use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{
    AccumulationMode, AnalysisFrequency, FrequencyUnit, IndicatorMethod, PerformanceMetrics,
    indicator_analysis, indicator_analysis_with_method, risk_analysis,
};
use std::str::FromStr;

#[test]
fn test_frequency_unit_parsing() {
    // Test day frequencies
    let freq = AnalysisFrequency::from_str("1day").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Day);
    assert_eq!(freq.count(), 1);

    let freq = AnalysisFrequency::from_str("5d").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Day);
    assert_eq!(freq.count(), 5);

    // Test week frequencies
    let freq = AnalysisFrequency::from_str("1week").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Week);
    assert_eq!(freq.count(), 1);

    let freq = AnalysisFrequency::from_str("2w").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Week);
    assert_eq!(freq.count(), 2);

    // Test month frequencies
    let freq = AnalysisFrequency::from_str("1month").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Month);
    assert_eq!(freq.count(), 1);

    let freq = AnalysisFrequency::from_str("3mon").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Month);
    assert_eq!(freq.count(), 3);

    // Test minute frequencies
    let freq = AnalysisFrequency::from_str("15minute").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Minute);
    assert_eq!(freq.count(), 15);

    let freq = AnalysisFrequency::from_str("30min").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Minute);
    assert_eq!(freq.count(), 30);
}

#[test]
fn test_frequency_unit_parsing_no_number() {
    // When no number is provided, default to 1
    let freq = AnalysisFrequency::from_str("day").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Day);
    assert_eq!(freq.count(), 1);

    let freq = AnalysisFrequency::from_str("week").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Week);
    assert_eq!(freq.count(), 1);
}

#[test]
fn test_frequency_periods_per_year() {
    let freq = AnalysisFrequency::new(1, FrequencyUnit::Day);
    assert_relative_eq!(freq.periods_per_year(), 238.0, epsilon = 1e-10);

    let freq = AnalysisFrequency::new(1, FrequencyUnit::Week);
    assert_relative_eq!(freq.periods_per_year(), 50.0, epsilon = 1e-10);

    let freq = AnalysisFrequency::new(1, FrequencyUnit::Month);
    assert_relative_eq!(freq.periods_per_year(), 12.0, epsilon = 1e-10);

    let freq = AnalysisFrequency::new(1, FrequencyUnit::Minute);
    assert_relative_eq!(freq.periods_per_year(), 240.0 * 238.0, epsilon = 1e-10);
}

#[test]
fn test_frequency_zero_count() {
    // Zero count should be normalized to 1
    let freq = AnalysisFrequency::new(0, FrequencyUnit::Day);
    assert_eq!(freq.count(), 1);
}

#[test]
fn test_accumulation_mode_from_str() {
    let mode = AccumulationMode::from_str("sum").unwrap();
    assert_eq!(mode, AccumulationMode::Sum);

    let mode = AccumulationMode::from_str("product").unwrap();
    assert_eq!(mode, AccumulationMode::Product);

    let mode = AccumulationMode::from_str("SUM").unwrap();
    assert_eq!(mode, AccumulationMode::Sum);

    let mode = AccumulationMode::from_str("PRODUCT").unwrap();
    assert_eq!(mode, AccumulationMode::Product);
}

#[test]
fn test_accumulation_mode_default() {
    let mode = AccumulationMode::default();
    assert_eq!(mode, AccumulationMode::Sum);
}

#[test]
fn test_indicator_method_from_str() {
    let method = IndicatorMethod::from_str("mean").unwrap();
    assert_eq!(method, IndicatorMethod::Mean);

    let method = IndicatorMethod::from_str("amount_weighted").unwrap();
    assert_eq!(method, IndicatorMethod::AmountWeighted);

    let method = IndicatorMethod::from_str("value_weighted").unwrap();
    assert_eq!(method, IndicatorMethod::ValueWeighted);

    let method = IndicatorMethod::from_str("MEAN").unwrap();
    assert_eq!(method, IndicatorMethod::Mean);
}

#[test]
fn test_performance_metrics_basic() {
    let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // Basic sanity checks
    assert!(metrics.mean_return.is_finite());
    assert!(metrics.std_dev >= 0.0);
    assert!(metrics.cumulative_return.is_finite());
    assert!(metrics.annualized_return.is_finite());
    assert!(metrics.annualized_volatility >= 0.0);
    assert!(metrics.sharpe_ratio.is_finite());
    assert!(metrics.information_ratio.is_finite());
    assert!(metrics.max_drawdown <= 0.0);
}

#[test]
fn test_performance_metrics_sum_mode() {
    let returns = vec![0.10, 0.05, -0.02, 0.03];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate_with_mode(
        &returns,
        periods_per_year,
        AccumulationMode::Sum,
    );

    // Cumulative return should be sum of returns in sum mode
    let expected_cumulative = 0.10 + 0.05 - 0.02 + 0.03;
    assert_relative_eq!(metrics.cumulative_return, expected_cumulative, epsilon = 1e-10);

    // Mean return
    let expected_mean = expected_cumulative / 4.0;
    assert_relative_eq!(metrics.mean_return, expected_mean, epsilon = 1e-10);
}

#[test]
fn test_performance_metrics_product_mode() {
    let returns = vec![0.10, 0.05, -0.02, 0.03];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate_with_mode(
        &returns,
        periods_per_year,
        AccumulationMode::Product,
    );

    // Cumulative return should use geometric accumulation
    let expected_final = (1.0 + 0.10) * (1.0 + 0.05) * (1.0 - 0.02) * (1.0 + 0.03);
    let expected_cumulative = expected_final - 1.0;
    assert_relative_eq!(metrics.cumulative_return, expected_cumulative, epsilon = 1e-10);

    // Mean should be geometric mean
    assert!(metrics.mean_return.is_finite());
}

#[test]
fn test_performance_metrics_empty_returns() {
    let returns: Vec<f64> = vec![];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // Should return zeroed metrics for empty input
    assert_eq!(metrics.mean_return, 0.0);
    assert_eq!(metrics.std_dev, 0.0);
    assert_eq!(metrics.cumulative_return, 0.0);
    assert_eq!(metrics.annualized_return, 0.0);
    assert_eq!(metrics.annualized_volatility, 0.0);
    assert_eq!(metrics.sharpe_ratio, 0.0);
    assert_eq!(metrics.max_drawdown, 0.0);
}

#[test]
fn test_performance_metrics_with_nan() {
    let returns = vec![0.01, f64::NAN, 0.02, 0.03];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // Should filter out NaN and compute on valid values
    assert!(metrics.mean_return.is_finite());
    assert!(metrics.std_dev.is_finite());
}

#[test]
fn test_performance_metrics_with_infinity() {
    let returns = vec![0.01, f64::INFINITY, 0.02, 0.03];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // Should filter out infinity and compute on valid values
    assert!(metrics.mean_return.is_finite());
    assert!(metrics.std_dev.is_finite());
}

#[test]
fn test_performance_metrics_max_drawdown() {
    // Simulate a clear drawdown scenario
    let returns = vec![0.10, 0.05, -0.15, -0.10, 0.05];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate_with_mode(
        &returns,
        periods_per_year,
        AccumulationMode::Sum,
    );

    // Max drawdown should be negative
    assert!(metrics.max_drawdown < 0.0);
}

#[test]
fn test_performance_metrics_all_positive_returns() {
    let returns = vec![0.01, 0.02, 0.03, 0.04];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // No drawdown with all positive returns
    assert_eq!(metrics.max_drawdown, 0.0);
}

#[test]
fn test_performance_metrics_with_frequency() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let frequency = AnalysisFrequency::new(1, FrequencyUnit::Day);

    let metrics = PerformanceMetrics::evaluate_with_frequency(
        &returns,
        frequency,
        AccumulationMode::Sum,
    );

    assert!(metrics.mean_return.is_finite());
    assert!(metrics.annualized_return.is_finite());
}

#[test]
fn test_performance_metrics_with_frequency_str() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];

    let result = PerformanceMetrics::evaluate_with_frequency_str(
        &returns,
        "day",
        AccumulationMode::Sum,
    );

    assert!(result.is_ok());
    let metrics = result.unwrap();
    assert!(metrics.mean_return.is_finite());
}

#[test]
fn test_performance_metrics_to_risk_dataframe() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);
    let df = metrics.to_risk_dataframe().unwrap();

    // Should have metric names and risk values
    assert!(df.column("metric").is_ok());
    assert!(df.column("risk").is_ok());

    // Should have 5 rows (mean, std, annualized_return, information_ratio, max_drawdown)
    assert_eq!(df.height(), 5);
}

#[test]
fn test_risk_analysis_with_scaler() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let df = risk_analysis(&returns, Some(252.0), None, None).unwrap();

    assert!(df.column("metric").is_ok());
    assert!(df.column("risk").is_ok());
    assert_eq!(df.height(), 5);
}

#[test]
fn test_risk_analysis_with_frequency() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let df = risk_analysis(&returns, None, Some("day"), None).unwrap();

    assert!(df.column("metric").is_ok());
    assert!(df.column("risk").is_ok());
}

#[test]
fn test_risk_analysis_with_mode() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let df = risk_analysis(&returns, Some(252.0), None, Some("product")).unwrap();

    assert!(df.column("metric").is_ok());
    assert!(df.column("risk").is_ok());
}

#[test]
fn test_risk_analysis_sum_mode() {
    let returns = vec![0.01, 0.02, -0.01, 0.03];
    let df = risk_analysis(&returns, Some(252.0), None, Some("sum")).unwrap();

    assert!(df.column("metric").is_ok());
    assert!(df.column("risk").is_ok());
}

#[test]
fn test_indicator_analysis_mean() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::Mean).unwrap();

    assert!(result.column("indicator").is_ok());
    assert!(result.column("value").is_ok());
    assert_eq!(result.height(), 3); // ffr, pa, pos
}

#[test]
fn test_indicator_analysis_amount_weighted() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "deal_amount" => &[100.0, 200.0, 300.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::AmountWeighted).unwrap();

    assert!(result.column("indicator").is_ok());
    assert!(result.column("value").is_ok());
}

#[test]
fn test_indicator_analysis_value_weighted() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "value" => &[1000.0, 2000.0, 3000.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::ValueWeighted).unwrap();

    assert!(result.column("indicator").is_ok());
    assert!(result.column("value").is_ok());
}

#[test]
fn test_indicator_analysis_with_method_string() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis_with_method(&df, "mean").unwrap();

    assert!(result.column("indicator").is_ok());
    assert!(result.column("value").is_ok());
}

#[test]
fn test_indicator_analysis_missing_count_column() {
    let df = df! {
        "ffr" => &[0.5, 0.6, 0.7]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::Mean);
    assert!(result.is_err());
}

#[test]
fn test_indicator_analysis_missing_ffr_column() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::Mean);
    assert!(result.is_err());
}

#[test]
fn test_indicator_analysis_missing_deal_amount() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::AmountWeighted);
    assert!(result.is_err());
}

#[test]
fn test_indicator_analysis_missing_value_column() {
    let df = df! {
        "count" => &[10.0, 20.0, 30.0],
        "ffr" => &[0.5, 0.6, 0.7],
        "pa" => &[1.2, 1.3, 1.4],
        "pos" => &[0.3, 0.4, 0.5]
    }
    .unwrap();

    let result = indicator_analysis(&df, IndicatorMethod::ValueWeighted);
    assert!(result.is_err());
}

#[test]
fn test_performance_metrics_single_return() {
    let returns = vec![0.05];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    assert_relative_eq!(metrics.mean_return, 0.05, epsilon = 1e-10);
    assert_eq!(metrics.std_dev, 0.0); // Only one sample, variance is 0
    assert_relative_eq!(metrics.cumulative_return, 0.05, epsilon = 1e-10);
}

#[test]
fn test_performance_metrics_zero_volatility() {
    let returns = vec![0.02, 0.02, 0.02, 0.02];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    assert_relative_eq!(metrics.mean_return, 0.02, epsilon = 1e-10);
    assert_eq!(metrics.std_dev, 0.0); // All returns are the same
    assert_eq!(metrics.sharpe_ratio, 0.0); // Sharpe is 0 when std is 0
}

#[test]
fn test_performance_metrics_negative_returns() {
    let returns = vec![-0.01, -0.02, -0.03, -0.01];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    assert!(metrics.mean_return < 0.0);
    assert!(metrics.cumulative_return < 0.0);
    assert!(metrics.annualized_return < 0.0);
    assert!(metrics.max_drawdown < 0.0);
}

#[test]
fn test_frequency_try_from() {
    let freq = AnalysisFrequency::try_from("week").unwrap();
    assert_eq!(freq.unit(), FrequencyUnit::Week);
    assert_eq!(freq.count(), 1);
}

#[test]
fn test_invalid_frequency_string() {
    let result = AnalysisFrequency::from_str("invalid");
    assert!(result.is_err());
}

#[test]
fn test_invalid_accumulation_mode() {
    let result = AccumulationMode::from_str("invalid");
    assert!(result.is_err());
}

#[test]
fn test_invalid_indicator_method() {
    let result = IndicatorMethod::from_str("invalid");
    assert!(result.is_err());
}

#[test]
fn test_performance_metrics_sharpe_calculation() {
    let returns = vec![0.01, 0.02, 0.01, 0.03, 0.02];
    let periods_per_year = 252.0;

    let metrics = PerformanceMetrics::evaluate(&returns, periods_per_year);

    // Sharpe should equal information ratio in this implementation
    assert_relative_eq!(metrics.sharpe_ratio, metrics.information_ratio, epsilon = 1e-10);

    // Sharpe should be mean / std * sqrt(periods_per_year)
    if metrics.std_dev > 0.0 {
        let expected_sharpe = (metrics.mean_return / metrics.std_dev) * periods_per_year.sqrt();
        assert_relative_eq!(metrics.sharpe_ratio, expected_sharpe, epsilon = 1e-6);
    }
}

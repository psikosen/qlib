use std::io::Write;

use approx::assert_abs_diff_eq;
use chrono::{TimeZone, Utc};
use tempfile::NamedTempFile;

use polars::prelude::*;

use qliber::dataset::MarketData;
use qliber::features::{with_daily_returns, with_moving_average, with_z_score};
use qliber::logging;
use qliber::metrics::{
    AccumulationMode, AnalysisFrequency, FrequencyUnit, IndicatorMethod, PerformanceMetrics,
    indicator_analysis,
};

#[test]
fn end_to_end_pipeline_produces_expected_statistics() -> anyhow::Result<()> {
    logging::init_logging()?;

    let mut file = NamedTempFile::new()?;
    writeln!(
        file,
        "timestamp,close\n2024-01-01T00:00:00Z,100\n2024-01-02T00:00:00Z,101\n2024-01-03T00:00:00Z,102\n2024-01-04T00:00:00Z,104\n2024-01-05T00:00:00Z,103"
    )?;

    let market = MarketData::from_csv(file.path())?;
    let start = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
    let end = Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap();
    let filtered = market.filter_date_range("timestamp", Some(start), Some(end))?;

    let df = filtered.collect()?;
    assert_eq!(df.shape().0, 4);

    let with_returns = with_daily_returns(&df, "close", "return")?;
    let with_ma = with_moving_average(&with_returns, "close", 2, "ma_2")?;
    let enriched = with_z_score(&with_ma, "close", 3, "z_close")?;

    let returns = enriched
        .column("return")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<_>>();

    let expected_return = (102.0 / 101.0) - 1.0;
    assert_abs_diff_eq!(returns[1], expected_return, epsilon = 1e-8);
    assert!(
        enriched
            .column("ma_2")?
            .f64()?
            .into_iter()
            .flatten()
            .all(|v| v > 0.0)
    );

    let metrics = PerformanceMetrics::evaluate(&returns, 252.0);
    assert!(metrics.cumulative_return > 0.0);
    assert!(metrics.mean_return > 0.0);
    assert!(metrics.annualized_volatility >= 0.0);
    assert_abs_diff_eq!(
        metrics.sharpe_ratio,
        metrics.information_ratio,
        epsilon = 1e-12
    );

    Ok(())
}

#[test]
fn performance_metrics_align_with_python_risk_analysis() {
    let returns = vec![0.01, -0.015, 0.02, -0.005];

    let sum_mode = PerformanceMetrics::evaluate_with_mode(&returns, 252.0, AccumulationMode::Sum);
    assert_abs_diff_eq!(sum_mode.mean_return, 0.0025, epsilon = 1e-12);
    assert_abs_diff_eq!(sum_mode.std_dev, 0.015545631755148026, epsilon = 1e-12);
    assert_abs_diff_eq!(sum_mode.cumulative_return, 0.01, epsilon = 1e-12);
    assert_abs_diff_eq!(sum_mode.annualized_return, 0.63, epsilon = 1e-12);
    assert_abs_diff_eq!(
        sum_mode.annualized_volatility,
        0.24677925358506136,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        sum_mode.information_ratio,
        2.5528888301902897,
        epsilon = 1e-9
    );
    assert_abs_diff_eq!(sum_mode.max_drawdown, -0.015, epsilon = 1e-12);
    assert_abs_diff_eq!(
        sum_mode.sharpe_ratio,
        sum_mode.information_ratio,
        epsilon = 1e-12
    );

    let product_mode =
        PerformanceMetrics::evaluate_with_mode(&returns, 252.0, AccumulationMode::Product);
    assert_abs_diff_eq!(
        product_mode.mean_return,
        0.002409593043190217,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(product_mode.std_dev, 0.015508406743410254, epsilon = 1e-12);
    assert_abs_diff_eq!(
        product_mode.cumulative_return,
        0.009673264999999986,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        product_mode.annualized_return,
        0.8339773917946953,
        epsilon = 1e-9
    );
    assert_abs_diff_eq!(
        product_mode.annualized_volatility,
        0.24618832484340372,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        product_mode.information_ratio,
        2.4664753995550988,
        epsilon = 1e-9
    );
    assert_abs_diff_eq!(
        product_mode.max_drawdown,
        -0.015000000000000013,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        product_mode.sharpe_ratio,
        product_mode.information_ratio,
        epsilon = 1e-12
    );
}

#[test]
fn performance_metrics_frequency_parity() {
    let returns = vec![0.01, -0.015, 0.02, -0.005];
    let frequency = AnalysisFrequency::new(1, FrequencyUnit::Day);
    let from_frequency =
        PerformanceMetrics::evaluate_with_frequency(&returns, frequency, AccumulationMode::Sum);
    let direct = PerformanceMetrics::evaluate_with_mode(&returns, 238.0, AccumulationMode::Sum);

    assert_abs_diff_eq!(
        from_frequency.annualized_return,
        direct.annualized_return,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        from_frequency.information_ratio,
        direct.information_ratio,
        epsilon = 1e-12
    );

    let from_str =
        PerformanceMetrics::evaluate_with_frequency_str(&returns, "2week", AccumulationMode::Sum)
            .expect("valid frequency string");
    let manual = PerformanceMetrics::evaluate_with_mode(&returns, 25.0, AccumulationMode::Sum);
    assert_abs_diff_eq!(
        from_str.annualized_return,
        manual.annualized_return,
        epsilon = 1e-12
    );
}

#[test]
fn indicator_analysis_matches_python_behaviour() -> anyhow::Result<()> {
    let frame = df! {
        "count" => &[5.0, 10.0, 20.0],
        "ffr" => &[0.1, 0.5, 0.9],
        "pa" => &[0.2, 0.8, 0.4],
        "pos" => &[0.3, 0.6, 0.7],
        "deal_amount" => &[100.0, 400.0, 50.0],
        "value" => &[1000.0, 200.0, 800.0],
    }?;

    let mean = indicator_analysis(&frame, IndicatorMethod::Mean)?;
    let amount = indicator_analysis(&frame, IndicatorMethod::AmountWeighted)?;
    let value = indicator_analysis(&frame, IndicatorMethod::ValueWeighted)?;

    let extract = |df: &DataFrame, indicator: &str| -> f64 {
        df.column("indicator")
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter()
            .zip(df.column("value").unwrap().f64().unwrap().into_iter())
            .find_map(|(name, value)| match (name, value) {
                (Some(name), Some(value)) if name == indicator => Some(value),
                _ => None,
            })
            .unwrap()
    };

    assert_abs_diff_eq!(extract(&mean, "ffr"), 0.6714285714285715, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&mean, "pa"), 0.48571428571428577, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&mean, "pos"), 0.6142857142857143, epsilon = 1e-12);

    assert_abs_diff_eq!(extract(&amount, "ffr"), 0.4636363636363636, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&amount, "pa"), 0.6545454545454545, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&amount, "pos"), 0.6142857142857143, epsilon = 1e-12);

    assert_abs_diff_eq!(extract(&value, "ffr"), 0.46, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&value, "pa"), 0.34, epsilon = 1e-12);
    assert_abs_diff_eq!(extract(&value, "pos"), 0.6142857142857143, epsilon = 1e-12);

    Ok(())
}

use std::io::Write;

use approx::assert_abs_diff_eq;
use chrono::{TimeZone, Utc};
use tempfile::NamedTempFile;

use qliber::dataset::MarketData;
use qliber::features::{with_daily_returns, with_moving_average, with_z_score};
use qliber::logging;
use qliber::metrics::{AccumulationMode, PerformanceMetrics};

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

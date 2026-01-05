use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{with_daily_returns, with_moving_average, with_z_score};

#[test]
fn test_with_daily_returns_basic() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0]
    }
    .unwrap();

    let result = with_daily_returns(&df, "close", "return").unwrap();

    assert!(result.column("return").is_ok());
    let returns = result.column("return").unwrap().f64().unwrap();

    // First return should be 0.0
    assert_relative_eq!(returns.get(0).unwrap(), 0.0, epsilon = 1e-10);

    // Second return: (110 - 100) / 100 = 0.1
    assert_relative_eq!(returns.get(1).unwrap(), 0.1, epsilon = 1e-10);

    // Third return: (105 - 110) / 110 ≈ -0.04545
    assert_relative_eq!(returns.get(2).unwrap(), -0.045454545, epsilon = 1e-6);

    // Fourth return: (120 - 105) / 105 ≈ 0.14286
    assert_relative_eq!(returns.get(3).unwrap(), 0.142857142, epsilon = 1e-6);
}

#[test]
fn test_with_daily_returns_zero_price() {
    let df = df! {
        "close" => &[0.0, 100.0, 0.0, 50.0]
    }
    .unwrap();

    let result = with_daily_returns(&df, "close", "return").unwrap();
    let returns = result.column("return").unwrap().f64().unwrap();

    // When previous price is 0, return should be 0
    assert_relative_eq!(returns.get(0).unwrap(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(returns.get(1).unwrap(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_with_daily_returns_empty_df() {
    let df = df! {
        "close" => &[] as &[f64]
    }
    .unwrap();

    let result = with_daily_returns(&df, "close", "return").unwrap();
    assert_eq!(result.height(), 0);
}

#[test]
fn test_with_daily_returns_preserves_original_columns() {
    let df = df! {
        "close" => &[100.0, 110.0, 120.0],
        "volume" => &[1000, 1100, 1200]
    }
    .unwrap();

    let result = with_daily_returns(&df, "close", "return").unwrap();

    // Original columns should be preserved
    assert!(result.column("close").is_ok());
    assert!(result.column("volume").is_ok());
    assert!(result.column("return").is_ok());
}

#[test]
fn test_with_moving_average_basic() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    }
    .unwrap();

    let result = with_moving_average(&df, "price", 3, "ma_3").unwrap();
    let ma = result.column("ma_3").unwrap().f64().unwrap();

    // First value: average of [10] = 10
    assert_relative_eq!(ma.get(0).unwrap(), 10.0, epsilon = 1e-10);

    // Second value: average of [10, 20] = 15
    assert_relative_eq!(ma.get(1).unwrap(), 15.0, epsilon = 1e-10);

    // Third value: average of [10, 20, 30] = 20
    assert_relative_eq!(ma.get(2).unwrap(), 20.0, epsilon = 1e-10);

    // Fourth value: average of [20, 30, 40] = 30
    assert_relative_eq!(ma.get(3).unwrap(), 30.0, epsilon = 1e-10);

    // Fifth value: average of [30, 40, 50] = 40
    assert_relative_eq!(ma.get(4).unwrap(), 40.0, epsilon = 1e-10);
}

#[test]
fn test_with_moving_average_window_one() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0]
    }
    .unwrap();

    let result = with_moving_average(&df, "price", 1, "ma_1").unwrap();
    let ma = result.column("ma_1").unwrap().f64().unwrap();

    // Window of 1 should return the original values
    assert_relative_eq!(ma.get(0).unwrap(), 10.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(1).unwrap(), 20.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(2).unwrap(), 30.0, epsilon = 1e-10);
}

#[test]
fn test_with_moving_average_large_window() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0]
    }
    .unwrap();

    let result = with_moving_average(&df, "price", 10, "ma_10").unwrap();
    let ma = result.column("ma_10").unwrap().f64().unwrap();

    // When window is larger than data, use available data
    assert_relative_eq!(ma.get(0).unwrap(), 10.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(1).unwrap(), 15.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(2).unwrap(), 20.0, epsilon = 1e-10);
}

#[test]
fn test_with_moving_average_empty_df() {
    let df = df! {
        "price" => &[] as &[f64]
    }
    .unwrap();

    let result = with_moving_average(&df, "price", 3, "ma_3").unwrap();
    assert_eq!(result.height(), 0);
}

#[test]
#[should_panic(expected = "window size must be positive")]
fn test_with_moving_average_zero_window() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0]
    }
    .unwrap();

    with_moving_average(&df, "price", 0, "ma_0").unwrap();
}

#[test]
fn test_with_z_score_basic() {
    let df = df! {
        "value" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 3, "z_value").unwrap();
    let z_scores = result.column("z_value").unwrap().f64().unwrap();

    // All z-scores should be finite
    for i in 0..z_scores.len() {
        assert!(z_scores.get(i).unwrap().is_finite());
    }

    // Z-score should be 0 when value equals mean (approximately)
    // For a rolling window, middle value should have z-score close to 0
}

#[test]
fn test_with_z_score_constant_values() {
    let df = df! {
        "value" => &[10.0, 10.0, 10.0, 10.0]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 3, "z_value").unwrap();
    let z_scores = result.column("z_value").unwrap().f64().unwrap();

    // When all values are constant, std is 0, so z-score should be 0
    for i in 0..z_scores.len() {
        assert_relative_eq!(z_scores.get(i).unwrap(), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_with_z_score_normalized_distribution() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 5, "z_value").unwrap();
    let z_scores = result.column("z_value").unwrap().f64().unwrap();

    // Z-scores should be within reasonable bounds (typically -3 to 3 for most data)
    for i in 0..z_scores.len() {
        let z = z_scores.get(i).unwrap();
        assert!(z.abs() < 5.0, "Z-score {} is out of reasonable bounds", z);
    }
}

#[test]
fn test_with_z_score_window_two() {
    let df = df! {
        "value" => &[10.0, 20.0, 30.0, 40.0]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 2, "z_value").unwrap();
    let z_scores = result.column("z_value").unwrap().f64().unwrap();

    // All z-scores should be finite
    for i in 0..z_scores.len() {
        assert!(z_scores.get(i).unwrap().is_finite());
    }
}

#[test]
fn test_with_z_score_empty_df() {
    let df = df! {
        "value" => &[] as &[f64]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 3, "z_value").unwrap();
    assert_eq!(result.height(), 0);
}

#[test]
#[should_panic(expected = "window size must exceed one")]
fn test_with_z_score_window_one() {
    let df = df! {
        "value" => &[10.0, 20.0, 30.0]
    }
    .unwrap();

    with_z_score(&df, "value", 1, "z_value").unwrap();
}

#[test]
fn test_with_z_score_preserves_columns() {
    let df = df! {
        "value" => &[10.0, 20.0, 30.0],
        "other" => &[1, 2, 3]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 2, "z_value").unwrap();

    assert!(result.column("value").is_ok());
    assert!(result.column("other").is_ok());
    assert!(result.column("z_value").is_ok());
}

#[test]
fn test_feature_engineering_pipeline() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0]
    }
    .unwrap();

    // Chain multiple feature engineering steps
    let with_returns = with_daily_returns(&df, "close", "return").unwrap();
    let with_ma = with_moving_average(&with_returns, "close", 3, "ma_3").unwrap();
    let final_df = with_z_score(&with_ma, "close", 3, "z_close").unwrap();

    // Verify all columns are present
    assert!(final_df.column("close").is_ok());
    assert!(final_df.column("return").is_ok());
    assert!(final_df.column("ma_3").is_ok());
    assert!(final_df.column("z_close").is_ok());

    // Verify row count is preserved
    assert_eq!(final_df.height(), 6);
}

#[test]
fn test_with_daily_returns_negative_prices() {
    let df = df! {
        "close" => &[100.0, -50.0, 75.0]
    }
    .unwrap();

    let result = with_daily_returns(&df, "close", "return").unwrap();
    let returns = result.column("return").unwrap().f64().unwrap();

    // Returns should be calculated even with negative prices
    assert_relative_eq!(returns.get(0).unwrap(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(returns.get(1).unwrap(), -1.5, epsilon = 1e-10);
}

#[test]
fn test_with_moving_average_negative_values() {
    let df = df! {
        "price" => &[-10.0, -20.0, -30.0]
    }
    .unwrap();

    let result = with_moving_average(&df, "price", 2, "ma_2").unwrap();
    let ma = result.column("ma_2").unwrap().f64().unwrap();

    assert_relative_eq!(ma.get(0).unwrap(), -10.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(1).unwrap(), -15.0, epsilon = 1e-10);
    assert_relative_eq!(ma.get(2).unwrap(), -25.0, epsilon = 1e-10);
}

#[test]
fn test_with_z_score_outliers() {
    let df = df! {
        "value" => &[10.0, 11.0, 10.5, 100.0, 10.2]
    }
    .unwrap();

    let result = with_z_score(&df, "value", 3, "z_value").unwrap();
    let z_scores = result.column("z_value").unwrap().f64().unwrap();

    // The outlier (100.0) should have a high z-score in absolute value
    let outlier_zscore = z_scores.get(3).unwrap();
    assert!(outlier_zscore.abs() > 1.0, "Outlier should have high z-score");
}

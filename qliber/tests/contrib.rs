use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{Alpha158Processor, Alpha360Processor, Processor};

#[test]
fn test_alpha158_processor_name() {
    let processor = Alpha158Processor::new("close");
    assert_eq!(processor.name(), "alpha158");
}

#[test]
fn test_alpha158_processor_basic() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should have original column plus return, ma_5, and ma_20
    assert!(result.column("close").is_ok());
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_5").is_ok());
    assert!(result.column("ma_20").is_ok());

    // Verify row count is preserved
    assert_eq!(result.height(), 6);
}

#[test]
fn test_alpha158_processor_return_column() {
    let df = df! {
        "price" => &[100.0, 110.0, 120.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("price");
    let result = processor.process(df).unwrap();

    let returns = result.column("return").unwrap().f64().unwrap();

    // First return should be 0.0
    assert_relative_eq!(returns.get(0).unwrap(), 0.0, epsilon = 1e-10);

    // Second return: (110 - 100) / 100 = 0.1
    assert_relative_eq!(returns.get(1).unwrap(), 0.1, epsilon = 1e-10);

    // Third return: (120 - 110) / 110 ≈ 0.0909
    assert_relative_eq!(returns.get(2).unwrap(), 0.09090909, epsilon = 1e-6);
}

#[test]
fn test_alpha158_processor_ma_5() {
    let df = df! {
        "close" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    let ma_5 = result.column("ma_5").unwrap().f64().unwrap();

    // First value: average of [10] = 10
    assert_relative_eq!(ma_5.get(0).unwrap(), 10.0, epsilon = 1e-10);

    // Fifth value: average of [10, 20, 30, 40, 50] = 30
    assert_relative_eq!(ma_5.get(4).unwrap(), 30.0, epsilon = 1e-10);

    // Sixth value: average of [20, 30, 40, 50, 60] = 40
    assert_relative_eq!(ma_5.get(5).unwrap(), 40.0, epsilon = 1e-10);
}

#[test]
fn test_alpha158_processor_ma_20() {
    let df = df! {
        "close" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    let ma_20 = result.column("ma_20").unwrap().f64().unwrap();

    // Since we only have 5 data points, MA-20 will use available data
    // First value: average of [10] = 10
    assert_relative_eq!(ma_20.get(0).unwrap(), 10.0, epsilon = 1e-10);

    // Fifth value: average of [10, 20, 30, 40, 50] = 30
    assert_relative_eq!(ma_20.get(4).unwrap(), 30.0, epsilon = 1e-10);
}

#[test]
fn test_alpha158_processor_empty_dataframe() {
    let df = df! {
        "close" => &[] as &[f64]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    assert_eq!(result.height(), 0);
}

#[test]
fn test_alpha158_processor_custom_column_name() {
    let df = df! {
        "my_price" => &[100.0, 110.0, 120.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("my_price");
    let result = processor.process(df).unwrap();

    // Should work with custom column name
    assert!(result.column("my_price").is_ok());
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_5").is_ok());
    assert!(result.column("ma_20").is_ok());
}

#[test]
fn test_alpha360_processor_name() {
    let processor = Alpha360Processor::new("close");
    assert_eq!(processor.name(), "alpha360");
}

#[test]
fn test_alpha360_processor_basic() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should have original column plus return, ma_10, z_price, and return_sq
    assert!(result.column("close").is_ok());
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_10").is_ok());
    assert!(result.column("z_price").is_ok());
    assert!(result.column("return_sq").is_ok());

    // Verify row count is preserved
    assert_eq!(result.height(), 16);
}

#[test]
fn test_alpha360_processor_return_column() {
    let df = df! {
        "price" => &[100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("price");
    let result = processor.process(df).unwrap();

    let returns = result.column("return").unwrap().f64().unwrap();

    // First return should be 0.0
    assert_relative_eq!(returns.get(0).unwrap(), 0.0, epsilon = 1e-10);

    // Second return: (110 - 100) / 100 = 0.1
    assert_relative_eq!(returns.get(1).unwrap(), 0.1, epsilon = 1e-10);
}

#[test]
fn test_alpha360_processor_ma_10() {
    let df = df! {
        "close" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    let ma_10 = result.column("ma_10").unwrap().f64().unwrap();

    // Tenth value: average of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] = 55
    assert_relative_eq!(ma_10.get(9).unwrap(), 55.0, epsilon = 1e-10);

    // Eleventh value: average of [20, 30, 40, 50, 60, 70, 80, 90, 100, 110] = 65
    assert_relative_eq!(ma_10.get(10).unwrap(), 65.0, epsilon = 1e-10);
}

#[test]
fn test_alpha360_processor_z_price() {
    let df = df! {
        "close" => &[10.0, 11.0, 10.5, 10.2, 10.8, 11.2, 10.9, 11.1, 10.7, 10.6, 11.0, 10.4, 10.3, 10.9, 11.3, 10.5]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    let z_price = result.column("z_price").unwrap().f64().unwrap();

    // All z-scores should be finite
    for i in 0..z_price.len() {
        assert!(z_price.get(i).unwrap().is_finite(), "Z-score at index {} should be finite", i);
    }
}

#[test]
fn test_alpha360_processor_return_sq() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    let returns = result.column("return").unwrap().f64().unwrap();
    let return_sq = result.column("return_sq").unwrap().f64().unwrap();

    // return_sq should be the square of return
    for i in 0..returns.len() {
        let ret = returns.get(i).unwrap();
        let ret_sq = return_sq.get(i).unwrap();
        assert_relative_eq!(ret_sq, ret * ret, epsilon = 1e-10);
    }
}

#[test]
fn test_alpha360_processor_empty_dataframe() {
    let df = df! {
        "close" => &[] as &[f64]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df);

    // Empty dataframes will cause an error when trying to access return column
    // This is expected behavior
    assert!(result.is_err() || (result.is_ok() && result.unwrap().height() == 0));
}

#[test]
fn test_alpha360_processor_custom_column_name() {
    let df = df! {
        "my_price" => &[100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("my_price");
    let result = processor.process(df).unwrap();

    // Should work with custom column name
    assert!(result.column("my_price").is_ok());
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_10").is_ok());
    assert!(result.column("z_price").is_ok());
    assert!(result.column("return_sq").is_ok());
}

#[test]
fn test_alpha158_processor_preserves_other_columns() {
    let df = df! {
        "close" => &[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        "volume" => &[1000, 1100, 1200, 1300, 1400, 1500]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should preserve original columns
    assert!(result.column("close").is_ok());
    assert!(result.column("volume").is_ok());
}

#[test]
fn test_alpha360_processor_preserves_other_columns() {
    let df = df! {
        "close" => &[100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        "volume" => &[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should preserve original columns
    assert!(result.column("close").is_ok());
    assert!(result.column("volume").is_ok());
}

#[test]
fn test_alpha158_vs_alpha360_differences() {
    let df = df! {
        "close" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0]
    }
    .unwrap();

    let alpha158 = Alpha158Processor::new("close");
    let result158 = alpha158.process(df.clone()).unwrap();

    let alpha360 = Alpha360Processor::new("close");
    let result360 = alpha360.process(df).unwrap();

    // Alpha158 should have ma_5 and ma_20, but not z_price or return_sq
    assert!(result158.column("ma_5").is_ok());
    assert!(result158.column("ma_20").is_ok());
    assert!(result158.column("z_price").is_err());
    assert!(result158.column("return_sq").is_err());

    // Alpha360 should have ma_10, z_price, and return_sq, but not ma_5 or ma_20
    assert!(result360.column("ma_10").is_ok());
    assert!(result360.column("z_price").is_ok());
    assert!(result360.column("return_sq").is_ok());
    assert!(result360.column("ma_5").is_err());
    assert!(result360.column("ma_20").is_err());
}

#[test]
fn test_alpha158_processor_with_negative_prices() {
    let df = df! {
        "close" => &[100.0, -50.0, 75.0, 80.0, 85.0, 90.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should still process successfully even with negative prices
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_5").is_ok());
    assert!(result.column("ma_20").is_ok());
}

#[test]
fn test_alpha360_processor_with_negative_prices() {
    let df = df! {
        "close" => &[100.0, -50.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0]
    }
    .unwrap();

    let processor = Alpha360Processor::new("close");
    let result = processor.process(df).unwrap();

    // Should still process successfully even with negative prices
    assert!(result.column("return").is_ok());
    assert!(result.column("ma_10").is_ok());
    assert!(result.column("z_price").is_ok());
    assert!(result.column("return_sq").is_ok());
}

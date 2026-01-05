use polars::prelude::*;
use qliber::{Alpha158Processor, Alpha360Processor, Processor};

#[test]
fn test_alpha158_processor_name() {
    let processor = Alpha158Processor::new();
    assert_eq!(processor.name(), "alpha158");
}

#[test]
fn test_alpha158_processor_basic() {
    // Alpha158 needs open, close, high, low, volume columns
    let df = df! {
        "open" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0, 130.0, 128.0, 135.0, 140.0],
        "close" => &[105.0, 108.0, 110.0, 118.0, 120.0, 123.0, 128.0, 132.0, 138.0, 142.0],
        "high" => &[106.0, 112.0, 112.0, 122.0, 122.0, 127.0, 132.0, 134.0, 140.0, 145.0],
        "low" => &[99.0, 107.0, 104.0, 117.0, 114.0, 122.0, 127.0, 126.0, 134.0, 139.0],
        "volume" => &[1000.0, 1100.0, 1050.0, 1200.0, 1150.0, 1250.0, 1300.0, 1280.0, 1350.0, 1400.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new();
    let result = processor.process(df).unwrap();

    // Should have original columns plus many Alpha158 features
    assert!(result.column("close").is_ok());
    assert!(result.column("open").is_ok());
    assert!(result.column("high").is_ok());
    assert!(result.column("low").is_ok());
    assert!(result.column("volume").is_ok());

    // Check for KBAR features
    assert!(result.column("KMID").is_ok());
    assert!(result.column("KLEN").is_ok());
    assert!(result.column("KUP").is_ok());
    assert!(result.column("KDOWN").is_ok());

    // Check for rolling price features
    assert!(result.column("ROC_5").is_ok());
    assert!(result.column("MA_5").is_ok());
    assert!(result.column("STD_5").is_ok());

    // Check for rolling volume features
    assert!(result.column("VROC_5").is_ok());
    assert!(result.column("VMA_5").is_ok());

    // Check for advanced rolling features
    assert!(result.column("BETA_5").is_ok());
    assert!(result.column("RSQR_5").is_ok());
    assert!(result.column("RESI_5").is_ok());
    assert!(result.column("MAX_5").is_ok());
    assert!(result.column("MIN_5").is_ok());
    assert!(result.column("RANK_5").is_ok());

    // Verify row count is preserved
    assert_eq!(result.height(), 10);
}

#[test]
fn test_alpha158_kbar_features() {
    let df = df! {
        "open" => &[100.0, 100.0, 100.0],
        "close" => &[105.0, 98.0, 100.0],
        "high" => &[106.0, 102.0, 101.0],
        "low" => &[99.0, 97.0, 99.0],
        "volume" => &[1000.0, 1100.0, 1050.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new();
    let result = processor.process(df).unwrap();

    // Verify KBAR features exist
    assert!(result.column("KMID").is_ok());
    assert!(result.column("KLEN").is_ok());
    assert!(result.column("KMID2").is_ok());
}

#[test]
fn test_alpha158_empty_dataframe() {
    let df = df! {
        "open" => &[] as &[f64],
        "close" => &[] as &[f64],
        "high" => &[] as &[f64],
        "low" => &[] as &[f64],
        "volume" => &[] as &[f64]
    }
    .unwrap();

    let processor = Alpha158Processor::new();
    let result = processor.process(df).unwrap();

    assert_eq!(result.height(), 0);
}

#[test]
fn test_alpha158_preserves_original_columns() {
    let df = df! {
        "open" => &[100.0, 110.0, 105.0, 120.0, 115.0, 125.0],
        "close" => &[105.0, 108.0, 110.0, 118.0, 120.0, 123.0],
        "high" => &[106.0, 112.0, 112.0, 122.0, 122.0, 127.0],
        "low" => &[99.0, 107.0, 104.0, 117.0, 114.0, 122.0],
        "volume" => &[1000.0, 1100.0, 1050.0, 1200.0, 1150.0, 1250.0]
    }
    .unwrap();

    let processor = Alpha158Processor::new();
    let result = processor.process(df.clone()).unwrap();

    // Verify all original columns are preserved
    let orig_open = df.column("open").unwrap();
    let result_open = result.column("open").unwrap();
    assert_eq!(orig_open.len(), result_open.len());
}

#[test]
fn test_alpha360_processor_name() {
    let processor = Alpha360Processor::new();
    assert_eq!(processor.name(), "alpha360");
}

#[test]
fn test_alpha360_processor_basic() {
    // Alpha360 needs open, close, high, low, vwap, volume columns
    // Need at least 60 rows for lookback
    let values: Vec<f64> = (0..70).map(|i| 100.0 + i as f64).collect();

    let df = df! {
        "open" => values.clone(),
        "close" => values.iter().map(|v| v + 1.0).collect::<Vec<f64>>(),
        "high" => values.iter().map(|v| v + 2.0).collect::<Vec<f64>>(),
        "low" => values.iter().map(|v| v - 1.0).collect::<Vec<f64>>(),
        "vwap" => values.iter().map(|v| v + 0.5).collect::<Vec<f64>>(),
        "volume" => values.iter().map(|v| v * 100.0).collect::<Vec<f64>>(),
    }
    .unwrap();

    let processor = Alpha360Processor::new();
    let result = processor.process(df).unwrap();

    // Should have original columns
    assert!(result.column("close").is_ok());
    assert!(result.column("open").is_ok());
    assert!(result.column("vwap").is_ok());

    // Should have lookback features for each field
    assert!(result.column("close_0").is_ok());
    assert!(result.column("close_10").is_ok());
    assert!(result.column("open_0").is_ok());
    assert!(result.column("high_0").is_ok());

    // Verify row count is preserved
    assert_eq!(result.height(), 70);
}

#[test]
fn test_alpha360_lookback_features() {
    let values: Vec<f64> = (0..70).map(|i| 100.0 + i as f64).collect();

    let df = df! {
        "open" => values.clone(),
        "close" => values.clone(),
        "high" => values.clone(),
        "low" => values.clone(),
        "vwap" => values.clone(),
        "volume" => values.clone(),
    }
    .unwrap();

    let processor = Alpha360Processor::new();
    let result = processor.process(df).unwrap();

    // Verify lookback features exist for different days
    for day in [0, 5, 10, 30, 59] {
        assert!(result.column(&format!("close_{}", day)).is_ok());
        assert!(result.column(&format!("open_{}", day)).is_ok());
    }
}

#[test]
fn test_alpha360_empty_dataframe() {
    let df = df! {
        "open" => &[] as &[f64],
        "close" => &[] as &[f64],
        "high" => &[] as &[f64],
        "low" => &[] as &[f64],
        "vwap" => &[] as &[f64],
        "volume" => &[] as &[f64]
    }
    .unwrap();

    let processor = Alpha360Processor::new();
    let result = processor.process(df).unwrap();

    assert_eq!(result.height(), 0);
}

#[test]
fn test_alpha360_preserves_original_columns() {
    let values: Vec<f64> = (0..70).map(|i| 100.0 + i as f64).collect();

    let df = df! {
        "open" => values.clone(),
        "close" => values.clone(),
        "high" => values.clone(),
        "low" => values.clone(),
        "vwap" => values.clone(),
        "volume" => values.clone(),
    }
    .unwrap();

    let processor = Alpha360Processor::new();
    let result = processor.process(df.clone()).unwrap();

    // Verify all original columns are preserved
    let orig_close = df.column("close").unwrap();
    let result_close = result.column("close").unwrap();
    assert_eq!(orig_close.len(), result_close.len());
}

#[test]
fn test_alpha158_with_nulls() {
    let df = df! {
        "open" => &[Some(100.0), None, Some(105.0), Some(120.0), Some(115.0)],
        "close" => &[Some(105.0), Some(108.0), Some(110.0), None, Some(120.0)],
        "high" => &[Some(106.0), Some(112.0), Some(112.0), Some(122.0), Some(122.0)],
        "low" => &[Some(99.0), Some(107.0), None, Some(117.0), Some(114.0)],
        "volume" => &[Some(1000.0), Some(1100.0), Some(1050.0), Some(1200.0), Some(1150.0)]
    }
    .unwrap();

    let processor = Alpha158Processor::new();
    let result = processor.process(df);

    // Should handle nulls gracefully
    assert!(result.is_ok());
}

#[test]
fn test_alpha360_with_nulls() {
    let mut values: Vec<Option<f64>> = (0..70).map(|i| Some(100.0 + i as f64)).collect();
    values[5] = None;
    values[10] = None;

    let df = df! {
        "open" => values.clone(),
        "close" => values.clone(),
        "high" => values.clone(),
        "low" => values.clone(),
        "vwap" => values.clone(),
        "volume" => values.clone(),
    }
    .unwrap();

    let processor = Alpha360Processor::new();
    let result = processor.process(df);

    // Should handle nulls gracefully
    assert!(result.is_ok());
}

use std::io::Write;

use chrono::{TimeZone, Utc};
use tempfile::NamedTempFile;

use polars::prelude::*;

use qliber::dataset::{
    DataHandler, DataHandlerConfig, DatasetBatch, ExpressionProcessor, FillForwardProcessor,
    LoaderOptions, ProcessorRef,
};
use qliber::ops::evaluate_expression;
use qliber::{Alpha158Processor, Alpha360Processor, MarketData, Processor};

fn processor(expr: &str, output: &str) -> ProcessorRef {
    std::sync::Arc::new(ExpressionProcessor::new(
        expr.to_string(),
        output.to_string(),
    ))
}

#[test]
fn data_handler_applies_processors() -> anyhow::Result<()> {
    let mut file = NamedTempFile::new()?;
    writeln!(
        file,
        "timestamp,open,close,high,low,volume\n\
        2024-01-01T00:00:00Z,100.0,102.0,103.0,99.0,1000.0\n\
        2024-01-02T00:00:00Z,101.0,103.0,104.0,100.0,1100.0\n\
        2024-01-03T00:00:00Z,105.0,107.0,108.0,104.0,1200.0\n\
        2024-01-04T00:00:00Z,104.0,106.0,107.0,103.0,1150.0\n\
        2024-01-05T00:00:00Z,106.0,108.0,109.0,105.0,1200.0\n\
        2024-01-06T00:00:00Z,107.0,109.0,110.0,106.0,1250.0\n\
        2024-01-07T00:00:00Z,108.0,110.0,111.0,107.0,1300.0"
    )?;

    let market = MarketData::from_csv(file.path())?;
    let config = DataHandlerConfig {
        feature_processors: vec![
            std::sync::Arc::new(FillForwardProcessor::new(None)),
            processor("close", "base"),
        ],
        label_processors: vec![processor("close", "label_price")],
        feature_columns: vec![
            "timestamp".to_string(),
            "close".to_string(),
            "base".to_string(),
        ],
        label_columns: vec!["label_price".to_string()],
    };
    let handler = DataHandler::new(market, config);
    let loader = handler.loader(LoaderOptions {
        date_column: Some("timestamp".to_string()),
        start: Some(Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap()),
        end: None,
        limit: None,
    })?;

    let DatasetBatch { features, labels } = loader.load()?;
    assert_eq!(features.height(), 6);
    assert!(features.column("base").is_ok());
    assert!(labels.column("label_price").is_ok());

    Ok(())
}

#[test]
fn contrib_processors_append_expected_columns() -> anyhow::Result<()> {
    let frame = df! {
        "open" => &[100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0, 108.0, 110.0],
        "close" => &[102.0, 103.0, 105.0, 104.0, 106.0, 107.0, 109.0, 108.0, 110.0, 112.0],
        "high" => &[103.0, 104.0, 106.0, 105.0, 107.0, 108.0, 110.0, 109.0, 111.0, 113.0],
        "low" => &[99.0, 100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0, 107.0, 109.0],
        "volume" => &[1000.0, 1100.0, 1200.0, 1150.0, 1250.0, 1300.0, 1400.0, 1350.0, 1450.0, 1500.0],
        "timestamp" => &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }?;

    let alpha158 = Alpha158Processor::new();
    let processed_158 = alpha158.process(frame.clone())?;
    assert!(processed_158.column("MA_5").is_ok());
    assert!(processed_158.column("KMID").is_ok());

    // Alpha360 needs at least 60 rows for lookback
    let values: Vec<f64> = (0..70).map(|i| 100.0 + i as f64).collect();
    let frame_360 = df! {
        "open" => values.clone(),
        "close" => values.clone(),
        "high" => values.clone(),
        "low" => values.clone(),
        "vwap" => values.clone(),
        "volume" => values.clone(),
    }?;

    let alpha360 = Alpha360Processor::new();
    let processed_360 = alpha360.process(frame_360)?;
    assert!(processed_360.column("close_0").is_ok());
    assert!(processed_360.column("open_10").is_ok());

    Ok(())
}

#[test]
fn expression_engine_supports_operations() -> anyhow::Result<()> {
    let frame = df! {
        "a" => &[1.0, 2.0, 3.0, 4.0],
        "b" => &[2.0, 2.0, 2.0, 2.0],
    }?;
    let result = evaluate_expression("rolling_mean(a, 2) + lag(b, 1)", &frame)?;
    assert_eq!(result.len(), frame.height());
    Ok(())
}

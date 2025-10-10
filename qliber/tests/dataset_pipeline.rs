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
        "timestamp,price\n2024-01-01T00:00:00Z,100\n2024-01-02T00:00:00Z,101\n2024-01-03T00:00:00Z,105\n2024-01-04T00:00:00Z,104"
    )?;

    let market = MarketData::from_csv(file.path())?;
    let config = DataHandlerConfig {
        feature_processors: vec![
            std::sync::Arc::new(FillForwardProcessor::new(None)),
            processor("price", "base"),
            std::sync::Arc::new(Alpha158Processor::new("price")),
        ],
        label_processors: vec![processor("price", "label_price")],
        feature_columns: vec![
            "timestamp".to_string(),
            "price".to_string(),
            "ma_5".to_string(),
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
    assert_eq!(features.height(), 3);
    assert!(features.column("ma_5").is_ok());
    assert!(labels.column("label_price").is_ok());

    Ok(())
}

#[test]
fn contrib_processors_append_expected_columns() -> anyhow::Result<()> {
    let frame = df! {
        "price" => &[100.0, 101.0, 103.0, 102.0, 104.0],
        "timestamp" => &[1, 2, 3, 4, 5],
    }?;

    let alpha158 = Alpha158Processor::new("price");
    let processed_158 = alpha158.process(frame.clone())?;
    assert!(processed_158.column("ma_5").is_ok());

    let alpha360 = Alpha360Processor::new("price");
    let processed_360 = alpha360.process(frame.clone())?;
    assert!(processed_360.column("z_price").is_ok());
    assert!(processed_360.column("return_sq").is_ok());

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

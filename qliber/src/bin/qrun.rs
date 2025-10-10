use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use polars::prelude::*;
use serde::Deserialize;

use qliber::logging;
use qliber::workflow::ExperimentManager;
use qliber::{MeanModel, Trainer};

#[derive(Debug, Deserialize)]
struct Config {
    task: String,
    label_column: Option<String>,
    epochs: Option<usize>,
}

fn main() -> Result<()> {
    logging::init_logging()?;
    let args: Vec<String> = env::args().collect();
    let mut config_path: Option<PathBuf> = None;
    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--config" | "-c" => {
                if idx + 1 < args.len() {
                    config_path = Some(PathBuf::from(&args[idx + 1]));
                    idx += 1;
                }
            }
            _ => {}
        }
        idx += 1;
    }

    let config = if let Some(path) = config_path {
        let contents = fs::read_to_string(&path)?;
        serde_json::from_str::<Config>(&contents)?
    } else {
        Config {
            task: "train".to_string(),
            label_column: Some("label".to_string()),
            epochs: Some(1),
        }
    };

    match config.task.as_str() {
        "train" => run_training(config)?,
        other => {
            eprintln!("Unsupported task: {other}");
        }
    }

    Ok(())
}

fn run_training(config: Config) -> Result<()> {
    let manager = ExperimentManager::new();
    let recorder = manager.start("cli-training");
    let label_column = config.label_column.unwrap_or_else(|| "label".to_string());
    let epochs = config.epochs.unwrap_or(1);

    let mut model = MeanModel::new(&label_column);
    let mut trainer = Trainer::new(recorder.clone(), &mut model, label_column.clone(), epochs);

    let features = DataFrame::new(vec![Series::new("feature", vec![1.0, 2.0, 3.0])])?;
    let labels = DataFrame::new(vec![Series::new(&label_column, vec![0.5, 1.5, 2.5])])?;
    trainer.train(&features, &labels)?;

    let snapshot = recorder.snapshot();
    println!("Training complete. Metrics: {:?}", snapshot.metrics);
    Ok(())
}

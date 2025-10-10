use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;
use crate::workflow::Recorder;

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("model error: {0}")]
    Model(String),
}

pub type TrainingResult<T> = Result<T, TrainingError>;

pub trait TrainableModel {
    fn fit(&mut self, features: &DataFrame, labels: &DataFrame) -> TrainingResult<()>;
    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame>;
}

#[derive(Debug, Default)]
pub struct MeanModel {
    label_column: String,
    mean: f64,
}

impl MeanModel {
    pub fn new(label_column: impl Into<String>) -> Self {
        Self {
            label_column: label_column.into(),
            mean: 0.0,
        }
    }
}

impl TrainableModel for MeanModel {
    fn fit(&mut self, _: &DataFrame, labels: &DataFrame) -> TrainingResult<()> {
        let series = labels
            .column(&self.label_column)
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        let chunked = series
            .f64()
            .map_err(|_| TrainingError::Model("label column must be f64".into()))?;
        let mut count = 0.0;
        let mut sum = 0.0;
        for value in chunked.into_iter().flatten() {
            count += 1.0;
            sum += value;
        }
        if count == 0.0 {
            return Err(TrainingError::Model("label column is empty".into()));
        }
        self.mean = sum / count;
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame> {
        let rows = features.height();
        let predictions = Float64Chunked::full("prediction", self.mean, rows).into_series();
        Ok(DataFrame::new(vec![predictions])?)
    }
}

pub struct Trainer<'a> {
    recorder: Recorder,
    model: &'a mut dyn TrainableModel,
    epochs: usize,
    label_column: String,
}

impl<'a> Trainer<'a> {
    pub fn new(
        recorder: Recorder,
        model: &'a mut dyn TrainableModel,
        label_column: impl Into<String>,
        epochs: usize,
    ) -> Self {
        Self {
            recorder,
            model,
            epochs,
            label_column: label_column.into(),
        }
    }

    pub fn train(&mut self, features: &DataFrame, labels: &DataFrame) -> TrainingResult<()> {
        self.model.fit(features, labels)?;
        for epoch in 0..self.epochs {
            let predictions = self.model.predict(features)?;
            let loss = mean_squared_error(labels, &predictions, &self.label_column, "prediction")?;
            self.recorder.log_metric(format!("epoch_{epoch}_mse"), loss);
            log_event(
                file!(),
                "Trainer",
                "train",
                "trainer.progress",
                line!(),
                &format!("Epoch {epoch} mse={loss:.6}"),
                None,
                "post",
                "PUT",
            );
        }
        Ok(())
    }

    pub fn label_column(&self) -> &str {
        &self.label_column
    }
}

fn mean_squared_error(
    labels: &DataFrame,
    predictions: &DataFrame,
    label_column: &str,
    prediction_column: &str,
) -> TrainingResult<f64> {
    let label_series = labels
        .column(label_column)
        .map_err(|err| TrainingError::Model(err.to_string()))?;
    let prediction_series = predictions
        .column(prediction_column)
        .map_err(|err| TrainingError::Model(err.to_string()))?;
    let label_chunked = label_series
        .f64()
        .map_err(|_| TrainingError::Model("labels must be f64".into()))?;
    let prediction_chunked = prediction_series
        .f64()
        .map_err(|_| TrainingError::Model("predictions must be f64".into()))?;

    let mut total = 0.0;
    let mut count = 0.0;
    for (label, pred) in label_chunked
        .into_iter()
        .zip(prediction_chunked.into_iter())
    {
        if let (Some(label), Some(pred)) = (label, pred) {
            let diff = label - pred;
            total += diff * diff;
            count += 1.0;
        }
    }

    if count == 0.0 {
        return Err(TrainingError::Model("no observations for mse".into()));
    }

    Ok(total / count)
}

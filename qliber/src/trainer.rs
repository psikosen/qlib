use polars::prelude::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::xgboost::{XGRegressor, XGRegressorParameters};
use thiserror::Error;

use crate::logging::log_event;
use crate::workflow::Recorder;

pub use smartcore::xgboost::XGRegressorParameters as XgBoostParameters;
pub use smartcore::xgboost::xgb_regressor::Objective as XgBoostObjective;

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

type SmartcoreRegressor = XGRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

pub struct XgBoostModel {
    label_column: String,
    feature_columns: Vec<String>,
    parameters: XGRegressorParameters,
    model: Option<SmartcoreRegressor>,
}

impl std::fmt::Debug for XgBoostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XgBoostModel")
            .field("label_column", &self.label_column)
            .field("feature_columns", &self.feature_columns)
            .field("parameters", &self.parameters)
            .field("is_trained", &self.model.is_some())
            .finish()
    }
}

impl XgBoostModel {
    pub fn new(label_column: impl Into<String>, feature_columns: impl Into<Vec<String>>) -> Self {
        Self {
            label_column: label_column.into(),
            feature_columns: feature_columns.into(),
            parameters: XGRegressorParameters::default(),
            model: None,
        }
    }

    pub fn with_parameters(mut self, parameters: XGRegressorParameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn set_parameters(&mut self, parameters: XGRegressorParameters) {
        self.parameters = parameters;
    }

    fn feature_matrix(&self, features: &DataFrame) -> TrainingResult<DenseMatrix<f64>> {
        if self.feature_columns.is_empty() {
            return Err(TrainingError::Model(
                "no feature columns configured for XgBoostModel".into(),
            ));
        }

        let mut columns = Vec::with_capacity(self.feature_columns.len());
        for name in &self.feature_columns {
            let series = features
                .column(name)
                .map_err(|err| TrainingError::Model(err.to_string()))?;
            let chunked = series
                .f64()
                .map_err(|_| TrainingError::Model(format!("feature column '{name}' must be f64")))?
                .clone();
            columns.push((name, chunked));
        }

        let row_count = features.height();
        let mut data = Vec::with_capacity(row_count);
        for row_idx in 0..row_count {
            let mut row = Vec::with_capacity(columns.len());
            for (name, column) in &columns {
                let value = column.get(row_idx).ok_or_else(|| {
                    TrainingError::Model(format!(
                        "null value encountered in feature column '{name}' at row {row_idx}"
                    ))
                })?;
                row.push(value);
            }
            data.push(row);
        }

        DenseMatrix::from_2d_vec(&data).map_err(|err| TrainingError::Model(err.to_string()))
    }

    fn label_vector(&self, labels: &DataFrame) -> TrainingResult<Vec<f64>> {
        let series = labels
            .column(&self.label_column)
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        let chunked = series.f64().map_err(|_| {
            TrainingError::Model(format!("label column '{}' must be f64", self.label_column))
        })?;
        let mut values = Vec::with_capacity(chunked.len());
        for (idx, value) in chunked.into_iter().enumerate() {
            let value = value.ok_or_else(|| {
                TrainingError::Model(format!(
                    "null value encountered in label column '{}' at row {idx}",
                    self.label_column
                ))
            })?;
            values.push(value);
        }
        Ok(values)
    }
}

impl TrainableModel for XgBoostModel {
    fn fit(&mut self, features: &DataFrame, labels: &DataFrame) -> TrainingResult<()> {
        if features.height() != labels.height() {
            return Err(TrainingError::Model(format!(
                "feature rows ({}) do not match label rows ({})",
                features.height(),
                labels.height()
            )));
        }

        let matrix = self.feature_matrix(features)?;
        let target = self.label_vector(labels)?;

        let model = XGRegressor::fit(&matrix, &target, self.parameters.clone())
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| TrainingError::Model("model is not fitted yet".into()))?;
        let matrix = self.feature_matrix(features)?;
        let predictions = model
            .predict(&matrix)
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        let series = Series::new("prediction", predictions);
        Ok(DataFrame::new(vec![series])?)
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

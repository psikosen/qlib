use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum MetaError {
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("missing column '{0}' in meta dataset")]
    MissingColumn(String),
    #[error("meta label column '{0}' contains null values")]
    NullLabel(String),
    #[error("prediction column '{0}' contains null values")]
    NullPrediction(String),
    #[error("meta learning matrix is singular")]
    SingularMatrix,
    #[error("no prediction columns provided")]
    EmptyPredictions,
}

pub type MetaResult<T> = Result<T, MetaError>;

#[derive(Debug, Clone)]
pub struct MetaLabelGenerator {
    label_column: String,
    prediction_columns: Vec<String>,
}

impl MetaLabelGenerator {
    pub fn new(
        label_column: impl Into<String>,
        prediction_columns: impl Into<Vec<String>>,
    ) -> Self {
        Self {
            label_column: label_column.into(),
            prediction_columns: prediction_columns.into(),
        }
    }

    pub fn label_column(&self) -> &str {
        &self.label_column
    }

    pub fn prediction_columns(&self) -> &[String] {
        &self.prediction_columns
    }

    pub fn generate(&self, frame: &DataFrame) -> MetaResult<DataFrame> {
        log_event(
            file!(),
            "MetaLabelGenerator",
            "generate",
            "meta.labels",
            line!(),
            "Generating meta labels from prediction columns",
            None,
            "pre",
            "POST",
        );

        if self.prediction_columns.is_empty() {
            return Err(MetaError::EmptyPredictions);
        }

        let label_series = frame
            .column(&self.label_column)
            .map_err(|_| MetaError::MissingColumn(self.label_column.clone()))?;
        let label_chunked = label_series
            .f64()
            .map_err(|_| MetaError::MissingColumn(self.label_column.clone()))?;

        if label_chunked.null_count() > 0 {
            return Err(MetaError::NullLabel(self.label_column.clone()));
        }

        let mut prediction_data = Vec::with_capacity(self.prediction_columns.len());
        for column in &self.prediction_columns {
            let series = frame
                .column(column)
                .map_err(|_| MetaError::MissingColumn(column.clone()))?;
            let chunked = series
                .f64()
                .map_err(|_| MetaError::MissingColumn(column.clone()))?;
            if chunked.null_count() > 0 {
                return Err(MetaError::NullPrediction(column.clone()));
            }
            prediction_data.push((column.clone(), chunked.clone()));
        }

        let row_count = frame.height();
        let mut winners = Vec::with_capacity(row_count);
        let mut errors = Vec::with_capacity(row_count);
        let mut error_columns = Vec::with_capacity(prediction_data.len());

        for (name, chunked) in &prediction_data {
            let mut values = Vec::with_capacity(row_count);
            for value in chunked.into_no_null_iter() {
                values.push(value);
            }
            error_columns.push((name.clone(), values));
        }

        for row_idx in 0..row_count {
            let label = label_chunked.get(row_idx).expect("non-null label");
            let mut best_error = f64::INFINITY;
            let mut best_model = String::new();

            for (name, values) in &error_columns {
                let prediction = values[row_idx];
                let diff = (prediction - label).abs();
                if diff < best_error {
                    best_error = diff;
                    best_model = name.clone();
                }
            }
            winners.push(best_model);
            errors.push(best_error);
        }

        let mut columns = Vec::with_capacity(2 + error_columns.len());
        columns.push(Series::new("winner_model", winners));
        columns.push(Series::new("winner_error", errors));
        for (name, values) in error_columns {
            let errs: Vec<f64> = values
                .into_iter()
                .enumerate()
                .map(|(idx, prediction)| {
                    let label = label_chunked.get(idx).expect("non-null label");
                    (prediction - label).abs()
                })
                .collect();
            let column_name = format!("{}_abs_error", name);
            columns.push(Series::new(&column_name, errs));
        }

        let result = DataFrame::new(columns)?;
        log_event(
            file!(),
            "MetaLabelGenerator",
            "generate",
            "meta.labels",
            line!(),
            "Generated meta labels and error diagnostics",
            None,
            "post",
            "POST",
        );
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct WeightLearner {
    enforce_non_negative: bool,
    normalize: bool,
    l2_regularization: f64,
}

impl WeightLearner {
    pub fn new() -> Self {
        Self {
            enforce_non_negative: true,
            normalize: true,
            l2_regularization: 1e-8,
        }
    }

    pub fn with_non_negative(mut self, enforce: bool) -> Self {
        self.enforce_non_negative = enforce;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn with_l2_regularization(mut self, value: f64) -> Self {
        self.l2_regularization = value.max(0.0);
        self
    }

    pub fn learn_weights(
        &self,
        frame: &DataFrame,
        prediction_columns: &[String],
        label_column: &str,
    ) -> MetaResult<Vec<f64>> {
        if prediction_columns.is_empty() {
            return Err(MetaError::EmptyPredictions);
        }

        log_event(
            file!(),
            "WeightLearner",
            "learn_weights",
            "meta.weights",
            line!(),
            "Learning ensemble weights via ridge regression",
            None,
            "pre",
            "POST",
        );

        let label_series = frame
            .column(label_column)
            .map_err(|_| MetaError::MissingColumn(label_column.to_string()))?;
        let label_chunked = label_series
            .f64()
            .map_err(|_| MetaError::MissingColumn(label_column.to_string()))?;
        if label_chunked.null_count() > 0 {
            return Err(MetaError::NullLabel(label_column.to_string()));
        }

        let row_count = frame.height();
        let mut design_data = Vec::with_capacity(row_count * prediction_columns.len());

        let mut prediction_series = Vec::with_capacity(prediction_columns.len());
        for column in prediction_columns {
            let series = frame
                .column(column)
                .map_err(|_| MetaError::MissingColumn(column.clone()))?;
            let chunked = series
                .f64()
                .map_err(|_| MetaError::MissingColumn(column.clone()))?;
            if chunked.null_count() > 0 {
                return Err(MetaError::NullPrediction(column.clone()));
            }
            let mut column_values = Vec::with_capacity(row_count);
            for value in chunked.into_no_null_iter() {
                column_values.push(value);
            }
            prediction_series.push(column_values);
        }

        for row_idx in 0..row_count {
            for column_values in &prediction_series {
                design_data.push(column_values[row_idx]);
            }
        }

        let design = DMatrix::from_row_slice(row_count, prediction_columns.len(), &design_data);
        let mut xtx = design.transpose() * &design;
        for idx in 0..prediction_columns.len() {
            xtx[(idx, idx)] += self.l2_regularization;
        }

        let mut label_values = Vec::with_capacity(row_count);
        for value in label_chunked.into_no_null_iter() {
            label_values.push(value);
        }
        let target = DVector::from_column_slice(&label_values);
        let xty = design.transpose() * target;
        let solution = xtx.lu().solve(&xty).ok_or(MetaError::SingularMatrix)?;

        let mut weights: Vec<f64> = solution.iter().copied().collect::<Vec<f64>>();
        if self.enforce_non_negative {
            for weight in &mut weights {
                if *weight < 0.0 {
                    *weight = 0.0;
                }
            }
        }

        if self.normalize {
            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            } else {
                let uniform = 1.0 / weights.len() as f64;
                weights.fill(uniform);
            }
        }

        log_event(
            file!(),
            "WeightLearner",
            "learn_weights",
            "meta.weights",
            line!(),
            &format!("Learned {} ensemble weights", weights.len()),
            None,
            "post",
            "POST",
        );

        Ok(weights)
    }
}

impl Default for WeightLearner {
    fn default() -> Self {
        Self::new()
    }
}

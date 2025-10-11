use std::cmp::Ordering;

use polars::prelude::*;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::logging::log_event;
use crate::trainer::{TrainableModel, TrainingError, TrainingResult, mean_squared_error};

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureImportance {
    pub feature: String,
    pub importance: f64,
}

pub trait FeatureInterpreter {
    fn feature_importance(
        &mut self,
        features: &DataFrame,
        labels: &DataFrame,
    ) -> TrainingResult<Vec<FeatureImportance>>;
}

pub struct PermutationFeatureInterpreter<'a> {
    model: &'a dyn TrainableModel,
    label_column: String,
    prediction_column: String,
    repeats: usize,
    random_seed: Option<u64>,
    feature_columns: Option<Vec<String>>,
}

impl<'a> PermutationFeatureInterpreter<'a> {
    pub fn new(model: &'a dyn TrainableModel, label_column: impl Into<String>) -> Self {
        Self {
            model,
            label_column: label_column.into(),
            prediction_column: "prediction".to_string(),
            repeats: 5,
            random_seed: None,
            feature_columns: None,
        }
    }

    pub fn with_prediction_column(mut self, column: impl Into<String>) -> Self {
        self.prediction_column = column.into();
        self
    }

    pub fn with_repeats(mut self, repeats: usize) -> Self {
        self.repeats = repeats.max(1);
        self
    }

    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    pub fn with_feature_columns(mut self, columns: impl Into<Vec<String>>) -> Self {
        self.feature_columns = Some(columns.into());
        self
    }

    fn resolve_columns(&self, features: &DataFrame) -> TrainingResult<Vec<String>> {
        if let Some(columns) = &self.feature_columns {
            if columns.is_empty() {
                return Err(TrainingError::Model(
                    "permutation interpreter requires at least one feature column".into(),
                ));
            }
            return Ok(columns.clone());
        }

        Ok(features
            .get_columns()
            .iter()
            .map(|series| series.name().to_string())
            .collect())
    }
}

impl<'a> FeatureInterpreter for PermutationFeatureInterpreter<'a> {
    fn feature_importance(
        &mut self,
        features: &DataFrame,
        labels: &DataFrame,
    ) -> TrainingResult<Vec<FeatureImportance>> {
        if self.label_column.is_empty() {
            return Err(TrainingError::Model(
                "permutation interpreter requires a label column".into(),
            ));
        }

        let columns = self.resolve_columns(features)?;
        if columns.is_empty() {
            return Err(TrainingError::Model(
                "permutation interpreter received no feature columns".into(),
            ));
        }

        let baseline_predictions = self.model.predict(features)?;
        let baseline_loss = mean_squared_error(
            labels,
            &baseline_predictions,
            &self.label_column,
            &self.prediction_column,
        )?;

        let mut rng = match self.random_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        let mut importances = Vec::with_capacity(columns.len());
        for name in columns {
            let series = features
                .column(&name)
                .map_err(|err| TrainingError::Model(err.to_string()))?;
            let chunked = series.f64().map_err(|_| {
                TrainingError::Model(format!(
                    "permutation interpreter expects f64 feature column '{name}'"
                ))
            })?;

            if chunked.null_count() > 0 {
                return Err(TrainingError::Model(format!(
                    "feature column '{name}' contains null values and cannot be permuted"
                )));
            }

            let mut base_values = Vec::with_capacity(chunked.len());
            for (idx, value) in chunked.into_iter().enumerate() {
                let value = value.ok_or_else(|| {
                    TrainingError::Model(format!(
                        "feature column '{name}' contains null value at row {idx}"
                    ))
                })?;
                base_values.push(value);
            }

            let mut delta = 0.0;
            for repeat in 0..self.repeats {
                let mut permuted = base_values.clone();
                permuted.shuffle(&mut rng);

                let mut permuted_df = features.clone();
                permuted_df
                    .with_column(Series::new(&name, permuted))
                    .map_err(|err| TrainingError::Model(err.to_string()))?;

                let predictions = self.model.predict(&permuted_df)?;
                let loss = mean_squared_error(
                    labels,
                    &predictions,
                    &self.label_column,
                    &self.prediction_column,
                )?;
                delta += loss - baseline_loss;

                log_event(
                    file!(),
                    "PermutationFeatureInterpreter",
                    "feature_importance",
                    "trainer.interpret",
                    line!(),
                    &format!(
                        "Permuted feature '{name}' repeat {repeat} delta={:.6}",
                        loss - baseline_loss
                    ),
                    None,
                    "post",
                    "PUT",
                );
            }

            let importance = delta / self.repeats as f64;
            importances.push(FeatureImportance {
                feature: name.clone(),
                importance,
            });
        }

        importances.sort_by(|left, right| {
            right
                .importance
                .partial_cmp(&left.importance)
                .unwrap_or(Ordering::Equal)
        });

        log_event(
            file!(),
            "PermutationFeatureInterpreter",
            "feature_importance",
            "trainer.interpret",
            line!(),
            &format!(
                "Computed permutation importances for {} features",
                importances.len()
            ),
            None,
            "post",
            "GET",
        );

        Ok(importances)
    }
}

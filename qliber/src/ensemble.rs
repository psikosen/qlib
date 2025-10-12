use polars::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::logging::log_event;
use crate::meta::WeightLearner;
use crate::trainer::{
    GLOBAL_TRAINER_REGISTRY, TrainableModel, TrainerAdapter, TrainerRequest, TrainingError,
    TrainingResult,
};

struct EnsembleMember {
    name: String,
    feature_columns: Vec<String>,
    label_column: String,
    model: Box<dyn TrainableModel>,
}

impl EnsembleMember {
    fn new(
        name: impl Into<String>,
        feature_columns: Vec<String>,
        label_column: String,
        model: Box<dyn TrainableModel>,
    ) -> Self {
        Self {
            name: name.into(),
            feature_columns,
            label_column,
            model,
        }
    }

    fn select_features(&self, frame: &DataFrame) -> TrainingResult<DataFrame> {
        if self.feature_columns.is_empty() {
            return Ok(frame.clone());
        }
        let names: Vec<&str> = self
            .feature_columns
            .iter()
            .map(|value| value.as_str())
            .collect();
        frame
            .select(&names)
            .map_err(|err| TrainingError::Model(err.to_string()))
    }
}

#[derive(Clone)]
struct WeightLearningOptions {
    enforce_non_negative: bool,
    l2_regularization: f64,
}

pub struct WeightedEnsemble {
    members: Vec<EnsembleMember>,
    weights: Vec<f64>,
    label_column: String,
    normalize: bool,
    learn_options: Option<WeightLearningOptions>,
}

impl std::fmt::Debug for EnsembleMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleMember")
            .field("name", &self.name)
            .field("feature_columns", &self.feature_columns)
            .field("label_column", &self.label_column)
            .finish()
    }
}

impl std::fmt::Debug for WeightLearningOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightLearningOptions")
            .field("enforce_non_negative", &self.enforce_non_negative)
            .field("l2_regularization", &self.l2_regularization)
            .finish()
    }
}

impl std::fmt::Debug for WeightedEnsemble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightedEnsemble")
            .field("members", &self.members)
            .field("weights", &self.weights)
            .field("label_column", &self.label_column)
            .field("normalize", &self.normalize)
            .field("learn_options", &self.learn_options)
            .finish()
    }
}

impl WeightedEnsemble {
    fn new(
        members: Vec<EnsembleMember>,
        weights: Vec<f64>,
        label_column: impl Into<String>,
        normalize: bool,
        learn_options: Option<WeightLearningOptions>,
    ) -> TrainingResult<Self> {
        if members.is_empty() {
            return Err(TrainingError::Model(
                "ensemble requires at least one member".into(),
            ));
        }
        if weights.len() != members.len() {
            return Err(TrainingError::Model(
                "ensemble weights do not match member count".into(),
            ));
        }

        let mut ensemble = Self {
            members,
            weights,
            label_column: label_column.into(),
            normalize,
            learn_options,
        };
        if ensemble.normalize {
            ensemble.normalize_weights();
        }
        Ok(ensemble)
    }

    pub fn from_models(
        models: Vec<(String, Vec<String>, Box<dyn TrainableModel>)>,
        weights: Vec<f64>,
        label_column: impl Into<String>,
        normalize: bool,
    ) -> TrainingResult<Self> {
        let label = label_column.into();
        let members = models
            .into_iter()
            .map(|(name, features, model)| {
                EnsembleMember::new(name, features, label.clone(), model)
            })
            .collect();
        Self::new(members, weights, label, normalize, None)
    }

    pub fn with_weight_learning(
        mut self,
        enforce_non_negative: bool,
        l2_regularization: f64,
    ) -> Self {
        self.learn_options = Some(WeightLearningOptions {
            enforce_non_negative,
            l2_regularization,
        });
        self
    }

    fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.iter().copied().sum();
        if sum > 0.0 {
            for weight in &mut self.weights {
                *weight /= sum;
            }
        } else {
            let uniform = 1.0 / self.weights.len() as f64;
            for weight in &mut self.weights {
                *weight = uniform;
            }
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn prediction_column_name(index: usize, name: &str) -> String {
        format!("member_{index}_{}_prediction", name.replace(' ', "_"))
    }

    fn prepare_learning_frame(
        &self,
        features: &DataFrame,
        labels: &DataFrame,
    ) -> TrainingResult<(DataFrame, Vec<String>)> {
        let mut columns = Vec::with_capacity(self.members.len() + 1);

        let label_series = labels
            .column(&self.label_column)
            .map_err(|err| TrainingError::Model(err.to_string()))?
            .clone();
        columns.push(label_series.clone());

        let mut prediction_columns = Vec::with_capacity(self.members.len());
        for (idx, member) in self.members.iter().enumerate() {
            let member_features = member.select_features(features)?;
            let predictions = member.model.predict(&member_features)?;
            let series = predictions
                .column("prediction")
                .map_err(|err| TrainingError::Model(err.to_string()))?
                .f64()
                .map_err(|_| {
                    TrainingError::Model(format!(
                        "ensemble member '{}' returned non-f64 predictions",
                        member.name
                    ))
                })?
                .clone();
            if series.null_count() > 0 {
                return Err(TrainingError::Model(format!(
                    "ensemble member '{}' produced null predictions",
                    member.name
                )));
            }
            let column_name = Self::prediction_column_name(idx, &member.name);
            prediction_columns.push(column_name.clone());
            let mut renamed = series.into_series();
            renamed.rename(&column_name);
            columns.push(renamed);
        }

        let frame = DataFrame::new(columns).map_err(|err| TrainingError::Model(err.to_string()))?;
        Ok((frame, prediction_columns))
    }
}

impl TrainableModel for WeightedEnsemble {
    fn fit(&mut self, features: &DataFrame, labels: &DataFrame) -> TrainingResult<()> {
        log_event(
            file!(),
            "WeightedEnsemble",
            "fit",
            "trainer.ensemble",
            line!(),
            "Training ensemble members",
            None,
            "pre",
            "PUT",
        );

        for member in &mut self.members {
            let member_features = member.select_features(features)?;
            member.model.fit(&member_features, labels)?;
        }

        if let Some(options) = &self.learn_options {
            let (learning_frame, prediction_columns) =
                self.prepare_learning_frame(features, labels)?;
            let learner = WeightLearner::new()
                .with_non_negative(options.enforce_non_negative)
                .with_normalize(true)
                .with_l2_regularization(options.l2_regularization);
            match learner.learn_weights(&learning_frame, &prediction_columns, &self.label_column) {
                Ok(weights) => {
                    self.weights = weights;
                }
                Err(error) => {
                    log_event(
                        file!(),
                        "WeightedEnsemble",
                        "fit",
                        "trainer.ensemble",
                        line!(),
                        &format!(
                            "Falling back to configured weights after learning error: {error}"
                        ),
                        Some("meta"),
                        "post",
                        "POST",
                    );
                }
            }
        }

        if self.normalize {
            self.normalize_weights();
        }

        log_event(
            file!(),
            "WeightedEnsemble",
            "fit",
            "trainer.ensemble",
            line!(),
            &format!("Trained ensemble with {} members", self.members.len()),
            None,
            "post",
            "PUT",
        );
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame> {
        log_event(
            file!(),
            "WeightedEnsemble",
            "predict",
            "trainer.ensemble",
            line!(),
            "Aggregating ensemble predictions",
            None,
            "pre",
            "POST",
        );

        let row_count = features.height();
        let mut combined = vec![0.0; row_count];

        for (member, weight) in self.members.iter().zip(self.weights.iter()) {
            let member_features = member.select_features(features)?;
            let predictions = member.model.predict(&member_features)?;
            let series = predictions
                .column("prediction")
                .map_err(|err| TrainingError::Model(err.to_string()))?
                .f64()
                .map_err(|_| {
                    TrainingError::Model(format!(
                        "ensemble member '{}' returned non-f64 predictions",
                        member.name
                    ))
                })?;

            for (idx, value) in series.into_iter().enumerate() {
                let value = value.ok_or_else(|| {
                    TrainingError::Model(format!(
                        "ensemble member '{}' produced null predictions",
                        member.name
                    ))
                })?;
                combined[idx] += weight * value;
            }
        }

        let result = DataFrame::new(vec![Series::new("prediction", combined)])
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        log_event(
            file!(),
            "WeightedEnsemble",
            "predict",
            "trainer.ensemble",
            line!(),
            "Aggregated ensemble predictions",
            None,
            "post",
            "POST",
        );
        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
struct EnsembleMemberConfig {
    adapter: String,
    weight: Option<f64>,
    parameters: Option<Value>,
    feature_columns: Option<Vec<String>>,
    label_column: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EnsembleAdapterConfig {
    members: Vec<EnsembleMemberConfig>,
    normalize: Option<bool>,
    learn_weights: Option<bool>,
    non_negative: Option<bool>,
    l2_regularization: Option<f64>,
}

#[derive(Default)]
pub struct EnsembleTrainerAdapter;

impl TrainerAdapter for EnsembleTrainerAdapter {
    fn create(&self, request: &TrainerRequest) -> TrainingResult<Box<dyn TrainableModel>> {
        if request.label_column().is_empty() {
            return Err(TrainingError::Model(
                "ensemble adapter requires a label column".into(),
            ));
        }
        if request.feature_columns().is_empty() {
            return Err(TrainingError::Model(
                "ensemble adapter requires at least one feature column".into(),
            ));
        }

        let config: EnsembleAdapterConfig = if request.parameters().is_null() {
            return Err(TrainingError::Model(
                "ensemble adapter requires member configuration".into(),
            ));
        } else {
            serde_json::from_value(request.parameters().clone()).map_err(|err| {
                TrainingError::Model(format!("invalid ensemble parameters: {err}"))
            })?
        };

        if config.members.is_empty() {
            return Err(TrainingError::Model(
                "ensemble adapter requires at least one member".into(),
            ));
        }

        let mut members = Vec::with_capacity(config.members.len());
        let mut weights = Vec::with_capacity(config.members.len());

        for (index, member_cfg) in config.members.into_iter().enumerate() {
            let adapter_name = member_cfg.adapter;
            let weight = member_cfg.weight.unwrap_or(1.0);
            let label_column = member_cfg
                .label_column
                .unwrap_or_else(|| request.label_column().to_string());
            if label_column.is_empty() {
                return Err(TrainingError::Model(format!(
                    "ensemble member '{adapter_name}' must provide a label column"
                )));
            }
            let feature_columns = member_cfg
                .feature_columns
                .unwrap_or_else(|| request.feature_columns().to_vec());
            if feature_columns.is_empty() {
                return Err(TrainingError::Model(format!(
                    "ensemble member '{adapter_name}' must provide feature columns"
                )));
            }

            let mut child_request =
                TrainerRequest::new(label_column.clone(), feature_columns.clone());
            if let Some(parameters) = member_cfg.parameters {
                child_request = child_request.with_parameters(parameters);
            }

            let model = GLOBAL_TRAINER_REGISTRY.create(&adapter_name, &child_request)?;
            let member_name = member_cfg
                .name
                .unwrap_or_else(|| format!("{}_{}", adapter_name, index));
            members.push(EnsembleMember::new(
                member_name,
                feature_columns,
                label_column,
                model,
            ));
            weights.push(weight);
        }

        let learn_options = if config.learn_weights.unwrap_or(false) {
            Some(WeightLearningOptions {
                enforce_non_negative: config.non_negative.unwrap_or(true),
                l2_regularization: config.l2_regularization.unwrap_or(1e-8),
            })
        } else {
            None
        };

        let model = WeightedEnsemble::new(
            members,
            weights,
            request.label_column().to_string(),
            config.normalize.unwrap_or(true),
            learn_options,
        )?;

        Ok(Box::new(model))
    }
}

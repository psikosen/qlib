use std::sync::Arc;

use polars::prelude::*;
use serde_json::json;

use qliber::{
    ExperimentManager, GLOBAL_TRAINER_REGISTRY, MeanModel, TrainableModel, Trainer, TrainerAdapter,
    TrainerRequest, TrainingError, TrainingResult, XgBoostModel, XgBoostParameters,
};

struct DummyAdapter;

impl TrainerAdapter for DummyAdapter {
    fn create(&self, request: &TrainerRequest) -> TrainingResult<Box<dyn TrainableModel>> {
        if request.label_column().is_empty() {
            return Err(TrainingError::Model(
                "dummy adapter requires a label column".into(),
            ));
        }
        Ok(Box::new(MeanModel::new(request.label_column().to_string())))
    }
}

#[test]
fn xgboost_model_fits_linear_series() -> anyhow::Result<()> {
    let feature_values: Vec<f64> = (0..100).map(|v| v as f64).collect();
    let label_values: Vec<f64> = feature_values.iter().map(|v| 2.0 * v + 1.0).collect();

    let features = DataFrame::new(vec![Series::new("feature", feature_values.clone())])?;
    let labels = DataFrame::new(vec![Series::new("label", label_values.clone())])?;

    let manager = ExperimentManager::new();
    let recorder = manager.start("xgboost-linear");

    let params = XgBoostParameters::default()
        .with_n_estimators(200)
        .with_learning_rate(0.1)
        .with_max_depth(5)
        .with_min_child_weight(1)
        .with_lambda(1.0)
        .with_gamma(0.0)
        .with_subsample(1.0)
        .with_seed(42);

    let mut model = XgBoostModel::new("label", vec!["feature".to_string()]).with_parameters(params);
    let mut trainer = Trainer::new(recorder.clone(), &mut model, "label", 1);
    trainer.train(&features, &labels)?;
    drop(trainer);

    let predictions = model.predict(&features)?;
    let predicted = predictions
        .column("prediction")?
        .f64()
        .map_err(|_| anyhow::anyhow!("prediction column must be f64"))?;
    let truth = labels
        .column("label")?
        .f64()
        .map_err(|_| anyhow::anyhow!("label column must be f64"))?;

    let mut mse = 0.0;
    let mut count = 0.0;
    for (pred, actual) in predicted.into_iter().zip(truth.into_iter()) {
        let pred = pred.ok_or_else(|| anyhow::anyhow!("prediction contains null"))?;
        let actual = actual.ok_or_else(|| anyhow::anyhow!("label contains null"))?;
        let diff = pred - actual;
        mse += diff * diff;
        count += 1.0;
    }
    assert!(count > 0.0);
    mse /= count;

    assert!(mse < 1e-2, "mse was {mse}");
    Ok(())
}

#[test]
fn registry_supports_custom_adapter() -> anyhow::Result<()> {
    let adapter_name = "dummy-test-adapter";
    let registry = &*GLOBAL_TRAINER_REGISTRY;
    let _ = registry.unregister_adapter(adapter_name);
    registry.register_adapter(adapter_name, Arc::new(DummyAdapter));

    let features = DataFrame::new(vec![Series::new("feature", vec![1.0, 2.0, 3.0])])?;
    let labels = DataFrame::new(vec![Series::new("label", vec![2.0, 4.0, 6.0])])?;

    let request = TrainerRequest::new("label", Vec::<String>::new());
    let mut model = registry.create(adapter_name, &request)?;
    model.fit(&features, &labels)?;
    let predictions = model.predict(&features)?;
    assert_eq!(predictions.height(), features.height());

    assert!(registry.unregister_adapter(adapter_name).is_some());
    Ok(())
}

#[test]
fn registry_creates_xgboost_adapter_with_parameters() -> anyhow::Result<()> {
    let feature_values: Vec<f64> = (0..50).map(|v| v as f64).collect();
    let label_values: Vec<f64> = feature_values.iter().map(|v| 3.0 * v - 2.0).collect();

    let features = DataFrame::new(vec![Series::new("feature", feature_values.clone())])?;
    let labels = DataFrame::new(vec![Series::new("label", label_values.clone())])?;

    let request =
        TrainerRequest::new("label", vec!["feature".to_string()]).with_parameters(json!({
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 4,
            "seed": 13,
        }));

    let registry = &*GLOBAL_TRAINER_REGISTRY;
    let mut model = registry.create("xgboost", &request)?;
    model.fit(&features, &labels)?;

    let predictions = model.predict(&features)?;
    let prediction_series = predictions
        .column("prediction")?
        .f64()
        .map_err(|_| anyhow::anyhow!("prediction column must be f64"))?;
    let truth = labels
        .column("label")?
        .f64()
        .map_err(|_| anyhow::anyhow!("label column must be f64"))?;

    let mut mse = 0.0;
    let mut count = 0.0;
    for (pred, actual) in prediction_series.into_iter().zip(truth.into_iter()) {
        let pred = pred.ok_or_else(|| anyhow::anyhow!("prediction contains null"))?;
        let actual = actual.ok_or_else(|| anyhow::anyhow!("label contains null"))?;
        let diff = pred - actual;
        mse += diff * diff;
        count += 1.0;
    }
    assert!(count > 0.0);
    mse /= count;
    assert!(mse < 0.1, "mse was {mse}");
    Ok(())
}

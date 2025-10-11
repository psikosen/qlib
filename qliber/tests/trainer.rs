use polars::prelude::*;

use qliber::{ExperimentManager, TrainableModel, Trainer, XgBoostModel, XgBoostParameters};

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

use polars::prelude::*;

use qliber::{
    ExperimentManager, FeatureInterpreter, PermutationFeatureInterpreter, Trainer, XgBoostModel,
    XgBoostParameters,
};

#[test]
fn permutation_importance_prioritizes_signal_feature() -> anyhow::Result<()> {
    let signal: Vec<f64> = (0..80).map(|value| value as f64).collect();
    let noise: Vec<f64> = signal.iter().map(|value| (value * 0.5).sin()).collect();
    let labels: Vec<f64> = signal.iter().map(|value| 4.0 * value + 3.0).collect();

    let features = DataFrame::new(vec![
        Series::new("signal", signal.clone()),
        Series::new("noise", noise.clone()),
    ])?;
    let labels_df = DataFrame::new(vec![Series::new("label", labels.clone())])?;

    let manager = ExperimentManager::new();
    let recorder = manager.start("permutation-importance");

    let mut model = XgBoostModel::new("label", vec!["signal".into(), "noise".into()])
        .with_parameters(
            XgBoostParameters::default()
                .with_n_estimators(150)
                .with_seed(7),
        );
    let mut trainer = Trainer::new(recorder, &mut model, "label", 1);
    trainer.train(&features, &labels_df)?;
    drop(trainer);

    let mut interpreter = PermutationFeatureInterpreter::new(&model, "label")
        .with_feature_columns(vec!["signal".to_string(), "noise".to_string()])
        .with_random_seed(42)
        .with_repeats(10);
    let importances = interpreter.feature_importance(&features, &labels_df)?;

    assert_eq!(importances.len(), 2);
    assert_eq!(importances[0].feature, "signal");
    assert!(importances[0].importance > importances[1].importance);
    Ok(())
}

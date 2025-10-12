use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{
    MetaLabelGenerator, TrainableModel, TrainingError, TrainingResult, WeightLearner,
    WeightedEnsemble,
};

struct ConstantModel {
    value: f64,
}

impl TrainableModel for ConstantModel {
    fn fit(&mut self, _: &DataFrame, _: &DataFrame) -> TrainingResult<()> {
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame> {
        let predictions = vec![self.value; features.height()];
        Ok(DataFrame::new(vec![Series::new("prediction", predictions)])
            .map_err(|err| TrainingError::Model(err.to_string()))?)
    }
}

struct LinearModel {
    column: String,
    scale: f64,
}

impl TrainableModel for LinearModel {
    fn fit(&mut self, _: &DataFrame, _: &DataFrame) -> TrainingResult<()> {
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> TrainingResult<DataFrame> {
        let series = features
            .column(&self.column)
            .map_err(|err| TrainingError::Model(err.to_string()))?;
        let chunked = series
            .f64()
            .map_err(|_| TrainingError::Model("feature column must be f64".into()))?;
        if chunked.null_count() > 0 {
            return Err(TrainingError::Model("feature column contains null".into()));
        }
        let values: Vec<f64> = chunked
            .into_no_null_iter()
            .map(|value| value * self.scale)
            .collect();
        Ok(DataFrame::new(vec![Series::new("prediction", values)])
            .map_err(|err| TrainingError::Model(err.to_string()))?)
    }
}

#[test]
fn weighted_ensemble_combines_predictions() -> anyhow::Result<()> {
    let features = df! { "x" => &[1.0, 2.0, 3.0] }?;
    let labels = df! { "label" => &[0.0, 0.0, 0.0] }?;

    let models = vec![
        (
            "constant_a".to_string(),
            vec!["x".to_string()],
            Box::new(ConstantModel { value: 1.0 }) as Box<dyn TrainableModel>,
        ),
        (
            "constant_b".to_string(),
            vec!["x".to_string()],
            Box::new(ConstantModel { value: 2.0 }) as Box<dyn TrainableModel>,
        ),
    ];

    let mut ensemble = WeightedEnsemble::from_models(models, vec![0.25, 0.75], "label", true)?;
    ensemble.fit(&features, &labels)?;
    let predictions = ensemble.predict(&features)?;
    let values: Vec<f64> = predictions
        .column("prediction")?
        .f64()?
        .into_no_null_iter()
        .collect();
    assert_eq!(values, vec![1.75, 1.75, 1.75]);
    Ok(())
}

#[test]
fn weighted_ensemble_learns_weights() -> anyhow::Result<()> {
    let features = df! { "x" => &[1.0, 2.0, 3.0, 4.0] }?;
    let labels = df! { "label" => &[1.0, 1.8, 2.6, 3.4] }?;

    let models = vec![
        (
            "constant".to_string(),
            vec!["x".to_string()],
            Box::new(ConstantModel { value: 1.0 }) as Box<dyn TrainableModel>,
        ),
        (
            "linear".to_string(),
            vec!["x".to_string()],
            Box::new(LinearModel {
                column: "x".to_string(),
                scale: 1.0,
            }) as Box<dyn TrainableModel>,
        ),
    ];

    let mut ensemble = WeightedEnsemble::from_models(models, vec![0.5, 0.5], "label", true)?
        .with_weight_learning(true, 1e-8);
    ensemble.fit(&features, &labels)?;
    let weights = ensemble.weights();
    assert_eq!(weights.len(), 2);
    assert_relative_eq!(weights[0], 0.2, epsilon = 1e-4);
    assert_relative_eq!(weights[1], 0.8, epsilon = 1e-4);
    Ok(())
}

#[test]
fn meta_label_generator_marks_winners() -> anyhow::Result<()> {
    let frame = df! {
        "label" => &[1.0, -1.0, 0.5],
        "model_a" => &[0.8, -0.85, 0.6],
        "model_b" => &[1.05, -1.25, 0.4],
        "model_c" => &[0.88, -1.1, 0.55],
    }?;

    let generator = MetaLabelGenerator::new(
        "label",
        vec![
            "model_a".to_string(),
            "model_b".to_string(),
            "model_c".to_string(),
        ],
    );
    let result = generator.generate(&frame)?;

    let winners: Vec<&str> = result
        .column("winner_model")?
        .utf8()?
        .into_no_null_iter()
        .collect();
    assert_eq!(winners, vec!["model_b", "model_c", "model_c"]);

    let errors: Vec<f64> = result
        .column("winner_error")?
        .f64()?
        .into_no_null_iter()
        .collect();
    assert_relative_eq!(errors[0], 0.05, epsilon = 1e-9);
    assert_relative_eq!(errors[1], 0.1, epsilon = 1e-9);
    assert_relative_eq!(errors[2], 0.05, epsilon = 1e-9);

    let learner = WeightLearner::new();
    let weights = learner.learn_weights(
        &frame,
        &vec![
            "model_a".to_string(),
            "model_b".to_string(),
            "model_c".to_string(),
        ],
        "label",
    )?;
    assert_eq!(weights.len(), 3);
    let sum: f64 = weights.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-9);
    Ok(())
}

use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{MetaError, MetaLabelGenerator, WeightLearner};

#[test]
fn test_meta_label_generator_basic() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred_a" => &[12.0, 18.0, 35.0],
        "pred_b" => &[9.0, 22.0, 28.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("label", vec!["pred_a".to_string(), "pred_b".to_string()]);
    let result = generator.generate(&df).unwrap();

    // Should have winner_model, winner_error, and error columns for each predictor
    assert!(result.column("winner_model").is_ok());
    assert!(result.column("winner_error").is_ok());
    assert!(result.column("pred_a_abs_error").is_ok());
    assert!(result.column("pred_b_abs_error").is_ok());

    let winners = result.column("winner_model").unwrap().utf8().unwrap();

    // Row 0: label=10, pred_a=12 (err=2), pred_b=9 (err=1) -> pred_b wins
    assert_eq!(winners.get(0).unwrap(), "pred_b");

    // Row 1: label=20, pred_a=18 (err=2), pred_b=22 (err=2) -> tie (first wins)
    assert!(winners.get(1).unwrap() == "pred_a" || winners.get(1).unwrap() == "pred_b");

    // Row 2: label=30, pred_a=35 (err=5), pred_b=28 (err=2) -> pred_b wins
    assert_eq!(winners.get(2).unwrap(), "pred_b");
}

#[test]
fn test_meta_label_generator_error_columns() {
    let df = df! {
        "label" => &[100.0, 200.0],
        "model1" => &[110.0, 190.0],
        "model2" => &[95.0, 210.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("label", vec!["model1".to_string(), "model2".to_string()]);
    let result = generator.generate(&df).unwrap();

    let model1_errors = result.column("model1_abs_error").unwrap().f64().unwrap();
    let model2_errors = result.column("model2_abs_error").unwrap().f64().unwrap();

    // Row 0: label=100, model1=110 (err=10), model2=95 (err=5)
    assert_relative_eq!(model1_errors.get(0).unwrap(), 10.0, epsilon = 1e-10);
    assert_relative_eq!(model2_errors.get(0).unwrap(), 5.0, epsilon = 1e-10);

    // Row 1: label=200, model1=190 (err=10), model2=210 (err=10)
    assert_relative_eq!(model1_errors.get(1).unwrap(), 10.0, epsilon = 1e-10);
    assert_relative_eq!(model2_errors.get(1).unwrap(), 10.0, epsilon = 1e-10);
}

#[test]
fn test_meta_label_generator_winner_error() {
    let df = df! {
        "label" => &[50.0, 100.0],
        "pred_a" => &[60.0, 95.0],
        "pred_b" => &[45.0, 110.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("label", vec!["pred_a".to_string(), "pred_b".to_string()]);
    let result = generator.generate(&df).unwrap();

    let winner_errors = result.column("winner_error").unwrap().f64().unwrap();

    // Row 0: pred_a err=10, pred_b err=5 -> winner_error=5
    assert_relative_eq!(winner_errors.get(0).unwrap(), 5.0, epsilon = 1e-10);

    // Row 1: pred_a err=5, pred_b err=10 -> winner_error=5
    assert_relative_eq!(winner_errors.get(1).unwrap(), 5.0, epsilon = 1e-10);
}

#[test]
fn test_meta_label_generator_empty_predictions() {
    let df = df! {
        "label" => &[10.0, 20.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("label", Vec::<String>::new());
    let result = generator.generate(&df);

    assert!(result.is_err());
    match result {
        Err(MetaError::EmptyPredictions) => {}
        _ => panic!("Expected EmptyPredictions error"),
    }
}

#[test]
fn test_meta_label_generator_missing_label_column() {
    let df = df! {
        "pred_a" => &[10.0, 20.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("missing_label", vec!["pred_a".to_string()]);
    let result = generator.generate(&df);

    assert!(result.is_err());
    match result {
        Err(MetaError::MissingColumn(col)) => {
            assert_eq!(col, "missing_label");
        }
        _ => panic!("Expected MissingColumn error"),
    }
}

#[test]
fn test_meta_label_generator_missing_prediction_column() {
    let df = df! {
        "label" => &[10.0, 20.0]
    }
    .unwrap();

    let generator = MetaLabelGenerator::new("label", vec!["missing_pred".to_string()]);
    let result = generator.generate(&df);

    assert!(result.is_err());
    match result {
        Err(MetaError::MissingColumn(col)) => {
            assert_eq!(col, "missing_pred");
        }
        _ => panic!("Expected MissingColumn error"),
    }
}

#[test]
fn test_meta_label_generator_getters() {
    let generator = MetaLabelGenerator::new(
        "my_label",
        vec!["pred1".to_string(), "pred2".to_string()]
    );

    assert_eq!(generator.label_column(), "my_label");
    assert_eq!(generator.prediction_columns().len(), 2);
    assert_eq!(generator.prediction_columns()[0], "pred1");
    assert_eq!(generator.prediction_columns()[1], "pred2");
}

#[test]
fn test_weight_learner_basic() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0, 40.0],
        "pred_a" => &[12.0, 18.0, 32.0, 38.0],
        "pred_b" => &[8.0, 22.0, 28.0, 42.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let weights = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string(), "pred_b".to_string()],
        "label"
    ).unwrap();

    assert_eq!(weights.len(), 2);

    // Weights should be non-negative (default enforcement)
    for weight in &weights {
        assert!(*weight >= 0.0, "Weight should be non-negative");
    }

    // Weights should sum to 1.0 (default normalization)
    let sum: f64 = weights.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

#[test]
fn test_weight_learner_perfect_predictor() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0, 40.0, 50.0],
        "perfect" => &[10.0, 20.0, 30.0, 40.0, 50.0],
        "bad" => &[0.0, 0.0, 0.0, 0.0, 0.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let weights = learner.learn_weights(
        &df,
        &vec!["perfect".to_string(), "bad".to_string()],
        "label"
    ).unwrap();

    // The perfect predictor should get most or all of the weight
    assert!(weights[0] > 0.9, "Perfect predictor should dominate: got {}", weights[0]);
}

#[test]
fn test_weight_learner_without_normalization() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred_a" => &[11.0, 19.0, 31.0],
        "pred_b" => &[9.0, 21.0, 29.0]
    }
    .unwrap();

    let learner = WeightLearner::new().with_normalize(false);
    let weights = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string(), "pred_b".to_string()],
        "label"
    ).unwrap();

    // Weights might not sum to 1.0
    let sum: f64 = weights.iter().sum();
    // Just verify we got weights (sum might not be 1.0)
    assert!(sum > 0.0);
}

#[test]
fn test_weight_learner_allow_negative() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0, 40.0],
        "pred_a" => &[15.0, 25.0, 35.0, 45.0],
        "pred_b" => &[5.0, 15.0, 25.0, 35.0]
    }
    .unwrap();

    let learner = WeightLearner::new().with_non_negative(false);
    let weights = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string(), "pred_b".to_string()],
        "label"
    ).unwrap();

    // Weights can be negative now, but should still sum to 1.0 (default normalization)
    let sum: f64 = weights.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

#[test]
fn test_weight_learner_with_l2_regularization() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred_a" => &[10.5, 19.5, 30.5],
        "pred_b" => &[9.5, 20.5, 29.5]
    }
    .unwrap();

    let learner = WeightLearner::new().with_l2_regularization(1.0);
    let weights = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string(), "pred_b".to_string()],
        "label"
    ).unwrap();

    assert_eq!(weights.len(), 2);
    let sum: f64 = weights.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

#[test]
fn test_weight_learner_empty_predictions() {
    let df = df! {
        "label" => &[10.0, 20.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let result = learner.learn_weights(&df, &vec![], "label");

    assert!(result.is_err());
    match result {
        Err(MetaError::EmptyPredictions) => {}
        _ => panic!("Expected EmptyPredictions error"),
    }
}

#[test]
fn test_weight_learner_missing_label() {
    let df = df! {
        "pred_a" => &[10.0, 20.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let result = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string()],
        "missing_label"
    );

    assert!(result.is_err());
    match result {
        Err(MetaError::MissingColumn(col)) => {
            assert_eq!(col, "missing_label");
        }
        _ => panic!("Expected MissingColumn error"),
    }
}

#[test]
fn test_weight_learner_missing_prediction() {
    let df = df! {
        "label" => &[10.0, 20.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let result = learner.learn_weights(
        &df,
        &vec!["missing_pred".to_string()],
        "label"
    );

    assert!(result.is_err());
    match result {
        Err(MetaError::MissingColumn(col)) => {
            assert_eq!(col, "missing_pred");
        }
        _ => panic!("Expected MissingColumn error"),
    }
}

#[test]
fn test_weight_learner_single_predictor() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred" => &[11.0, 19.0, 31.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let weights = learner.learn_weights(
        &df,
        &vec!["pred".to_string()],
        "label"
    ).unwrap();

    assert_eq!(weights.len(), 1);
    // Single predictor should get all the weight (normalized to 1.0)
    assert_relative_eq!(weights[0], 1.0, epsilon = 1e-10);
}

#[test]
fn test_weight_learner_default() {
    let learner1 = WeightLearner::new();
    let learner2 = WeightLearner::default();

    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred" => &[11.0, 19.0, 31.0]
    }
    .unwrap();

    let weights1 = learner1.learn_weights(&df, &vec!["pred".to_string()], "label").unwrap();
    let weights2 = learner2.learn_weights(&df, &vec!["pred".to_string()], "label").unwrap();

    assert_eq!(weights1.len(), weights2.len());
    for (w1, w2) in weights1.iter().zip(weights2.iter()) {
        assert_relative_eq!(w1, w2, epsilon = 1e-10);
    }
}

#[test]
fn test_weight_learner_all_zero_predictions() {
    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred_a" => &[0.0, 0.0, 0.0],
        "pred_b" => &[0.0, 0.0, 0.0]
    }
    .unwrap();

    let learner = WeightLearner::new();
    let weights = learner.learn_weights(
        &df,
        &vec!["pred_a".to_string(), "pred_b".to_string()],
        "label"
    ).unwrap();

    // When all weights would be zero/negative, should fall back to uniform
    let sum: f64 = weights.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

    // Should be uniform weights
    assert_relative_eq!(weights[0], 0.5, epsilon = 1e-6);
    assert_relative_eq!(weights[1], 0.5, epsilon = 1e-6);
}

#[test]
fn test_weight_learner_builder_pattern() {
    let learner = WeightLearner::new()
        .with_non_negative(false)
        .with_normalize(true)
        .with_l2_regularization(0.01);

    let df = df! {
        "label" => &[10.0, 20.0, 30.0],
        "pred" => &[11.0, 19.0, 31.0]
    }
    .unwrap();

    let weights = learner.learn_weights(&df, &vec!["pred".to_string()], "label").unwrap();
    assert_eq!(weights.len(), 1);
}

use polars::prelude::*;
use qliber::{evaluate_expression, Expression, OpsError};
use std::str::FromStr;

#[test]
fn test_simple_arithmetic() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0, 40.0]
    }
    .unwrap();

    // Test addition
    let result = evaluate_expression("price + 5", &df).unwrap();
    let expected = vec![15.0, 25.0, 35.0, 45.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);

    // Test subtraction
    let result = evaluate_expression("price - 5", &df).unwrap();
    let expected = vec![5.0, 15.0, 25.0, 35.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);

    // Test multiplication
    let result = evaluate_expression("price * 2", &df).unwrap();
    let expected = vec![20.0, 40.0, 60.0, 80.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);

    // Test division
    let result = evaluate_expression("price / 2", &df).unwrap();
    let expected = vec![5.0, 10.0, 15.0, 20.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_power_operation() {
    let df = df! {
        "base" => &[2.0, 3.0, 4.0]
    }
    .unwrap();

    let result = evaluate_expression("base ^ 2", &df).unwrap();
    let expected = vec![4.0, 9.0, 16.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_unary_negation() {
    let df = df! {
        "value" => &[10.0, -20.0, 30.0]
    }
    .unwrap();

    let result = evaluate_expression("-value", &df).unwrap();
    let expected = vec![-10.0, 20.0, -30.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_complex_expression() {
    let df = df! {
        "a" => &[10.0, 20.0, 30.0],
        "b" => &[2.0, 3.0, 4.0]
    }
    .unwrap();

    // Test (a + b) * 2
    let result = evaluate_expression("(a + b) * 2", &df).unwrap();
    let expected = vec![24.0, 46.0, 68.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);

    // Test a / b + 10
    let result = evaluate_expression("a / b + 10", &df).unwrap();
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert!((actual[0] - 15.0).abs() < 1e-10);
    assert!((actual[1] - 16.666666666666668).abs() < 1e-10);
    assert!((actual[2] - 17.5).abs() < 1e-10);
}

#[test]
fn test_rolling_mean() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    }
    .unwrap();

    let result = evaluate_expression("rolling_mean(price, 3)", &df).unwrap();
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();

    assert!((actual[0] - 10.0).abs() < 1e-10);
    assert!((actual[1] - 15.0).abs() < 1e-10);
    assert!((actual[2] - 20.0).abs() < 1e-10);
    assert!((actual[3] - 30.0).abs() < 1e-10);
    assert!((actual[4] - 40.0).abs() < 1e-10);
}

#[test]
fn test_rolling_std() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0, 4.0, 5.0]
    }
    .unwrap();

    let result = evaluate_expression("rolling_std(value, 3)", &df).unwrap();
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();

    // First value with window 1 should have std 0
    assert_eq!(actual[0], 0.0);
    // Verify non-zero std for later values
    assert!(actual[2] > 0.0);
    assert!(actual[4] > 0.0);
}

#[test]
fn test_expanding_sum() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0, 4.0, 5.0]
    }
    .unwrap();

    let result = evaluate_expression("expanding_sum(value)", &df).unwrap();
    let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_percentile() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0, 4.0, 5.0]
    }
    .unwrap();

    let result = evaluate_expression("percentile(value, 0.5)", &df).unwrap();
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();

    // All values should be the median (3.0)
    for val in actual {
        assert_eq!(val, 3.0);
    }
}

#[test]
fn test_lag() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    }
    .unwrap();

    let result = evaluate_expression("lag(price, 2)", &df).unwrap();
    let chunked = result.f64().unwrap();

    // First two values should be None
    assert!(chunked.get(0).is_none());
    assert!(chunked.get(1).is_none());

    // Remaining values should be lagged
    assert_eq!(chunked.get(2), Some(10.0));
    assert_eq!(chunked.get(3), Some(20.0));
    assert_eq!(chunked.get(4), Some(30.0));
}

#[test]
fn test_literal_values() {
    let df = df! {
        "price" => &[10.0, 20.0, 30.0]
    }
    .unwrap();

    let result = evaluate_expression("42.5", &df).unwrap();
    let expected = vec![42.5, 42.5, 42.5];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_scientific_notation() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0]
    }
    .unwrap();

    let result = evaluate_expression("value * 1e-3", &df).unwrap();
    let expected = vec![0.001, 0.002, 0.003];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-10);
    }
}

#[test]
fn test_expression_from_str() {
    let expr = Expression::from_str("price + 10").unwrap();
    let df = df! {
        "price" => &[5.0, 10.0, 15.0]
    }
    .unwrap();

    let result = expr.evaluate(&df).unwrap();
    let expected = vec![15.0, 20.0, 25.0];
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_unknown_column_error() {
    let df = df! {
        "price" => &[10.0, 20.0]
    }
    .unwrap();

    let result = evaluate_expression("unknown_column + 5", &df);
    assert!(result.is_err());
    match result {
        Err(OpsError::UnknownIdentifier(name)) => {
            assert_eq!(name, "unknown_column");
        }
        _ => panic!("Expected UnknownIdentifier error"),
    }
}

#[test]
fn test_unbalanced_parenthesis_error() {
    let result = Expression::from_str("(price + 5");
    assert!(result.is_err());
    match result {
        Err(OpsError::UnbalancedParenthesis) => {}
        _ => panic!("Expected UnbalancedParenthesis error"),
    }
}

#[test]
fn test_invalid_window_error() {
    let result = Expression::from_str("rolling_mean(price, 0)");
    assert!(result.is_err());
    match result {
        Err(OpsError::InvalidWindow) => {}
        _ => panic!("Expected InvalidWindow error"),
    }
}

#[test]
fn test_invalid_percentile_error() {
    let result1 = Expression::from_str("percentile(price, 1.5)");
    assert!(result1.is_err());
    match result1 {
        Err(OpsError::InvalidPercentile) => {}
        Err(e) => panic!("Test 1 - Expected InvalidPercentile error, got: {:?}", e),
        _ => panic!("Test 1 - Expected InvalidPercentile error"),
    }

    let result2 = Expression::from_str("percentile(price, -0.1)");
    assert!(result2.is_err());
    // Negative literals are parsed as unary negation, so this will return UnexpectedToken
    match result2 {
        Err(OpsError::UnexpectedToken(_)) => {}
        Err(e) => panic!("Test 2 - Expected UnexpectedToken error for negative literal, got: {:?}", e),
        _ => panic!("Test 2 - Expected error"),
    }
}

#[test]
fn test_unknown_function_error() {
    let result = Expression::from_str("unknown_func(price)");
    assert!(result.is_err());
    match result {
        Err(OpsError::UnknownIdentifier(name)) => {
            assert_eq!(name, "unknown_func");
        }
        _ => panic!("Expected UnknownIdentifier error"),
    }
}

#[test]
fn test_operator_precedence() {
    let df = df! {
        "a" => &[10.0],
        "b" => &[2.0],
        "c" => &[3.0]
    }
    .unwrap();

    // Test that multiplication has higher precedence than addition
    let result = evaluate_expression("a + b * c", &df).unwrap();
    let actual = result.f64().unwrap().get(0).unwrap();
    assert_eq!(actual, 16.0); // 10 + (2 * 3) = 16, not (10 + 2) * 3 = 36

    // Test that power has higher precedence than multiplication
    let result = evaluate_expression("b * c ^ 2", &df).unwrap();
    let actual = result.f64().unwrap().get(0).unwrap();
    assert_eq!(actual, 18.0); // 2 * (3 ^ 2) = 18, not (2 * 3) ^ 2 = 36
}

#[test]
fn test_nested_functions() {
    let df = df! {
        "value" => &[1.0, 2.0, 3.0, 4.0, 5.0]
    }
    .unwrap();

    // Test rolling mean of rolling mean
    let result = evaluate_expression("rolling_mean(rolling_mean(value, 2), 2)", &df).unwrap();
    assert!(result.len() == 5);

    // Verify values are finite
    let actual: Vec<f64> = result.f64().unwrap().into_no_null_iter().collect();
    for val in actual {
        assert!(val.is_finite());
    }
}

#[test]
fn test_expression_with_nulls() {
    let df = df! {
        "price" => &[Some(10.0), None, Some(30.0), Some(40.0)]
    }
    .unwrap();

    let result = evaluate_expression("price + 5", &df).unwrap();
    let chunked = result.f64().unwrap();

    assert_eq!(chunked.get(0), Some(15.0));
    assert!(chunked.get(1).is_none());
    assert_eq!(chunked.get(2), Some(35.0));
    assert_eq!(chunked.get(3), Some(45.0));
}

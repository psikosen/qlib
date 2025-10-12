use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{
    FactorRiskModel, PoetRiskModel, PoetThresholdMethod, ShrinkageMethod, StructuredFactorModel,
    StructuredRiskModel,
};

#[test]
fn factor_risk_model_sample_covariance() -> anyhow::Result<()> {
    let factors = df! {
        "f1" => &[0.01, 0.02, 0.015, 0.005],
        "f2" => &[0.03, 0.025, 0.02, 0.01],
    }?;

    let model = FactorRiskModel::from_factor_returns(
        &factors,
        vec!["f1".to_string(), "f2".to_string()],
        ShrinkageMethod::None,
    )?;
    let covariance = model.factor_covariance()?;

    let f1_values: Vec<f64> = covariance
        .column("f1")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let f2_values: Vec<f64> = covariance
        .column("f2")?
        .f64()?
        .into_no_null_iter()
        .collect();

    assert_relative_eq!(f1_values[0], 3.125e-5, epsilon = 1e-10);
    assert_relative_eq!(f1_values[1], 2.1875e-5, epsilon = 1e-10);
    assert_relative_eq!(f2_values[0], 2.1875e-5, epsilon = 1e-10);
    assert_relative_eq!(f2_values[1], 5.46875e-5, epsilon = 1e-10);

    let exposures = df! {
        "asset" => &["A", "B"],
        "f1" => &[0.5, 1.0],
        "f2" => &[1.0, 0.0],
    }?;

    let asset_cov = model.asset_covariance(&exposures, "asset")?;
    let a_cov: Vec<f64> = asset_cov.column("A")?.f64()?.into_no_null_iter().collect();
    let b_cov: Vec<f64> = asset_cov.column("B")?.f64()?.into_no_null_iter().collect();

    assert_relative_eq!(a_cov[0], 8.4375e-5, epsilon = 1e-10);
    assert_relative_eq!(a_cov[1], 3.75e-5, epsilon = 1e-10);
    assert_relative_eq!(b_cov[0], 3.75e-5, epsilon = 1e-10);
    assert_relative_eq!(b_cov[1], 3.125e-5, epsilon = 1e-10);

    let variance = model.portfolio_variance(
        &exposures,
        "asset",
        &[("A".to_string(), 0.6), ("B".to_string(), 0.4)],
    )?;
    assert_relative_eq!(variance, 5.3375e-5, epsilon = 1e-10);
    Ok(())
}

#[test]
fn factor_risk_model_shrinkage() -> anyhow::Result<()> {
    let factors = df! {
        "f1" => &[0.01, 0.02, 0.015, 0.005],
        "f2" => &[0.03, 0.025, 0.02, 0.01],
    }?;

    let model = FactorRiskModel::from_factor_returns(
        &factors,
        vec!["f1".to_string(), "f2".to_string()],
        ShrinkageMethod::Value(0.5),
    )?;
    assert_relative_eq!(model.shrinkage(), 0.5, epsilon = 1e-12);
    Ok(())
}

#[test]
fn poet_risk_model_matches_python() -> anyhow::Result<()> {
    let factors = df! {
        "f1" => &[0.01, 0.02, 0.015, 0.005, 0.018],
        "f2" => &[0.03, 0.025, 0.02, 0.01, 0.028],
        "f3" => &[0.015, 0.018, 0.017, 0.016, 0.019],
    }?;

    let model = PoetRiskModel::from_factor_returns(
        &factors,
        vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        1,
        1.0,
        PoetThresholdMethod::Soft,
    )?;
    let covariance = model.factor_covariance()?;
    let expected = vec![
        vec![2.984e-05, 3.0306826e-05, 2.935572e-06],
        vec![3.0306826e-05, 5.104e-05, 4.487231e-06],
        vec![2.935572e-06, 4.487231e-06, 2.0e-06],
    ];
    let names = ["f1", "f2", "f3"];
    for (col_idx, name) in names.iter().enumerate() {
        let column = covariance.column(name)?.f64()?;
        for (row_idx, value) in column.into_no_null_iter().enumerate() {
            assert_relative_eq!(value, expected[row_idx][col_idx], epsilon = 1e-10);
        }
    }
    Ok(())
}

#[test]
fn structured_risk_model_pca_parity() -> anyhow::Result<()> {
    let factors = df! {
        "f1" => &[0.01, 0.02, 0.015, 0.005, 0.018],
        "f2" => &[0.03, 0.025, 0.02, 0.01, 0.028],
        "f3" => &[0.015, 0.018, 0.017, 0.016, 0.019],
    }?;

    let model = StructuredRiskModel::from_observations(
        &factors,
        vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        StructuredFactorModel::Pca,
        2,
    )?;

    let covariance = model.factor_covariance()?;
    let expected = vec![
        vec![3.7291797e-05, 2.9312628e-05, 7.896651e-06],
        vec![2.9312628e-05, 6.3799222e-05, 2.704849e-06],
        vec![7.896651e-06, 2.704849e-06, 2.395131e-06],
    ];
    let names = ["f1", "f2", "f3"];
    for (col_idx, name) in names.iter().enumerate() {
        let column = covariance.column(name)?.f64()?;
        for (row_idx, value) in column.into_no_null_iter().enumerate() {
            assert_relative_eq!(value, expected[row_idx][col_idx], epsilon = 1e-10);
        }
    }

    let (exposures, factor_cov, idiosyncratic) = model.decomposed_components()?;
    let factor1: Vec<f64> = exposures
        .column("factor_1")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let factor2: Vec<f64> = exposures
        .column("factor_2")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let expected_factor1 = vec![0.54567049, 0.83409614, 0.08079201];
    let expected_factor2 = vec![0.7938458, -0.54539323, 0.26898896];
    for (value, expected) in factor1.into_iter().zip(expected_factor1) {
        assert_relative_eq!(value, expected, epsilon = 1e-8);
    }
    for (value, expected) in factor2.into_iter().zip(expected_factor2) {
        assert_relative_eq!(value, expected, epsilon = 1e-8);
    }

    let factor_diag: Vec<f64> = factor_cov
        .column("factor_1")?
        .f64()?
        .into_no_null_iter()
        .collect();
    assert_relative_eq!(factor_diag[0], 8.3234598e-05, epsilon = 1e-10);
    let factor_cov_col2: Vec<f64> = factor_cov
        .column("factor_2")?
        .f64()?
        .into_no_null_iter()
        .collect();
    assert_relative_eq!(factor_cov_col2[1], 1.9796151e-05, epsilon = 1e-10);

    let variances: Vec<f64> = idiosyncratic
        .column("variance")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let expected_var = vec![3.2813e-08, 3.11e-09, 4.19478e-07];
    for (value, expected) in variances.into_iter().zip(expected_var) {
        assert_relative_eq!(value, expected, epsilon = 1e-12);
    }

    Ok(())
}

#[test]
fn structured_risk_model_factor_analysis_parity() -> anyhow::Result<()> {
    let factors = df! {
        "f1" => &[0.01, 0.02, 0.015, 0.005, 0.018],
        "f2" => &[0.03, 0.025, 0.02, 0.01, 0.028],
        "f3" => &[0.015, 0.018, 0.017, 0.016, 0.019],
    }?;

    let model = StructuredRiskModel::from_observations(
        &factors,
        vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        StructuredFactorModel::FactorAnalysis,
        2,
    )?;

    let covariance = model.factor_covariance()?;
    let expected = vec![
        vec![3.4013501e-05, 2.7428158e-05, 7.444658e-06],
        vec![2.7428158e-05, 4.7816253e-05, 3.249591e-06],
        vec![7.444658e-06, 3.249591e-06, 2.15754e-06],
    ];
    let names = ["f1", "f2", "f3"];
    for (col_idx, name) in names.iter().enumerate() {
        let column = covariance.column(name)?.f64()?;
        for (row_idx, value) in column.into_no_null_iter().enumerate() {
            assert_relative_eq!(value, expected[row_idx][col_idx], epsilon = 1e-10);
        }
    }
    Ok(())
}

use approx::assert_relative_eq;
use polars::prelude::*;
use qliber::{FactorRiskModel, ShrinkageMethod};

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

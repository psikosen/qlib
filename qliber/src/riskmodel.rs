use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum RiskModelError {
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("missing column '{0}' in risk model input")]
    MissingColumn(String),
    #[error("column '{0}' contains null values")]
    NullValue(String),
    #[error("risk model requires at least one factor column")]
    EmptyFactors,
    #[error("risk model requires at least one observation")]
    EmptyObservations,
    #[error("exposure matrix column count mismatch")]
    FactorMismatch,
}

pub type RiskModelResult<T> = Result<T, RiskModelError>;

#[derive(Debug, Clone, Copy)]
pub enum ShrinkageMethod {
    None,
    LedoitWolf,
    Value(f64),
}

#[derive(Debug, Clone)]
pub struct FactorRiskModel {
    factor_names: Vec<String>,
    covariance: DMatrix<f64>,
    shrinkage: f64,
}

impl FactorRiskModel {
    pub fn from_factor_returns(
        frame: &DataFrame,
        factor_columns: impl Into<Vec<String>>,
        shrinkage: ShrinkageMethod,
    ) -> RiskModelResult<Self> {
        let factor_columns = factor_columns.into();
        if factor_columns.is_empty() {
            return Err(RiskModelError::EmptyFactors);
        }

        log_event(
            file!(),
            "FactorRiskModel",
            "from_factor_returns",
            "riskmodel.covariance",
            line!(),
            "Estimating factor covariance matrix",
            None,
            "pre",
            "POST",
        );

        let matrix = frame_to_matrix(frame, &factor_columns)?;
        if matrix.nrows() == 0 {
            return Err(RiskModelError::EmptyObservations);
        }

        let (covariance, shrinkage_value) = compute_covariance(&matrix, shrinkage);
        log_event(
            file!(),
            "FactorRiskModel",
            "from_factor_returns",
            "riskmodel.covariance",
            line!(),
            &format!(
                "Estimated {}x{} factor covariance with shrinkage {:.4}",
                covariance.nrows(),
                covariance.ncols(),
                shrinkage_value
            ),
            None,
            "post",
            "POST",
        );

        Ok(Self {
            factor_names: factor_columns,
            covariance,
            shrinkage: shrinkage_value,
        })
    }

    pub fn factor_covariance(&self) -> RiskModelResult<DataFrame> {
        let mut columns = Vec::with_capacity(self.factor_names.len() + 1);
        let index_series = Series::new("factor", self.factor_names.clone());
        columns.push(index_series);

        for (col_idx, name) in self.factor_names.iter().enumerate() {
            let mut values = Vec::with_capacity(self.factor_names.len());
            for row_idx in 0..self.factor_names.len() {
                values.push(self.covariance[(row_idx, col_idx)]);
            }
            columns.push(Series::new(name.as_str(), values));
        }

        DataFrame::new(columns).map_err(RiskModelError::from)
    }

    pub fn shrinkage(&self) -> f64 {
        self.shrinkage
    }

    pub fn asset_covariance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
    ) -> RiskModelResult<DataFrame> {
        log_event(
            file!(),
            "FactorRiskModel",
            "asset_covariance",
            "riskmodel.assets",
            line!(),
            "Projecting factor covariance into asset space",
            None,
            "pre",
            "POST",
        );

        let asset_names = asset_names(exposures, asset_column)?;
        let exposure_matrix = frame_to_matrix(exposures, &self.factor_names)?;
        if exposure_matrix.ncols() != self.factor_names.len() {
            return Err(RiskModelError::FactorMismatch);
        }

        let covariance = &exposure_matrix * &self.covariance * exposure_matrix.transpose();
        let mut columns = Vec::with_capacity(asset_names.len() + 1);
        columns.push(Series::new(asset_column, asset_names.clone()));
        for (col_idx, name) in asset_names.iter().enumerate() {
            let mut values = Vec::with_capacity(asset_names.len());
            for row_idx in 0..asset_names.len() {
                values.push(covariance[(row_idx, col_idx)]);
            }
            columns.push(Series::new(name.as_str(), values));
        }

        let frame = DataFrame::new(columns).map_err(RiskModelError::from)?;
        log_event(
            file!(),
            "FactorRiskModel",
            "asset_covariance",
            "riskmodel.assets",
            line!(),
            &format!("Computed asset covariance for {} assets", asset_names.len()),
            None,
            "post",
            "POST",
        );
        Ok(frame)
    }

    pub fn portfolio_variance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
        weights: &[(String, f64)],
    ) -> RiskModelResult<f64> {
        let asset_names = asset_names(exposures, asset_column)?;
        let exposure_matrix = frame_to_matrix(exposures, &self.factor_names)?;
        if exposure_matrix.ncols() != self.factor_names.len() {
            return Err(RiskModelError::FactorMismatch);
        }

        let covariance = &exposure_matrix * &self.covariance * exposure_matrix.transpose();
        let weight_map: HashMap<&str, f64> = weights
            .iter()
            .map(|(name, weight)| (name.as_str(), *weight))
            .collect();
        let mut weight_values = Vec::with_capacity(asset_names.len());
        for asset in &asset_names {
            weight_values.push(*weight_map.get(asset.as_str()).unwrap_or(&0.0));
        }

        let vector = DVector::from_column_slice(&weight_values);
        let variance = vector.transpose() * covariance * vector;
        Ok(variance[(0, 0)])
    }
}

fn asset_names(frame: &DataFrame, column: &str) -> RiskModelResult<Vec<String>> {
    let series = frame
        .column(column)
        .map_err(|_| RiskModelError::MissingColumn(column.to_string()))?;
    let utf8 = series
        .utf8()
        .map_err(|_| RiskModelError::MissingColumn(column.to_string()))?;
    if utf8.null_count() > 0 {
        return Err(RiskModelError::NullValue(column.to_string()));
    }
    Ok(utf8
        .into_no_null_iter()
        .map(|value| value.to_string())
        .collect())
}

fn frame_to_matrix(frame: &DataFrame, columns: &[String]) -> RiskModelResult<DMatrix<f64>> {
    if columns.is_empty() {
        return Err(RiskModelError::EmptyFactors);
    }
    let row_count = frame.height();
    if row_count == 0 {
        return Err(RiskModelError::EmptyObservations);
    }

    let mut data_columns = Vec::with_capacity(columns.len());
    for column in columns {
        let series = frame
            .column(column)
            .map_err(|_| RiskModelError::MissingColumn(column.clone()))?;
        let chunked = series
            .f64()
            .map_err(|_| RiskModelError::MissingColumn(column.clone()))?;
        if chunked.null_count() > 0 {
            return Err(RiskModelError::NullValue(column.clone()));
        }
        let mut values = Vec::with_capacity(row_count);
        for value in chunked.into_no_null_iter() {
            values.push(value);
        }
        data_columns.push(values);
    }

    let mut matrix_data = Vec::with_capacity(row_count * columns.len());
    for row_idx in 0..row_count {
        for column in &data_columns {
            matrix_data.push(column[row_idx]);
        }
    }

    Ok(DMatrix::from_row_slice(
        row_count,
        columns.len(),
        &matrix_data,
    ))
}

fn compute_covariance(data: &DMatrix<f64>, shrinkage: ShrinkageMethod) -> (DMatrix<f64>, f64) {
    let centered = center_columns(data);
    let empirical = sample_covariance(&centered);
    match shrinkage {
        ShrinkageMethod::None => (empirical, 0.0),
        ShrinkageMethod::Value(alpha) => {
            let alpha = alpha.clamp(0.0, 1.0);
            (apply_shrinkage(&empirical, alpha), alpha)
        }
        ShrinkageMethod::LedoitWolf => ledoit_wolf(&centered, &empirical),
    }
}

fn center_columns(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mut centered = data.clone_owned();
    if data.nrows() == 0 {
        return centered;
    }
    let mut means = vec![0.0; data.ncols()];
    for row_idx in 0..data.nrows() {
        for col_idx in 0..data.ncols() {
            means[col_idx] += data[(row_idx, col_idx)];
        }
    }
    for mean in &mut means {
        *mean /= data.nrows() as f64;
    }

    for row_idx in 0..centered.nrows() {
        for col_idx in 0..centered.ncols() {
            centered[(row_idx, col_idx)] -= means[col_idx];
        }
    }
    centered
}

fn sample_covariance(centered: &DMatrix<f64>) -> DMatrix<f64> {
    if centered.nrows() == 0 {
        return DMatrix::zeros(centered.ncols(), centered.ncols());
    }
    (&centered.transpose() * centered) / centered.nrows() as f64
}

fn apply_shrinkage(empirical: &DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    let mut shrunk = empirical * (1.0 - alpha);
    let mu = empirical.trace() / empirical.nrows() as f64;
    for idx in 0..empirical.nrows() {
        shrunk[(idx, idx)] += alpha * mu;
    }
    shrunk
}

fn ledoit_wolf(centered: &DMatrix<f64>, empirical: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    let n_samples = centered.nrows();
    let n_features = centered.ncols();
    if n_features == 0 {
        return (empirical.clone_owned(), 0.0);
    }

    let mu = empirical.trace() / n_features as f64;
    let mut delta_matrix = empirical.clone_owned();
    for idx in 0..n_features {
        delta_matrix[(idx, idx)] -= mu;
    }
    let delta = delta_matrix.iter().map(|value| value * value).sum::<f64>() / n_features as f64;

    let squared = centered.map(|value| value * value);
    let term = (&squared.transpose() * &squared) / n_samples as f64;
    let empirical_squared = empirical.component_mul(empirical);
    let beta_hat =
        (term - empirical_squared).iter().sum::<f64>() / (n_features as f64 * n_samples as f64);
    let beta = beta_hat.max(0.0).min(delta);
    let shrinkage = if delta > 0.0 { beta / delta } else { 0.0 };
    let shrunk = apply_shrinkage(empirical, shrinkage);
    (shrunk, shrinkage)
}

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, SVD, SymmetricEigen};
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
    #[error("covariance dimension mismatch")]
    DimensionMismatch,
    #[error("requested {requested} factors but only {available} are available")]
    InvalidFactorCount { requested: usize, available: usize },
    #[error("matrix inversion failed during risk model estimation")]
    SingularMatrix,
    #[error("{0}")]
    Computation(String),
}

pub type RiskModelResult<T> = Result<T, RiskModelError>;

#[derive(Debug, Clone)]
struct CovarianceSurface {
    factor_names: Vec<String>,
    covariance: DMatrix<f64>,
}

type StructuredComponents = (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, Vec<f64>);

impl CovarianceSurface {
    fn new(factor_names: Vec<String>, covariance: DMatrix<f64>) -> RiskModelResult<Self> {
        if covariance.nrows() != covariance.ncols() {
            return Err(RiskModelError::DimensionMismatch);
        }
        if covariance.nrows() != factor_names.len() {
            return Err(RiskModelError::DimensionMismatch);
        }
        Ok(Self {
            factor_names,
            covariance,
        })
    }

    fn names(&self) -> &[String] {
        &self.factor_names
    }

    fn factor_covariance(&self, label: &str) -> RiskModelResult<DataFrame> {
        let mut columns = Vec::with_capacity(self.factor_names.len() + 1);
        columns.push(Series::new(label, self.factor_names.clone()));
        for (col_idx, name) in self.factor_names.iter().enumerate() {
            let mut values = Vec::with_capacity(self.factor_names.len());
            for row_idx in 0..self.factor_names.len() {
                values.push(self.covariance[(row_idx, col_idx)]);
            }
            columns.push(Series::new(name.as_str(), values));
        }
        DataFrame::new(columns).map_err(RiskModelError::from)
    }

    fn asset_covariance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
    ) -> RiskModelResult<DataFrame> {
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
        DataFrame::new(columns).map_err(RiskModelError::from)
    }

    fn portfolio_variance(
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

#[derive(Debug, Clone, Copy)]
pub enum ShrinkageMethod {
    None,
    LedoitWolf,
    Value(f64),
}

#[derive(Debug, Clone)]
pub struct FactorRiskModel {
    surface: CovarianceSurface,
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
        let surface = CovarianceSurface::new(factor_columns.clone(), covariance)?;
        log_event(
            file!(),
            "FactorRiskModel",
            "from_factor_returns",
            "riskmodel.covariance",
            line!(),
            &format!(
                "Estimated {}x{} factor covariance with shrinkage {:.4}",
                surface.covariance.nrows(),
                surface.covariance.ncols(),
                shrinkage_value
            ),
            None,
            "post",
            "POST",
        );

        Ok(Self {
            surface,
            shrinkage: shrinkage_value,
        })
    }

    pub fn factor_covariance(&self) -> RiskModelResult<DataFrame> {
        self.surface.factor_covariance("factor")
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
        let frame = self.surface.asset_covariance(exposures, asset_column)?;
        log_event(
            file!(),
            "FactorRiskModel",
            "asset_covariance",
            "riskmodel.assets",
            line!(),
            &format!("Computed asset covariance for {} assets", frame.height()),
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
        self.surface
            .portfolio_variance(exposures, asset_column, weights)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PoetThresholdMethod {
    Hard,
    Soft,
    Scad,
}

#[derive(Debug, Clone)]
pub struct PoetRiskModel {
    surface: CovarianceSurface,
    num_factors: usize,
    threshold: f64,
    method: PoetThresholdMethod,
}

impl PoetRiskModel {
    pub fn from_factor_returns(
        frame: &DataFrame,
        factor_columns: impl Into<Vec<String>>,
        num_factors: usize,
        threshold: f64,
        method: PoetThresholdMethod,
    ) -> RiskModelResult<Self> {
        let factor_columns = factor_columns.into();
        if factor_columns.is_empty() {
            return Err(RiskModelError::EmptyFactors);
        }
        if threshold < 0.0 {
            return Err(RiskModelError::Computation(
                "POET threshold must be non-negative".into(),
            ));
        }

        log_event(
            file!(),
            "PoetRiskModel",
            "from_factor_returns",
            "riskmodel.poet",
            line!(),
            "Estimating POET covariance",
            None,
            "pre",
            "POST",
        );

        let matrix = frame_to_matrix(frame, &factor_columns)?;
        if matrix.nrows() == 0 {
            return Err(RiskModelError::EmptyObservations);
        }
        let covariance = poet_covariance(&matrix, num_factors, threshold, method)?;
        let surface = CovarianceSurface::new(factor_columns, covariance)?;

        log_event(
            file!(),
            "PoetRiskModel",
            "from_factor_returns",
            "riskmodel.poet",
            line!(),
            &format!(
                "Computed POET covariance with {} factors and threshold {:.4}",
                num_factors, threshold
            ),
            None,
            "post",
            "POST",
        );

        Ok(Self {
            surface,
            num_factors,
            threshold,
            method,
        })
    }

    pub fn factor_covariance(&self) -> RiskModelResult<DataFrame> {
        self.surface.factor_covariance("factor")
    }

    pub fn asset_covariance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
    ) -> RiskModelResult<DataFrame> {
        self.surface.asset_covariance(exposures, asset_column)
    }

    pub fn portfolio_variance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
        weights: &[(String, f64)],
    ) -> RiskModelResult<f64> {
        self.surface
            .portfolio_variance(exposures, asset_column, weights)
    }

    pub fn num_factors(&self) -> usize {
        self.num_factors
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn method(&self) -> PoetThresholdMethod {
        self.method
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StructuredFactorModel {
    Pca,
    FactorAnalysis,
}

#[derive(Debug, Clone)]
pub struct StructuredRiskModel {
    surface: CovarianceSurface,
    exposures: DMatrix<f64>,
    factor_covariance: DMatrix<f64>,
    specific_variance: Vec<f64>,
    method: StructuredFactorModel,
}

impl StructuredRiskModel {
    pub fn from_observations(
        frame: &DataFrame,
        columns: impl Into<Vec<String>>,
        method: StructuredFactorModel,
        num_factors: usize,
    ) -> RiskModelResult<Self> {
        let columns = columns.into();
        if columns.is_empty() {
            return Err(RiskModelError::EmptyFactors);
        }
        if num_factors == 0 {
            return Err(RiskModelError::Computation(
                "structured covariance requires at least one factor".into(),
            ));
        }

        log_event(
            file!(),
            "StructuredRiskModel",
            "from_observations",
            "riskmodel.structured",
            line!(),
            &format!(
                "Estimating structured covariance via {:?} with {} factors",
                method, num_factors
            ),
            None,
            "pre",
            "POST",
        );

        let matrix = frame_to_matrix(frame, &columns)?;
        if matrix.nrows() == 0 {
            return Err(RiskModelError::EmptyObservations);
        }

        let (covariance, exposures, factor_covariance, specific_variance) = match method {
            StructuredFactorModel::Pca => structured_covariance_pca(&matrix, num_factors)?,
            StructuredFactorModel::FactorAnalysis => {
                structured_covariance_factor_analysis(&matrix, num_factors)?
            }
        };
        let surface = CovarianceSurface::new(columns.clone(), covariance)?;

        log_event(
            file!(),
            "StructuredRiskModel",
            "from_observations",
            "riskmodel.structured",
            line!(),
            &format!(
                "Structured covariance estimated for {} variables",
                columns.len()
            ),
            None,
            "post",
            "POST",
        );

        Ok(Self {
            surface,
            exposures,
            factor_covariance,
            specific_variance,
            method,
        })
    }

    pub fn factor_covariance(&self) -> RiskModelResult<DataFrame> {
        self.surface.factor_covariance("variable")
    }

    pub fn asset_covariance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
    ) -> RiskModelResult<DataFrame> {
        self.surface.asset_covariance(exposures, asset_column)
    }

    pub fn portfolio_variance(
        &self,
        exposures: &DataFrame,
        asset_column: &str,
        weights: &[(String, f64)],
    ) -> RiskModelResult<f64> {
        self.surface
            .portfolio_variance(exposures, asset_column, weights)
    }

    pub fn method(&self) -> StructuredFactorModel {
        self.method
    }

    pub fn decomposed_components(&self) -> RiskModelResult<(DataFrame, DataFrame, DataFrame)> {
        let feature_names = self.surface.names().to_vec();

        let exposures = exposures_to_dataframe(&self.exposures, &feature_names)?;
        let factor_cov = matrix_to_dataframe(
            &self.factor_covariance,
            &(0..self.factor_covariance.nrows())
                .map(|idx| format!("factor_{}", idx + 1))
                .collect::<Vec<_>>(),
            "factor",
        )?;
        let idio = specific_variance_to_dataframe(&self.specific_variance, &feature_names)?;
        Ok((exposures, factor_cov, idio))
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

fn poet_covariance(
    data: &DMatrix<f64>,
    num_factors: usize,
    threshold: f64,
    method: PoetThresholdMethod,
) -> RiskModelResult<DMatrix<f64>> {
    let n = data.nrows();
    let p = data.ncols();
    if p == 0 || n == 0 {
        return Err(RiskModelError::EmptyObservations);
    }
    if num_factors > n.min(p) {
        return Err(RiskModelError::InvalidFactorCount {
            requested: num_factors,
            available: n.min(p),
        });
    }

    let centered = center_columns(data);
    let y = centered.transpose();
    let (lowrank, uhat) = if num_factors > 0 {
        let eigen = SymmetricEigen::new((&y.transpose() * &y).clone_owned());
        let mut indices: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
        indices.sort_by(|a, b| {
            eigen.eigenvalues[*a]
                .partial_cmp(&eigen.eigenvalues[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected: Vec<usize> = indices.into_iter().rev().take(num_factors).collect();
        let mut f_matrix = DMatrix::zeros(n, num_factors);
        let scale = (n as f64).sqrt();
        for (col_idx, eig_idx) in selected.iter().enumerate() {
            for row in 0..n {
                f_matrix[(row, col_idx)] = eigen.eigenvectors[(row, *eig_idx)] * scale;
            }
        }
        let lam = &y * &f_matrix / n as f64;
        let uhat = &y - &lam * f_matrix.transpose();
        let lowrank = &lam * lam.transpose();
        (lowrank, uhat)
    } else {
        (DMatrix::zeros(p, p), y.clone_owned())
    };

    let rate = if num_factors > 0 {
        1.0 / (p as f64).sqrt() + ((p as f64).ln() / n as f64).sqrt()
    } else {
        ((p as f64).ln() / n as f64).sqrt()
    };
    let lambda = rate * threshold;

    let supca = &uhat * uhat.transpose() / n as f64;
    let mut diag = Vec::with_capacity(p);
    for idx in 0..p {
        diag.push(ensure_positive(supca[(idx, idx)]));
    }
    let sqrt_diag: Vec<f64> = diag.iter().map(|value| value.sqrt()).collect();
    let inv_sqrt: Vec<f64> = sqrt_diag
        .iter()
        .map(|value| if *value > 0.0 { 1.0 / *value } else { 0.0 })
        .collect();

    let mut r = supca.clone_owned();
    for i in 0..p {
        for j in 0..p {
            r[(i, j)] *= inv_sqrt[i] * inv_sqrt[j];
        }
    }

    let mut thresholded = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            if i == j {
                thresholded[(i, j)] = 1.0;
            } else {
                thresholded[(i, j)] = apply_poet_threshold(r[(i, j)], lambda, method);
            }
        }
    }

    let mut sigma_u = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            sigma_u[(i, j)] = sqrt_diag[i] * thresholded[(i, j)] * sqrt_diag[j];
        }
    }

    let mut covariance = sigma_u + lowrank;
    symmetrize(&mut covariance);
    Ok(covariance)
}

fn structured_covariance_pca(
    data: &DMatrix<f64>,
    num_factors: usize,
) -> RiskModelResult<StructuredComponents> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    if num_factors > n_features.min(n_samples) {
        return Err(RiskModelError::InvalidFactorCount {
            requested: num_factors,
            available: n_features.min(n_samples),
        });
    }

    let centered = center_columns(data);
    let cov = unbiased_covariance(&centered);
    let eigen = SymmetricEigen::new(cov.clone_owned());
    let mut indices: Vec<usize> = (0..n_features).collect();
    indices.sort_by(|a, b| {
        eigen.eigenvalues[*a]
            .partial_cmp(&eigen.eigenvalues[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut exposures = DMatrix::zeros(n_features, num_factors);
    let mut factor_cov = DMatrix::zeros(num_factors, num_factors);
    for (col_idx, eig_idx) in indices.into_iter().rev().take(num_factors).enumerate() {
        let eigenvalue = eigen.eigenvalues[eig_idx].max(0.0);
        factor_cov[(col_idx, col_idx)] = eigenvalue;
        for row in 0..n_features {
            exposures[(row, col_idx)] = eigen.eigenvectors[(row, eig_idx)];
        }
    }

    orient_columns(&mut exposures);

    let scores = centered.clone_owned() * &exposures;
    let residual = centered - &scores * exposures.transpose();
    let specific_variance = column_variances(&residual);
    let diag = diag_matrix(&specific_variance);
    let mut covariance = &exposures * &factor_cov * exposures.transpose() + diag;
    symmetrize(&mut covariance);
    Ok((covariance, exposures, factor_cov, specific_variance))
}

fn structured_covariance_factor_analysis(
    data: &DMatrix<f64>,
    num_factors: usize,
) -> RiskModelResult<StructuredComponents> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    if num_factors > n_features.min(n_samples) {
        return Err(RiskModelError::InvalidFactorCount {
            requested: num_factors,
            available: n_features.min(n_samples),
        });
    }

    let centered = center_columns(data);
    let mut psi: Vec<f64> = vec![1.0; n_features];
    let var = column_variances(&centered);
    let nsqrt = (n_samples as f64).sqrt();
    let llconst = n_features as f64 * (2.0 * std::f64::consts::PI).ln() + num_factors as f64;
    let small = 1e-12;
    let max_iter = 1000;
    let tol = 1e-2;
    let mut loadings = DMatrix::zeros(num_factors, n_features);
    let mut old_ll = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        let mut sqrt_psi = vec![0.0; n_features];
        for (idx, value) in psi.iter().enumerate() {
            sqrt_psi[idx] = (*value).sqrt() + small;
        }

        let mut scaled = centered.clone_owned();
        for col in 0..n_features {
            let scale = sqrt_psi[col] * nsqrt;
            for row in 0..n_samples {
                scaled[(row, col)] /= scale;
            }
        }

        let svd = SVD::new(scaled, false, true);
        let singular_values = svd.singular_values;
        let v_t_full = svd
            .v_t
            .ok_or_else(|| RiskModelError::Computation("SVD failed to compute V^T".into()))?;

        let mut selected = Vec::with_capacity(num_factors);
        let mut unexplained = 0.0;
        for (idx, value) in singular_values.iter().enumerate() {
            if idx < num_factors {
                selected.push(*value);
            } else {
                unexplained += value * value;
            }
        }

        let mut w = v_t_full.rows(0, num_factors).into_owned();
        for row in 0..num_factors {
            let squared = selected[row] * selected[row];
            let scale = (squared - 1.0).max(0.0).sqrt();
            for col in 0..n_features {
                w[(row, col)] *= scale * sqrt_psi[col];
            }
        }

        loadings = w.clone();

        let mut ll = llconst;
        for value in &selected {
            let squared = (*value) * (*value);
            if squared > 0.0 {
                ll += squared.ln();
            }
        }
        ll += unexplained;
        for value in &psi {
            ll += (*value).ln();
        }
        ll *= -(n_samples as f64) / 2.0;

        if ll - old_ll < tol {
            break;
        }
        old_ll = ll;

        let mut next_psi = Vec::with_capacity(n_features);
        for feature in 0..n_features {
            let mut sum_sq = 0.0;
            for factor in 0..num_factors {
                sum_sq += loadings[(factor, feature)] * loadings[(factor, feature)];
            }
            next_psi.push((var[feature] - sum_sq).max(small));
        }
        psi = next_psi;
    }

    orient_rows(&mut loadings);

    let mut wpsi = loadings.clone();
    for factor in 0..num_factors {
        for feature in 0..n_features {
            wpsi[(factor, feature)] /= ensure_positive(psi[feature]);
        }
    }

    let identity = DMatrix::<f64>::identity(num_factors, num_factors);
    let covariance_z = (identity + &wpsi * loadings.transpose())
        .try_inverse()
        .ok_or(RiskModelError::SingularMatrix)?;

    let scores = centered.clone_owned() * wpsi.transpose() * covariance_z;
    let centered_scores = center_columns(&scores);
    let factor_covariance = unbiased_covariance(&centered_scores);
    let residual = centered.clone_owned() - &scores * &loadings;
    let specific_variance = column_variances(&residual);
    let exposures = loadings.transpose().into_owned();
    let diag = diag_matrix(&specific_variance);
    let mut covariance = &exposures * &factor_covariance * exposures.transpose() + diag;
    symmetrize(&mut covariance);
    Ok((covariance, exposures, factor_covariance, specific_variance))
}

fn apply_poet_threshold(value: f64, lambda: f64, method: PoetThresholdMethod) -> f64 {
    if lambda <= 0.0 {
        return value;
    }
    let abs_val = value.abs();
    match method {
        PoetThresholdMethod::Hard => {
            if abs_val > lambda {
                value
            } else {
                0.0
            }
        }
        PoetThresholdMethod::Soft => {
            let res = (abs_val - lambda).max(0.0);
            value.signum() * res
        }
        PoetThresholdMethod::Scad => {
            if abs_val <= lambda {
                0.0
            } else if abs_val <= 2.0 * lambda {
                value.signum() * (abs_val - lambda)
            } else if abs_val <= 3.7 * lambda {
                (2.7 * value - 3.7 * value.signum() * lambda) / 1.7
            } else {
                value
            }
        }
    }
}

fn symmetrize(matrix: &mut DMatrix<f64>) {
    for i in 0..matrix.nrows() {
        for j in (i + 1)..matrix.ncols() {
            let value = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }
}

fn orient_columns(matrix: &mut DMatrix<f64>) {
    for col in 0..matrix.ncols() {
        let mut max_idx = 0;
        let mut max_value = 0.0;
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)].abs();
            if value > max_value {
                max_value = value;
                max_idx = row;
            }
        }
        if matrix[(max_idx, col)] < 0.0 {
            for row in 0..matrix.nrows() {
                matrix[(row, col)] = -matrix[(row, col)];
            }
        }
    }
}

fn orient_rows(matrix: &mut DMatrix<f64>) {
    for row in 0..matrix.nrows() {
        let mut max_idx = 0;
        let mut max_value = 0.0;
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)].abs();
            if value > max_value {
                max_value = value;
                max_idx = col;
            }
        }
        if matrix[(row, max_idx)] < 0.0 {
            for col in 0..matrix.ncols() {
                matrix[(row, col)] = -matrix[(row, col)];
            }
        }
    }
}

fn ensure_positive(value: f64) -> f64 {
    if value.is_finite() {
        value.max(1e-12)
    } else {
        1e-12
    }
}

fn diag_matrix(values: &[f64]) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(values.len(), values.len());
    for (idx, value) in values.iter().enumerate() {
        matrix[(idx, idx)] = *value;
    }
    matrix
}

fn column_variances(matrix: &DMatrix<f64>) -> Vec<f64> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut variances = vec![0.0; cols];
    if rows == 0 {
        return variances;
    }
    for col in 0..cols {
        let mut sum = 0.0;
        for row in 0..rows {
            let value = matrix[(row, col)];
            sum += value * value;
        }
        variances[col] = sum / rows as f64;
    }
    variances
}

fn exposures_to_dataframe(
    exposures: &DMatrix<f64>,
    feature_names: &[String],
) -> RiskModelResult<DataFrame> {
    let mut columns = Vec::with_capacity(exposures.ncols() + 1);
    columns.push(Series::new("feature", feature_names.to_vec()));
    for factor_idx in 0..exposures.ncols() {
        let mut values = Vec::with_capacity(exposures.nrows());
        for feature_idx in 0..exposures.nrows() {
            values.push(exposures[(feature_idx, factor_idx)]);
        }
        let column_name = format!("factor_{}", factor_idx + 1);
        columns.push(Series::new(column_name.as_str(), values));
    }
    DataFrame::new(columns).map_err(RiskModelError::from)
}

fn matrix_to_dataframe(
    matrix: &DMatrix<f64>,
    names: &[String],
    index_label: &str,
) -> RiskModelResult<DataFrame> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != names.len() {
        return Err(RiskModelError::DimensionMismatch);
    }
    let mut columns = Vec::with_capacity(names.len() + 1);
    columns.push(Series::new(index_label, names.to_vec()));
    for (col_idx, name) in names.iter().enumerate() {
        let mut values = Vec::with_capacity(names.len());
        for row_idx in 0..names.len() {
            values.push(matrix[(row_idx, col_idx)]);
        }
        columns.push(Series::new(name.as_str(), values));
    }
    DataFrame::new(columns).map_err(RiskModelError::from)
}

fn specific_variance_to_dataframe(
    values: &[f64],
    feature_names: &[String],
) -> RiskModelResult<DataFrame> {
    if values.len() != feature_names.len() {
        return Err(RiskModelError::DimensionMismatch);
    }
    let series_feature = Series::new("feature", feature_names.to_vec());
    let series_value = Series::new("variance", values.to_vec());
    DataFrame::new(vec![series_feature, series_value]).map_err(RiskModelError::from)
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

fn unbiased_covariance(centered: &DMatrix<f64>) -> DMatrix<f64> {
    if centered.nrows() <= 1 {
        return DMatrix::zeros(centered.ncols(), centered.ncols());
    }
    (&centered.transpose() * centered) / (centered.nrows() as f64 - 1.0)
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

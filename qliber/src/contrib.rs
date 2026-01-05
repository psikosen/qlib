use polars::prelude::*;

use crate::dataset::{DatasetError, DatasetResult, Processor};

/// Configuration for Alpha158 feature generation
#[derive(Clone, Debug)]
pub struct Alpha158Config {
    pub windows: Vec<usize>,
    pub include_kbar: bool,
    pub include_price: bool,
    pub include_volume: bool,
    pub include_rolling: bool,
}

impl Default for Alpha158Config {
    fn default() -> Self {
        Self {
            windows: vec![5, 10, 20, 30, 60],
            include_kbar: true,
            include_price: true,
            include_volume: true,
            include_rolling: true,
        }
    }
}

pub struct Alpha158Processor {
    config: Alpha158Config,
}

impl Alpha158Processor {
    pub fn new() -> Self {
        Self {
            config: Alpha158Config::default(),
        }
    }

    pub fn with_config(config: Alpha158Config) -> Self {
        Self { config }
    }

    /// Generate KBAR features (9 features)
    fn generate_kbar_features(&self, frame: &DataFrame) -> DatasetResult<Vec<Series>> {
        let mut features = Vec::new();

        // Get required columns
        let open = frame.column("open")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;
        let close = frame.column("close")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;
        let high = frame.column("high")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;
        let low = frame.column("low")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;

        // KBAR features using comparison operators
        // KMID = (close - open) / open
        let kmid: Float64Chunked = close.into_iter()
            .zip(open.into_iter())
            .map(|(c, o)| match (c, o) {
                (Some(c_val), Some(o_val)) if o_val != 0.0 => Some((c_val - o_val) / o_val),
                _ => None,
            })
            .collect();
        features.push(kmid.into_series().with_name("KMID"));

        // KLEN = (high - low) / open
        let klen: Float64Chunked = high.into_iter()
            .zip(low.into_iter())
            .zip(open.into_iter())
            .map(|((h, l), o)| match (h, l, o) {
                (Some(h_val), Some(l_val), Some(o_val)) if o_val != 0.0 => Some((h_val - l_val) / o_val),
                _ => None,
            })
            .collect();
        features.push(klen.into_series().with_name("KLEN"));

        // KMID2 = (close - open) / (high - low)
        let kmid2: Float64Chunked = close.into_iter()
            .zip(open.into_iter())
            .zip(high.into_iter())
            .zip(low.into_iter())
            .map(|(((c, o), h), l)| match (c, o, h, l) {
                (Some(c_val), Some(o_val), Some(h_val), Some(l_val)) if (h_val - l_val) != 0.0 => {
                    Some((c_val - o_val) / (h_val - l_val))
                }
                _ => None,
            })
            .collect();
        features.push(kmid2.into_series().with_name("KMID2"));

        // KUP = (high - close.max(open)) / open
        let kup: Float64Chunked = high.into_iter()
            .zip(close.into_iter())
            .zip(open.into_iter())
            .map(|((h, c), o)| match (h, c, o) {
                (Some(h_val), Some(c_val), Some(o_val)) if o_val != 0.0 => {
                    Some((h_val - c_val.max(o_val)) / o_val)
                }
                _ => None,
            })
            .collect();
        features.push(kup.into_series().with_name("KUP"));

        // KDOWN = (close.min(open) - low) / open
        let kdown: Float64Chunked = close.into_iter()
            .zip(open.into_iter())
            .zip(low.into_iter())
            .map(|((c, o), l)| match (c, o, l) {
                (Some(c_val), Some(o_val), Some(l_val)) if o_val != 0.0 => {
                    Some((c_val.min(o_val) - l_val) / o_val)
                }
                _ => None,
            })
            .collect();
        features.push(kdown.into_series().with_name("KDOWN"));

        // KUP2 = (high - close.max(open)) / (high - low)
        let kup2: Float64Chunked = high.into_iter()
            .zip(close.into_iter())
            .zip(open.into_iter())
            .zip(low.into_iter())
            .map(|(((h, c), o), l)| match (h, c, o, l) {
                (Some(h_val), Some(c_val), Some(o_val), Some(l_val)) if (h_val - l_val) != 0.0 => {
                    Some((h_val - c_val.max(o_val)) / (h_val - l_val))
                }
                _ => None,
            })
            .collect();
        features.push(kup2.into_series().with_name("KUP2"));

        // KDOWN2 = (close.min(open) - low) / (high - low)
        let kdown2: Float64Chunked = close.into_iter()
            .zip(open.into_iter())
            .zip(high.into_iter())
            .zip(low.into_iter())
            .map(|(((c, o), h), l)| match (c, o, h, l) {
                (Some(c_val), Some(o_val), Some(h_val), Some(l_val)) if (h_val - l_val) != 0.0 => {
                    Some((c_val.min(o_val) - l_val) / (h_val - l_val))
                }
                _ => None,
            })
            .collect();
        features.push(kdown2.into_series().with_name("KDOWN2"));

        // KSFT = (close - open).shift(2) / open
        let close_open_diff: Vec<Option<f64>> = close.into_iter()
            .zip(open.into_iter())
            .map(|(c, o)| match (c, o) {
                (Some(c_val), Some(o_val)) => Some(c_val - o_val),
                _ => None,
            })
            .collect();

        let mut ksft_values = vec![None, None]; // First two values are None due to shift(2)
        for i in 0..close_open_diff.len().saturating_sub(2) {
            let diff = close_open_diff[i];
            let o = open.get(i + 2);
            ksft_values.push(match (diff, o) {
                (Some(d_val), Some(o_val)) if o_val != 0.0 => Some(d_val / o_val),
                _ => None,
            });
        }
        let ksft: Float64Chunked = ksft_values.into_iter().collect();
        features.push(ksft.into_series().with_name("KSFT"));

        // KSFT2 = (close - open).shift(2) / (high - low)
        let mut ksft2_values = vec![None, None];
        for i in 0..close_open_diff.len().saturating_sub(2) {
            let diff = close_open_diff[i];
            let h = high.get(i + 2);
            let l = low.get(i + 2);
            ksft2_values.push(match (diff, h, l) {
                (Some(d_val), Some(h_val), Some(l_val)) if (h_val - l_val) != 0.0 => {
                    Some(d_val / (h_val - l_val))
                }
                _ => None,
            });
        }
        let ksft2: Float64Chunked = ksft2_values.into_iter().collect();
        features.push(ksft2.into_series().with_name("KSFT2"));

        Ok(features)
    }

    /// Generate price-based rolling features
    fn generate_price_features(&self, frame: &DataFrame) -> DatasetResult<Vec<Series>> {
        let mut features = Vec::new();

        let close = frame.column("close")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;

        for &window in &self.config.windows {
            // ROC: ref(close, d) / close
            let roc = self.compute_roc(close, window)?;
            features.push(roc.into_series().with_name(&format!("ROC_{}", window)));

            // MA: mean(close, d) / close
            let ma = self.compute_ma(close, window)?;
            features.push(ma.into_series().with_name(&format!("MA_{}", window)));

            // STD: std(close, d) / close
            let std = self.compute_std(close, window)?;
            features.push(std.into_series().with_name(&format!("STD_{}", window)));
        }

        Ok(features)
    }

    /// Generate volume-based rolling features
    fn generate_volume_features(&self, frame: &DataFrame) -> DatasetResult<Vec<Series>> {
        let mut features = Vec::new();

        let volume = frame.column("volume")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;

        for &window in &self.config.windows {
            // VROC: ref(volume, d) / volume
            let vroc = self.compute_roc(volume, window)?;
            features.push(vroc.into_series().with_name(&format!("VROC_{}", window)));

            // VMA: mean(volume, d) / volume
            let vma = self.compute_ma(volume, window)?;
            features.push(vma.into_series().with_name(&format!("VMA_{}", window)));

            // VSTD: std(volume, d) / volume
            let vstd = self.compute_std(volume, window)?;
            features.push(vstd.into_series().with_name(&format!("VSTD_{}", window)));
        }

        Ok(features)
    }

    /// Generate advanced rolling features (BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK)
    fn generate_rolling_features(&self, frame: &DataFrame) -> DatasetResult<Vec<Series>> {
        let mut features = Vec::new();

        let close = frame.column("close")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;

        for &window in &self.config.windows {
            // BETA: slope(close, d) / close
            let beta = self.compute_beta(close, window)?;
            features.push(beta.into_series().with_name(&format!("BETA_{}", window)));

            // RSQR: rsquare(close, d)
            let rsqr = self.compute_rsqr(close, window)?;
            features.push(rsqr.into_series().with_name(&format!("RSQR_{}", window)));

            // RESI: resi(close, d) / close
            let resi = self.compute_resi(close, window)?;
            features.push(resi.into_series().with_name(&format!("RESI_{}", window)));

            // MAX: max(close, d) / close
            let max = self.compute_max(close, window)?;
            features.push(max.into_series().with_name(&format!("MAX_{}", window)));

            // MIN: min(close, d) / close
            let min = self.compute_min(close, window)?;
            features.push(min.into_series().with_name(&format!("MIN_{}", window)));

            // QTLU: quantile(close, d, 0.8) / close
            let qtlu = self.compute_quantile(close, window, 0.8)?;
            features.push(qtlu.into_series().with_name(&format!("QTLU_{}", window)));

            // QTLD: quantile(close, d, 0.2) / close
            let qtld = self.compute_quantile(close, window, 0.2)?;
            features.push(qtld.into_series().with_name(&format!("QTLD_{}", window)));

            // RANK: rank(close, d)
            let rank = self.compute_rank(close, window)?;
            features.push(rank.into_series().with_name(&format!("RANK_{}", window)));
        }

        Ok(features)
    }

    // Helper functions for feature computation
    fn compute_roc(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window {
                result.push(None);
            } else {
                let current = data.get(i);
                let lagged = data.get(i - window);
                result.push(match (current, lagged) {
                    (Some(c), Some(l)) if c != 0.0 => Some(l / c),
                    _ => None,
                });
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_ma(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                let current = data.get(i);
                if window_data.len() == window {
                    let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                    result.push(current.map(|c| if c != 0.0 { mean / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_std(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                let current = data.get(i);
                if window_data.len() == window {
                    let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                    let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window_data.len() as f64;
                    let std = variance.sqrt();
                    result.push(current.map(|c| if c != 0.0 { std / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_beta(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                let current = data.get(i);
                if window_data.len() == window {
                    let (slope, _, _) = self.linear_regression_stats(&window_data);
                    result.push(current.map(|c| if c != 0.0 { slope / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_rsqr(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                if window_data.len() == window {
                    let (_, r_square, _) = self.linear_regression_stats(&window_data);
                    result.push(Some(r_square));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_resi(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                let current = data.get(i);
                if window_data.len() == window {
                    let (_, _, residual) = self.linear_regression_stats(&window_data);
                    result.push(current.map(|c| if c != 0.0 { residual / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_max(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_max = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .fold(f64::NEG_INFINITY, f64::max);

                let current = data.get(i);
                if window_max.is_finite() {
                    result.push(current.map(|c| if c != 0.0 { window_max / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_min(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_min = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .fold(f64::INFINITY, f64::min);

                let current = data.get(i);
                if window_min.is_finite() {
                    result.push(current.map(|c| if c != 0.0 { window_min / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_quantile(&self, data: &Float64Chunked, window: usize, q: f64) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let mut window_data: Vec<f64> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx))
                    .collect();

                let current = data.get(i);
                if window_data.len() == window {
                    window_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let idx = ((window_data.len() as f64 - 1.0) * q) as usize;
                    let quantile = window_data[idx];
                    result.push(current.map(|c| if c != 0.0 { quantile / c } else { 0.0 }));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn compute_rank(&self, data: &Float64Chunked, window: usize) -> DatasetResult<Float64Chunked> {
        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(None);
            } else {
                let window_data: Vec<(usize, f64)> = (i.saturating_sub(window - 1)..=i)
                    .filter_map(|idx| data.get(idx).map(|v| (idx, v)))
                    .collect();

                if window_data.len() == window {
                    let current_val = data.get(i).unwrap();
                    let rank = window_data.iter().filter(|(_, v)| v < &current_val).count() as f64 + 1.0;
                    let normalized_rank = rank / window as f64;
                    result.push(Some(normalized_rank));
                } else {
                    result.push(None);
                }
            }
        }
        Ok(result.into_iter().collect())
    }

    fn linear_regression_stats(&self, values: &[f64]) -> (f64, f64, f64) {
        let n = values.len() as f64;
        let x_mean = (1.0 + n) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = (i + 1) as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = (i + 1) as f64;
            let y_pred = slope * x + intercept;
            ss_res += (y - y_pred).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        let r_square = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };
        let residual = values.last().unwrap() - (slope * n + intercept);

        (slope, r_square, residual)
    }
}

impl Default for Alpha158Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Processor for Alpha158Processor {
    fn name(&self) -> &str {
        "alpha158"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        // Handle empty DataFrame
        if frame.height() == 0 {
            return Ok(frame);
        }

        let mut all_features = Vec::new();

        // Generate all feature categories
        if self.config.include_kbar {
            let kbar_features = self.generate_kbar_features(&frame)?;
            all_features.extend(kbar_features);
        }

        if self.config.include_price {
            let price_features = self.generate_price_features(&frame)?;
            all_features.extend(price_features);
        }

        if self.config.include_volume {
            let volume_features = self.generate_volume_features(&frame)?;
            all_features.extend(volume_features);
        }

        if self.config.include_rolling {
            let rolling_features = self.generate_rolling_features(&frame)?;
            all_features.extend(rolling_features);
        }

        // Add all features to the DataFrame
        let mut result = frame.clone();
        for feature in all_features {
            let _ = result.with_column(feature)
                .map_err(|source| DatasetError::Transform { source })?;
        }

        Ok(result)
    }
}

/// Configuration for Alpha360 feature generation
#[derive(Clone, Debug)]
pub struct Alpha360Config {
    pub lookback_days: usize,
    pub fields: Vec<String>,
}

impl Default for Alpha360Config {
    fn default() -> Self {
        Self {
            lookback_days: 60,
            fields: vec![
                "open".to_string(),
                "close".to_string(),
                "high".to_string(),
                "low".to_string(),
                "vwap".to_string(),
                "volume".to_string(),
            ],
        }
    }
}

pub struct Alpha360Processor {
    config: Alpha360Config,
}

impl Alpha360Processor {
    pub fn new() -> Self {
        Self {
            config: Alpha360Config::default(),
        }
    }

    pub fn with_config(config: Alpha360Config) -> Self {
        Self { config }
    }

    /// Generate all 360 features (60 days × 6 fields)
    fn generate_alpha360_features(&self, frame: &DataFrame) -> DatasetResult<Vec<Series>> {
        let mut features = Vec::new();

        for field in &self.config.fields {
            let data = frame.column(field)
                .map_err(|source| DatasetError::Transform { source })?
                .f64()
                .map_err(|source| DatasetError::Transform { source })?;

            // Generate 60-day lookback features: ref(field, i) / close
            for day in 0..self.config.lookback_days {
                let feature = self.compute_lookback_feature(data, day, frame)?;
                features.push(feature.into_series().with_name(&format!("{}_{}", field, day)));
            }
        }

        Ok(features)
    }

    fn compute_lookback_feature(
        &self,
        data: &Float64Chunked,
        lookback: usize,
        frame: &DataFrame,
    ) -> DatasetResult<Float64Chunked> {
        let close = frame.column("close")
            .map_err(|source| DatasetError::Transform { source })?
            .f64()
            .map_err(|source| DatasetError::Transform { source })?;

        let mut result = Vec::new();
        for i in 0..data.len() {
            if i < lookback {
                result.push(None);
            } else {
                let historical = data.get(i - lookback);
                let current_close = close.get(i);
                result.push(match (historical, current_close) {
                    (Some(h), Some(c)) if c != 0.0 => Some(h / c),
                    _ => None,
                });
            }
        }
        Ok(result.into_iter().collect())
    }
}

impl Default for Alpha360Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Processor for Alpha360Processor {
    fn name(&self) -> &str {
        "alpha360"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        // Handle empty DataFrame
        if frame.height() == 0 {
            return Ok(frame);
        }

        // Generate all 360 features
        let all_features = self.generate_alpha360_features(&frame)?;

        // Add all features to the DataFrame
        let mut result = frame.clone();
        for feature in all_features {
            let _ = result.with_column(feature)
                .map_err(|source| DatasetError::Transform { source })?;
        }

        Ok(result)
    }
}

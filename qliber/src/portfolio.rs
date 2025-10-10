use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

use chrono::{DateTime, NaiveDate, Utc};
use polars::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum PortfolioError {
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("date column `{0}` not found in DataFrame")]
    MissingDateColumn(String),
    #[error("date column `{0}` has unsupported dtype {1}")]
    UnsupportedDateType(String, String),
    #[error("price for symbol `{symbol}` missing on {date}")]
    MissingPrice { date: NaiveDate, symbol: String },
    #[error("portfolio contains no positions")]
    EmptyPositions,
    #[error("insufficient overlapping observations for correlation/beta calculations")]
    InsufficientObservations,
    #[error("invalid interest method `{0}`; expected `ci` or `si`")]
    InvalidInterestMethod(String),
}

pub type PortfolioResult<T> = Result<T, PortfolioError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterestMethod {
    Compound,
    Simple,
}

impl FromStr for InterestMethod {
    type Err = PortfolioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "ci" | "compound" => Ok(InterestMethod::Compound),
            "si" | "simple" => Ok(InterestMethod::Simple),
            other => Err(PortfolioError::InvalidInterestMethod(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Holding {
    pub amount: f64,
    pub price: Option<f64>,
}

impl Holding {
    pub fn new(amount: f64, price: Option<f64>) -> Self {
        Self { amount, price }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PortfolioSnapshot {
    pub holdings: BTreeMap<String, Holding>,
    pub cash: f64,
}

impl PortfolioSnapshot {
    pub fn new(holdings: BTreeMap<String, Holding>, cash: f64) -> Self {
        Self { holdings, cash }
    }

    pub fn with_cash(cash: f64) -> Self {
        Self {
            holdings: BTreeMap::new(),
            cash,
        }
    }

    pub fn insert_holding(&mut self, symbol: impl Into<String>, holding: Holding) {
        self.holdings.insert(symbol.into(), holding);
    }
}

#[derive(Debug, Clone)]
pub struct PriceMatrix {
    data: BTreeMap<NaiveDate, HashMap<String, f64>>,
}

impl PriceMatrix {
    pub fn from_dataframe(frame: &DataFrame, date_column: &str) -> PortfolioResult<Self> {
        let date_series = frame
            .column(date_column)
            .map_err(|_| PortfolioError::MissingDateColumn(date_column.to_string()))?;

        let dates = extract_dates(date_series, date_column)?;
        let mut data: BTreeMap<NaiveDate, HashMap<String, f64>> = BTreeMap::new();

        let mut price_columns = Vec::new();
        for column in frame.get_columns() {
            if column.name() != date_column {
                let converted = series_to_f64(column)?;
                price_columns.push((column.name().to_string(), converted));
            }
        }

        for (row_idx, date) in dates.iter().enumerate() {
            let mut row = HashMap::new();
            for (name, column) in &price_columns {
                match column.get(row_idx) {
                    Some(value) if value.is_finite() => {
                        row.insert(name.clone(), value);
                    }
                    _ => {
                        return Err(PortfolioError::MissingPrice {
                            date: *date,
                            symbol: name.clone(),
                        });
                    }
                }
            }
            data.insert(*date, row);
        }

        log_event(
            file!(),
            "PriceMatrix",
            "from_dataframe",
            "portfolio.prices",
            line!(),
            &format!(
                "Constructed price matrix with {} rows and {} symbols",
                data.len(),
                price_columns.len()
            ),
            None,
            "none",
            "GET",
        );

        Ok(Self { data })
    }

    pub fn price(&self, date: &NaiveDate, symbol: &str) -> Option<f64> {
        self.data.get(date).and_then(|row| row.get(symbol)).copied()
    }

    pub fn dates(&self) -> impl Iterator<Item = &NaiveDate> {
        self.data.keys()
    }
}

pub fn position_value(
    snapshot: &PortfolioSnapshot,
    prices: &PriceMatrix,
    date: NaiveDate,
) -> PortfolioResult<f64> {
    let mut total = snapshot.cash;

    for (symbol, holding) in &snapshot.holdings {
        let price = prices
            .price(&date, symbol)
            .ok_or_else(|| PortfolioError::MissingPrice {
                date,
                symbol: symbol.clone(),
            })?;
        total += holding.amount * price;
    }

    log_event(
        file!(),
        "PortfolioAnalytics",
        "position_value",
        "portfolio.valuation",
        line!(),
        &format!(
            "Computed portfolio value on {date} across {} holdings",
            snapshot.holdings.len()
        ),
        None,
        "none",
        "GET",
    );

    Ok(total)
}

pub fn position_value_series(
    positions: &BTreeMap<NaiveDate, PortfolioSnapshot>,
    prices: &PriceMatrix,
) -> PortfolioResult<BTreeMap<NaiveDate, f64>> {
    if positions.is_empty() {
        return Err(PortfolioError::EmptyPositions);
    }

    let mut series = BTreeMap::new();
    for (date, snapshot) in positions {
        let value = position_value(snapshot, prices, *date)?;
        series.insert(*date, value);
    }

    log_event(
        file!(),
        "PortfolioAnalytics",
        "position_value_series",
        "portfolio.valuation",
        line!(),
        &format!("Evaluated {} position snapshots", series.len()),
        None,
        "none",
        "GET",
    );

    Ok(series)
}

pub fn daily_return_series(
    positions: &BTreeMap<NaiveDate, PortfolioSnapshot>,
    prices: &PriceMatrix,
    init_asset_value: f64,
) -> PortfolioResult<Vec<(NaiveDate, f64)>> {
    let series = position_value_series(positions, prices)?;
    let mut returns = Vec::new();
    let mut prev_value: Option<f64> = None;

    for (idx, (date, value)) in series.iter().enumerate() {
        let daily = if idx == 0 {
            value / init_asset_value - 1.0
        } else if let Some(prev) = prev_value {
            if prev.abs() <= f64::EPSILON {
                0.0
            } else {
                value / prev - 1.0
            }
        } else {
            0.0
        };
        returns.push((*date, daily));
        prev_value = Some(*value);
    }

    log_event(
        file!(),
        "PortfolioAnalytics",
        "daily_return_series",
        "portfolio.returns",
        line!(),
        &format!("Computed {} daily returns", returns.len()),
        None,
        "none",
        "GET",
    );

    Ok(returns)
}

pub fn annual_return_from_positions(
    positions: &BTreeMap<NaiveDate, PortfolioSnapshot>,
    prices: &PriceMatrix,
    init_asset_value: f64,
    periods_per_year: f64,
) -> PortfolioResult<f64> {
    let series = position_value_series(positions, prices)?;
    let count = series.len();
    if count == 0 {
        return Err(PortfolioError::EmptyPositions);
    }

    let final_value = series.values().last().copied().unwrap_or(init_asset_value);
    let n_period = count as f64;
    let annual = if init_asset_value <= f64::EPSILON {
        0.0
    } else {
        (final_value / init_asset_value).powf(periods_per_year / n_period) - 1.0
    };

    log_event(
        file!(),
        "PortfolioAnalytics",
        "annual_return_from_positions",
        "portfolio.returns",
        line!(),
        &format!(
            "Annualized portfolio return {:.6} over {} periods",
            annual, count
        ),
        None,
        "none",
        "GET",
    );

    Ok(annual)
}

pub fn annual_return_from_returns(
    returns: &[f64],
    method: InterestMethod,
    periods_per_year: f64,
) -> f64 {
    let clean_returns: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();

    if clean_returns.is_empty() {
        return 0.0;
    }

    let mean = clean_returns.iter().copied().sum::<f64>() / clean_returns.len() as f64;

    let annual = match method {
        InterestMethod::Compound => (1.0 + mean).powf(periods_per_year) - 1.0,
        InterestMethod::Simple => mean * periods_per_year,
    };

    log_event(
        file!(),
        "PortfolioAnalytics",
        "annual_return_from_returns",
        "portfolio.returns",
        line!(),
        &format!(
            "Computed annual return {:.6} using {:?} method",
            annual, method
        ),
        None,
        "none",
        "GET",
    );

    annual
}

pub fn sharpe_ratio_from_returns(
    returns: &[f64],
    risk_free_rate: f64,
    method: InterestMethod,
    periods_per_year: f64,
) -> f64 {
    let clean_returns: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();

    if clean_returns.len() < 2 {
        return 0.0;
    }

    let mean = clean_returns.iter().copied().sum::<f64>() / clean_returns.len() as f64;
    let variance = sample_variance(&clean_returns, mean);
    let std_dev = variance.sqrt();
    if std_dev <= f64::EPSILON {
        return 0.0;
    }

    let annual = annual_return_from_returns(&clean_returns, method, periods_per_year);
    let sharpe = (annual - risk_free_rate) / std_dev / periods_per_year.sqrt();

    log_event(
        file!(),
        "PortfolioAnalytics",
        "sharpe_ratio_from_returns",
        "portfolio.risk",
        line!(),
        &format!(
            "Computed Sharpe ratio {:.6} with risk-free rate {:.4}",
            sharpe, risk_free_rate
        ),
        None,
        "none",
        "GET",
    );

    sharpe
}

pub fn max_drawdown_from_returns(returns: &[f64]) -> f64 {
    let clean_returns: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();

    if clean_returns.is_empty() {
        return 0.0;
    }

    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_drawdown = 0.0;

    for value in clean_returns {
        cumulative *= 1.0 + value;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (cumulative / peak) - 1.0;
        if drawdown < max_drawdown {
            max_drawdown = drawdown;
        }
    }

    log_event(
        file!(),
        "PortfolioAnalytics",
        "max_drawdown_from_returns",
        "portfolio.risk",
        line!(),
        &format!("Computed max drawdown {:.6}", max_drawdown),
        None,
        "none",
        "GET",
    );

    max_drawdown
}

pub fn beta(returns: &[f64], benchmark: &[f64]) -> PortfolioResult<f64> {
    let (clean_r, clean_b) = align_series(returns, benchmark);
    if clean_r.len() < 2 {
        return Err(PortfolioError::InsufficientObservations);
    }

    let cov = covariance(&clean_r, &clean_b);
    let var_b = variance(&clean_b);
    if var_b <= f64::EPSILON {
        return Err(PortfolioError::InsufficientObservations);
    }

    let beta = cov / var_b;

    log_event(
        file!(),
        "PortfolioAnalytics",
        "beta",
        "portfolio.risk",
        line!(),
        &format!("Computed beta {:.6}", beta),
        None,
        "none",
        "GET",
    );

    Ok(beta)
}

pub fn alpha(
    returns: &[f64],
    benchmark: &[f64],
    risk_free_rate: f64,
    method: InterestMethod,
    periods_per_year: f64,
) -> PortfolioResult<f64> {
    let beta_val = beta(returns, benchmark)?;
    let annual_r = annual_return_from_returns(returns, method, periods_per_year);
    let annual_b = annual_return_from_returns(benchmark, method, periods_per_year);
    let alpha = annual_r - risk_free_rate - beta_val * (annual_b - risk_free_rate);

    log_event(
        file!(),
        "PortfolioAnalytics",
        "alpha",
        "portfolio.risk",
        line!(),
        &format!("Computed alpha {:.6}", alpha),
        None,
        "none",
        "GET",
    );

    Ok(alpha)
}

pub fn volatility(returns: &[f64]) -> f64 {
    let clean_returns: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();

    if clean_returns.len() < 2 {
        return 0.0;
    }

    let mean = clean_returns.iter().copied().sum::<f64>() / clean_returns.len() as f64;
    let variance = sample_variance(&clean_returns, mean);
    let vol = variance.sqrt();

    log_event(
        file!(),
        "PortfolioAnalytics",
        "volatility",
        "portfolio.risk",
        line!(),
        &format!("Computed volatility {:.6}", vol),
        None,
        "none",
        "GET",
    );

    vol
}

pub fn rank_information_coefficient(a: &[f64], b: &[f64]) -> PortfolioResult<f64> {
    let (clean_a, clean_b) = align_series(a, b);
    if clean_a.len() < 2 {
        return Err(PortfolioError::InsufficientObservations);
    }

    let ranks_a = average_ranks(&clean_a);
    let ranks_b = average_ranks(&clean_b);
    let ic = pearson_correlation(&ranks_a, &ranks_b).unwrap_or(0.0);

    log_event(
        file!(),
        "PortfolioAnalytics",
        "rank_information_coefficient",
        "portfolio.correlation",
        line!(),
        &format!("Computed rank IC {:.6}", ic),
        None,
        "none",
        "GET",
    );

    Ok(ic)
}

pub fn information_coefficient(a: &[f64], b: &[f64]) -> PortfolioResult<f64> {
    let (clean_a, clean_b) = align_series(a, b);
    if clean_a.len() < 2 {
        return Err(PortfolioError::InsufficientObservations);
    }

    let ic = pearson_correlation(&clean_a, &clean_b).unwrap_or(0.0);

    log_event(
        file!(),
        "PortfolioAnalytics",
        "information_coefficient",
        "portfolio.correlation",
        line!(),
        &format!("Computed normal IC {:.6}", ic),
        None,
        "none",
        "GET",
    );

    Ok(ic)
}

fn extract_dates(series: &Series, name: &str) -> PortfolioResult<Vec<NaiveDate>> {
    match series.dtype() {
        DataType::Date => {
            let casted = series.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
            extract_from_datetime(&casted)
        }
        DataType::Datetime(_, _) => extract_from_datetime(series),
        DataType::Utf8 => {
            let utf8 = series.utf8()?;
            let mut dates = Vec::with_capacity(utf8.len());
            for value in utf8.into_iter() {
                if let Some(value) = value {
                    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
                        .or_else(|_| NaiveDate::parse_from_str(value, "%Y/%m/%d"))
                        .map_err(|_| {
                            PortfolioError::UnsupportedDateType(
                                name.to_string(),
                                "utf8 string".to_string(),
                            )
                        })?;
                    dates.push(date);
                } else {
                    return Err(PortfolioError::UnsupportedDateType(
                        name.to_string(),
                        "null".to_string(),
                    ));
                }
            }
            Ok(dates)
        }
        other => Err(PortfolioError::UnsupportedDateType(
            name.to_string(),
            format!("{other:?}"),
        )),
    }
}

fn extract_from_datetime(series: &Series) -> PortfolioResult<Vec<NaiveDate>> {
    let dtype_description = format!("{:?}", series.dtype());
    let datetime = series.datetime().map_err(|_| {
        PortfolioError::UnsupportedDateType(series.name().to_string(), dtype_description.clone())
    })?;
    let mut dates = Vec::with_capacity(datetime.len());
    for value in datetime.into_iter() {
        if let Some(millis) = value {
            if let Some(dt) = DateTime::<Utc>::from_timestamp_millis(millis) {
                dates.push(dt.date_naive());
            } else {
                return Err(PortfolioError::UnsupportedDateType(
                    series.name().to_string(),
                    "out-of-range datetime".to_string(),
                ));
            }
        } else {
            return Err(PortfolioError::UnsupportedDateType(
                series.name().to_string(),
                "null".to_string(),
            ));
        }
    }
    Ok(dates)
}

fn series_to_f64(series: &Series) -> PortfolioResult<Float64Chunked> {
    if matches!(series.dtype(), DataType::Float64) {
        Ok(series.f64()?.clone())
    } else {
        let casted = series.cast(&DataType::Float64)?;
        Ok(casted.f64().expect("series cast to f64").clone())
    }
}

fn align_series(a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .unzip()
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let sum_squares = values
        .par_iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>();
    sum_squares / (values.len() as f64 - 1.0)
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    sample_variance(values, mean)
}

fn covariance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }

    let mean_a = a.iter().copied().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().copied().sum::<f64>() / b.len() as f64;

    let cov_sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    cov_sum / (a.len() as f64 - 1.0)
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }

    let mean_a = a.iter().copied().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().copied().sum::<f64>() / b.len() as f64;

    let mut numerator = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        numerator += dx * dy;
        denom_a += dx * dx;
        denom_b += dy * dy;
    }

    let denominator = (denom_a * denom_b).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < indexed.len() {
        let start = i;
        let value = indexed[i].1;
        let mut end = i + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let first_rank = start + 1;
        let last_rank = end;
        let average_rank = (first_rank + last_rank) as f64 / 2.0;
        for (original, _) in indexed[start..end].iter() {
            ranks[*original] = average_rank;
        }
        i = end;
    }

    ranks
}

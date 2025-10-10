//! qliber is a Rust-native port of core concepts from Microsoft's Qlib project.
//! It provides efficient dataset ingestion, feature engineering, metrics evaluation,
//! and structured logging tailored for quantitative research pipelines.

pub mod dataset;
pub mod features;
pub mod logging;
pub mod metrics;
pub mod portfolio;

pub use dataset::{DatasetError, MarketData};
pub use features::{with_daily_returns, with_moving_average, with_z_score};
pub use metrics::{
    AccumulationMode, AnalysisFrequency, FrequencyUnit, IndicatorMethod, MetricsError,
    MetricsResult, PerformanceMetrics, indicator_analysis, indicator_analysis_with_method,
    risk_analysis,
};
pub use portfolio::{
    Holding, InterestMethod, PortfolioError, PortfolioResult, PortfolioSnapshot, PriceMatrix,
    alpha, annual_return_from_positions, annual_return_from_returns, beta, daily_return_series,
    information_coefficient, max_drawdown_from_returns, position_value, position_value_series,
    rank_information_coefficient, sharpe_ratio_from_returns, volatility,
};

pub type Result<T> = anyhow::Result<T>;

//! qliber is a Rust-native port of core concepts from Microsoft's Qlib project.
//! It provides efficient dataset ingestion, feature engineering, metrics evaluation,
//! and structured logging tailored for quantitative research pipelines.

pub mod dataset;
pub mod features;
pub mod logging;
pub mod metrics;

pub use dataset::{DatasetError, MarketData};
pub use features::{with_daily_returns, with_moving_average, with_z_score};
pub use metrics::{
    AccumulationMode, AnalysisFrequency, FrequencyUnit, IndicatorMethod, MetricsError,
    MetricsResult, PerformanceMetrics, indicator_analysis,
};

pub type Result<T> = anyhow::Result<T>;

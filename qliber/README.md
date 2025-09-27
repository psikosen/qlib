# qliber

qliber is a Rust-native, performance-focused reimagination of [Microsoft's Qlib](https://github.com/microsoft/qlib).
It provides fast dataset ingestion, feature engineering, and evaluation utilities tailored for
quantitative finance research pipelines.

## Design Goals

- **Performance-first analytics** powered by [Polars](https://www.pola.rs/), leveraging efficient memory layouts and lazy execution.
- **Deterministic feature engineering** implemented with numerically stable rolling windows.
- **Robust evaluation** metrics with data-parallel aggregation via [Rayon](https://docs.rs/rayon/latest/rayon/), matching
  Qlib's risk analysis outputs (cumulative return, annualized stats, information ratio, and max drawdown).
- **Structured observability** implemented through [tracing](https://docs.rs/tracing/latest/tracing/) with a canonical JSON schema and the "Continuous skepticism" prompts required by the guidelines.

## Porting Scope

Microsoft Qlib delivers a research platform composed of three pillars: dataset ingestion, feature engineering, and evaluation.
qliber mirrors these building blocks with the following Rust-native equivalents:

- **Data Server → `dataset` module:** lazy CSV ingestion, column projection, and temporal filtering.
- **Feature Library → `features` module:** rolling statistics, return computation, and normalization helpers.
- **Workflow & Evaluation → `metrics` module:** cumulative/annualized return aggregation, Sharpe/information ratios,
  and drawdown metrics with both arithmetic and geometric accumulation modes.

This initial slice prioritizes correctness and extensibility; additional modules such as model training or portfolio optimization can be layered atop these primitives in follow-up iterations.

## Project Layout

```
qliber/
├── Cargo.toml
├── README.md
├── src
│   ├── dataset.rs      # Lazy CSV ingestion and column selection utilities
│   ├── features.rs     # Feature engineering helpers (returns, moving averages, z-scores)
│   ├── logging.rs      # Structured logging initialization and helpers
│   ├── metrics.rs      # Performance metric calculations (cumulative, annualized, ratios, drawdowns)
│   └── lib.rs          # Public crate exports
└── tests
    └── pipeline.rs     # End-to-end regression test covering the primary flow
```

## Usage

```rust
use chrono::Utc;
use qliber::{
    with_daily_returns, with_moving_average, with_z_score, AccumulationMode, MarketData, PerformanceMetrics,
};

fn main() -> anyhow::Result<()> {
    qliber::logging::init_logging()?;
    let market = MarketData::from_csv("data/market.csv")?;
    let filtered = market.filter_date_range(
        "timestamp",
        Some(Utc::with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()),
        None,
    )?;

    let dataframe = filtered.collect()?;
    let dataframe = with_daily_returns(&dataframe, "close", "return")?;
    let dataframe = with_moving_average(&dataframe, "close", 5, "ma_5")?;
    let dataframe = with_z_score(&dataframe, "close", 10, "z_close")?;

    let returns = dataframe
        .column("return")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let metrics = PerformanceMetrics::evaluate_with_mode(&returns, 252.0, AccumulationMode::Product);

    println!(
        "Annualized return: {:.3}, Information ratio: {:.3}",
        metrics.annualized_return, metrics.information_ratio
    );

    Ok(())
}
```

## Development

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

The integration test exercises the full data pipeline using temporary CSV inputs to ensure end-to-end correctness.

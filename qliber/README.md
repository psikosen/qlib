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
- **Provider Infrastructure → `provider` module:** instrument registry, trading calendars, in-memory feature caching, and point-in-time storage primitives mirroring `qlib.data` providers.
- **Feature Library → `features` module:** rolling statistics, return computation, and normalization helpers.
- **Workflow & Evaluation → `metrics` module:** cumulative/annualized return aggregation, Sharpe/information ratios,
  and drawdown metrics with both arithmetic and geometric accumulation modes, frequency-aware scaling,
  and trade indicator weighting analysis that mirrors Qlib's Python helpers.
- **Portfolio Evaluation → `portfolio` module:** position valuation, portfolio-level return series, Sharpe/alpha/beta,
  and information coefficient utilities aligned with `qlib.contrib.evaluate_portfolio` semantics.

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
│   ├── portfolio.rs    # Portfolio analytics matching qlib.contrib.evaluate_portfolio
│   ├── provider.rs     # Trading calendars, instrument registry, feature caching, and PIT storage
│   └── lib.rs          # Public crate exports
└── tests
    ├── pipeline.rs     # End-to-end regression test covering the primary flow
    └── provider.rs     # Provider infrastructure unit coverage (calendars, cache, PIT)
```

## Usage

```rust
use chrono::Utc;
use qliber::{
    alpha, annual_return_from_returns, beta, indicator_analysis, max_drawdown_from_returns,
    rank_information_coefficient, with_daily_returns, with_moving_average, with_z_score,
    AccumulationMode, Holding, IndicatorMethod, InterestMethod, MarketData, PerformanceMetrics,
    PortfolioSnapshot, PriceMatrix,
};

fn main() -> anyhow::Result<()> {
    qliber::init(qliber::DefaultConfig::Client, qliber::InitOptions::default())?;
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
    let metrics = PerformanceMetrics::evaluate_with_frequency_str(&returns, "day", AccumulationMode::Product)?;

    println!(
        "Annualized return: {:.3}, Information ratio: {:.3}",
        metrics.annualized_return, metrics.information_ratio
    );

    let trade_frame = polars::df! {
        "count" => &[5.0, 10.0, 20.0],
        "ffr" => &[0.1, 0.5, 0.9],
        "pa" => &[0.2, 0.8, 0.4],
        "pos" => &[0.3, 0.6, 0.7],
        "deal_amount" => &[100.0, 400.0, 50.0],
        "value" => &[1000.0, 200.0, 800.0],
    }?;
    let indicator_stats = indicator_analysis(&trade_frame, IndicatorMethod::AmountWeighted)?;
    println!("Indicator analysis:\n{}", indicator_stats);

    let risk_frame = qliber::risk_analysis(&returns, Some(252.0), None, Some("sum"))?;
    let value_weighted = qliber::indicator_analysis_with_method(&trade_frame, "value_weighted")?;
    println!("Risk analysis:\n{}", risk_frame);
    println!("Value-weighted indicators:\n{}", value_weighted);

    // Portfolio evaluation helpers mirroring qlib.contrib.evaluate_portfolio
    let price_frame = polars::df! {
        "date" => &["2024-01-02", "2024-01-03"],
        "AAA" => &[10.0, 11.0],
        "BBB" => &[20.0, 21.0],
    }?;
    let prices = PriceMatrix::from_dataframe(&price_frame, "date")?;
    let mut day1 = PortfolioSnapshot::with_cash(50.0);
    day1.insert_holding("AAA", Holding::new(5.0, None));
    let mut day2 = PortfolioSnapshot::with_cash(50.0);
    day2.insert_holding("AAA", Holding::new(5.0, None));
    let mut positions = std::collections::BTreeMap::new();
    positions.insert(chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(), day1);
    positions.insert(chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(), day2);

    let portfolio_returns = qliber::daily_return_series(&positions, &prices, 100.0)?;
    let sharpe = qliber::sharpe_ratio_from_returns(
        &portfolio_returns.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
        0.02,
        InterestMethod::Compound,
        252.0,
    );
    let alpha_value = alpha(
        &returns,
        &returns, // placeholder benchmark
        0.02,
        InterestMethod::Compound,
        252.0,
    )?;
    let beta_value = beta(&returns, &returns)?;
    let rank_ic = rank_information_coefficient(&returns, &returns)?;
    let max_dd = max_drawdown_from_returns(&returns);
    let annual = annual_return_from_returns(&returns, InterestMethod::Compound, 252.0);

    println!("Portfolio daily returns: {:?}", portfolio_returns);
    println!("Sharpe ratio: {:.3}", sharpe);
    println!("Alpha: {:.3}, Beta: {:.3}", alpha_value, beta_value);
    println!("Rank IC: {:.3}, Max Drawdown: {:.3}", rank_ic, max_dd);
    println!("Annualized return: {:.3}", annual);

    Ok(())
}
```

The `risk_analysis` and `indicator_analysis_with_method` helpers accept the same string-based
options as Qlib's Python API, making it straightforward to port workflows that rely on
`mode="sum"/"product"` or indicator weighting strings without changing call sites.

### Modeling

`qliber::trainer` mirrors Qlib's model orchestration layer. Alongside the baseline
`MeanModel`, the crate now provides an `XgBoostModel` built on SmartCore's gradient
boosting implementation to cover the LightGBM-style scenarios documented in the Python
stack. Both models expose the common `TrainableModel` trait so they can be dropped into
existing trainer pipelines:

```rust
use polars::prelude::*;
use qliber::{ExperimentManager, Trainer, XgBoostModel, XgBoostParameters};

let features = DataFrame::new(vec![Series::new("feature", vec![1.0, 2.0, 3.0])])?;
let labels = DataFrame::new(vec![Series::new("label", vec![3.0, 5.0, 7.0])])?;

let manager = ExperimentManager::new();
let recorder = manager.start("boosting-demo");
let params = XgBoostParameters::default().with_n_estimators(50).with_max_depth(3);
let mut model = XgBoostModel::new("label", vec!["feature".into()]).with_parameters(params);
let mut trainer = Trainer::new(recorder, &mut model, "label", 1);
trainer.train(&features, &labels)?;
```

`XgBoostParameters` re-exports SmartCore's builder so downstream applications can tune
learning rate, tree depth, regularization, and sampling hyper-parameters without depending
on SmartCore directly.

### Configuration

`qliber::init` mirrors the behavior of `qlib.config` and `qlib.init` by providing global
configuration, logging initialization, and cache management:

- Select a `DefaultConfig` (`Client` or `Server`) and override parameters with
  `InitOptions` (provider URIs, mount paths, logging level/filter, cache settings, and
  computation kernels).
- Control the geographic preset via the exported `REG_CN`, `REG_US`, and `REG_TW`
  region constants.
- Automatically clear registered in-memory feature caches on initialization, matching the
  Python `clear_mem_cache` semantics.
- Query resolved provider and mount paths through `qliber::config_snapshot()` or by using
  `qliber::with_data_path` to inspect file-system mappings.

## Development

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

The integration test exercises the full data pipeline using temporary CSV inputs to ensure end-to-end correctness.

## Large Language Model integration

`qliber::llm` provides optional tooling to run natural-language models alongside the
quantitative pipeline:

- `OllamaClient` issues blocking requests to a local Ollama runtime, enabling reuse of
  chat or instruct models already managed by Ollama.
- `GgufRunner` (guarded behind the `gguf` Cargo feature) loads GGUF checkpoints directly
  through [`llama.cpp`](https://github.com/ggerganov/llama.cpp) bindings for fully
  offline inference.

```rust
use qliber::{GenerationOptions, OllamaClient};

fn main() -> anyhow::Result<()> {
    let client = OllamaClient::new("http://localhost:11434", "phi3")?;
    let mut options = GenerationOptions::default();
    options.system_prompt = Some("You are a helpful quant assistant.".into());
    options.max_tokens = Some(128);
    let reply = client.generate_with_options("Summarize today's factor exposures.", &options)?;
    println!("Ollama reply: {reply}");
    Ok(())
}
```

To enable direct GGUF execution, compile with `--features gguf` and provide a GGUF
checkpoint path when constructing `GgufRunner`.

## Build script

The repository ships a helper script for reproducible release builds:

```bash
./scripts/build.sh             # Release build with default features
./scripts/build.sh gguf        # Release build enabling the gguf feature
```

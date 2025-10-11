# Python vs. Rust Qlib Parity

This document tracks the remaining functional differences between the original
Python implementation (`qlib/`) and the Rust port (`qliber/`).

## Covered surfaces

- **Data ingestion and provider stack**: `qliber` re-implements dataset loaders,
  feature processors, and provider infrastructure to mirror `qlib.data` modules.
- **Backtesting and workflow orchestration**: strategy abstractions, execution
  loops, and experiment tracking mirror the Python `backtest` and `workflow`
  packages.
- **Portfolio analytics and evaluation metrics**: cumulative returns, drawdown,
  indicator statistics, and Sharpe/alpha utilities are available in both code
  bases.
- **Trainer registry & interpretation**: the Rust trainer mirrors Python's
  adapter registry with pluggable backends and now includes permutation-based
  feature interpretation utilities for model explainability.
- **CLI tooling**: the `qrun` binary provides end-to-end orchestration similar to
  the Python CLI entrypoints.

## Outstanding functionality

1. **Model zoo parity** – The Python stack includes a wide collection of machine
   learning models (e.g., gradient boosting, meta-learning ensembles, and risk
   models) under `qlib/model`. `qliber` now ships both the baseline `MeanModel`
   and an `XgBoostModel` backed by SmartCore's gradient boosting implementation,
   covering the LightGBM-style workflows. Remaining work focuses on ensemble
   operators and meta-learning helpers defined in `qlib/model/ens` and
   `qlib/model/meta`.
2. **Risk model generation** – The risk modeling utilities (`qlib/model/riskmodel`)
   for factor covariance estimation remain Python-only. Rust side exposes the
   downstream analytics but not the model fitting workflow.

## Next steps

- Recreate the ensemble/grouping abstractions for production parity with
  `qlib.model.ens`.

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

The current Rust surface matches the Python implementation across the original
parity checklist. Core areas now aligned include:

- **Ensemble & meta-learning** – `WeightedEnsemble`, `MetaLabelGenerator`, and
  the built-in trainer adapter mirror `qlib.model.ens` and `qlib.model.meta`
  behaviors, including learned weight blending and meta-label generation.
- **Risk models** – `FactorRiskModel` provides shrinkage-aware covariance
  estimation, `PoetRiskModel` mirrors the Principal Orthogonal Complement
  Thresholding estimator, and `StructuredRiskModel` reconstructs
  PCA/factor-analysis driven covariance surfaces equivalent to the Python
  `qlib.model.riskmodel` implementations.

## Next steps

- Continue expanding the model zoo with additional learners from
  `qlib.model` (e.g., ensemble stacking variants, riskmodel utilities beyond
  shrinkage estimators) as new research needs emerge.

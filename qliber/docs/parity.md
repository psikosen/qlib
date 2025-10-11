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
- **CLI tooling**: the `qrun` binary provides end-to-end orchestration similar to
  the Python CLI entrypoints.

## Outstanding functionality

1. **Model zoo parity** – The Python stack includes a wide collection of machine
   learning models (e.g., gradient boosting, meta-learning ensembles, and risk
   models) under `qlib/model`. The Rust crate currently exposes a baseline
   `MeanModel` to prove out the training pipeline and lacks equivalents for the
   richer algorithms, particularly the ensemble operators and meta-learning
   helpers defined in `qlib/model/ens` and `qlib/model/meta`.
2. **Advanced interpretation utilities** – Python provides interpretation and
   explainability helpers (`qlib/model/interpret`) that are not yet mirrored in
   Rust.
3. **Risk model generation** – The risk modeling utilities (`qlib/model/riskmodel`)
   for factor covariance estimation remain Python-only. Rust side exposes the
   downstream analytics but not the model fitting workflow.
4. **Extensible trainer registry** – Python's trainer module integrates with
   third-party frameworks (LightGBM, PyTorch). Rust currently implements the
   trainer orchestration but only ships an in-memory baseline and lacks bindings
   to external ML libraries.

## Next steps

- Port a representative gradient boosting model (e.g., LightGBM) to Rust or
  integrate via FFI.
- Recreate the ensemble/grouping abstractions for production parity with
  `qlib.model.ens`.
- Extend the trainer module with plugin support mirroring the Python
  configuration surface.

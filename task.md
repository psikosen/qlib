# Task List

- [x] Survey repository structure and outstanding TODO markers.
- [x] Identify failing tests caused by missing package path configuration.
- [x] Update test configuration to ensure the qlib package is importable without installation.
- [x] Re-run affected tests to confirm success.
- [x] Summarize changes and testing results.
- [x] Harden PIT data collector against missing optional dependencies.
- [x] Re-run full pytest suite after dependency guard changes (fails due to missing offline dataset assets).
- [x] Audit qliber parity for LLM/gguf requirements and record gaps.
- [x] Add Rust-side build orchestration script for reproducible releases.
- [x] Extend the trainer registry with pluggable adapters for external ML frameworks.
- [x] Port model interpretation utilities (`qlib/model/interpret`) into qliber.
- [x] Port ensemble/meta-learning helpers (`qlib/model/ens` & `qlib/model/meta`) into qliber.
- [x] Implement factor risk model generation with shrinkage parity to `qlib.model.riskmodel`.
- [x] Rerun targeted ensemble and risk model integration tests after adjustments.
- [x] Execute full `cargo test` to validate the expanded module surface.

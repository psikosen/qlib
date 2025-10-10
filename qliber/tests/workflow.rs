use polars::prelude::*;
use serde_json::json;

use qliber::workflow::{ExperimentManager, TaskManager};
use qliber::{MeanModel, Trainer};

#[test]
fn workflow_records_metrics() -> anyhow::Result<()> {
    let manager = ExperimentManager::new();
    let recorder = manager.start("test-experiment");
    let mut tasks = TaskManager::new();
    tasks.register(|recorder| recorder.log_param("alpha", json!(0.1)));
    tasks.run(&recorder);

    let mut model = MeanModel::new("label");
    let mut trainer = Trainer::new(recorder.clone(), &mut model, "label", 2);
    let features = DataFrame::new(vec![Series::new("feature", vec![1.0, 2.0, 3.0])])?;
    let labels = DataFrame::new(vec![Series::new("label", vec![1.0, 1.5, 2.0])])?;
    trainer.train(&features, &labels)?;

    let snapshot = recorder.snapshot();
    assert!(snapshot.metrics.contains_key("epoch_0_mse"));
    assert!(snapshot.parameters.contains_key("alpha"));
    Ok(())
}

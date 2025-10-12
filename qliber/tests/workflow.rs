use polars::prelude::*;
use serde_json::json;

use std::sync::{Arc, Mutex};

use qliber::workflow::{ExperimentManager, TaskManager, TaskStatus};
use qliber::{MeanModel, Trainer, WorkflowError};

#[test]
fn workflow_records_metrics() -> anyhow::Result<()> {
    let manager = ExperimentManager::new();
    let recorder = manager.start("test-experiment");
    let mut tasks = TaskManager::new();
    let _ = tasks.register(|recorder| recorder.log_param("alpha", json!(0.1)));
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

#[test]
fn task_manager_tracks_status_transitions() {
    let manager = ExperimentManager::new();
    let recorder = manager.start("task-tracker");
    let tasks = TaskManager::new();

    let _always_ok = tasks.enqueue("log", |recorder| {
        recorder.log_metric("ok", 1.0);
        Ok(())
    });

    let toggle = Arc::new(Mutex::new(true));
    let toggle_task = Arc::clone(&toggle);
    let fail_id = tasks.enqueue("flaky", move |_recorder| {
        let mut flag = toggle_task.lock().expect("lock toggle");
        if *flag {
            *flag = false;
            Err(WorkflowError::Io("transient".into()))
        } else {
            Ok(())
        }
    });

    tasks.run(&recorder);

    let summaries = tasks.tasks();
    assert_eq!(summaries.len(), 2);
    let failed = tasks.by_status(TaskStatus::Failed);
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].id, fail_id);
    assert_eq!(failed[0].attempts, 1);

    let retried = tasks.retry_failed();
    assert_eq!(retried, 1);

    tasks.run(&recorder);

    let flaky = tasks.get(fail_id).expect("flaky task recorded");
    assert_eq!(flaky.status, TaskStatus::Completed);
    assert_eq!(flaky.attempts, 2);
    assert!(flaky.last_error.is_some());
}

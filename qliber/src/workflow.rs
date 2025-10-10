use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde_json::Value;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum WorkflowError {
    #[error("experiment `{0}` not found")]
    ExperimentMissing(String),
    #[error("io error: {0}")]
    Io(String),
}

pub type WorkflowResult<T> = Result<T, WorkflowError>;

#[derive(Debug, Clone, Default)]
pub struct ExperimentRecord {
    pub name: String,
    pub parameters: HashMap<String, Value>,
    pub metrics: HashMap<String, f64>,
    pub artifacts: HashMap<String, PathBuf>,
    pub created_at: chrono::DateTime<Utc>,
}

impl ExperimentRecord {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: HashMap::new(),
            metrics: HashMap::new(),
            artifacts: HashMap::new(),
            created_at: Utc::now(),
        }
    }
}

#[derive(Clone)]
pub struct Recorder {
    inner: Arc<Mutex<ExperimentRecord>>,
}

impl Recorder {
    pub fn new(record: ExperimentRecord) -> Self {
        Self {
            inner: Arc::new(Mutex::new(record)),
        }
    }

    pub fn log_param(&self, key: impl Into<String>, value: Value) {
        if let Ok(mut record) = self.inner.lock() {
            record.parameters.insert(key.into(), value);
        }
    }

    pub fn log_metric(&self, key: impl Into<String>, value: f64) {
        if let Ok(mut record) = self.inner.lock() {
            record.metrics.insert(key.into(), value);
        }
    }

    pub fn log_artifact(&self, key: impl Into<String>, path: PathBuf) {
        if let Ok(mut record) = self.inner.lock() {
            record.artifacts.insert(key.into(), path);
        }
    }

    pub fn snapshot(&self) -> ExperimentRecord {
        self.inner
            .lock()
            .map(|record| record.clone())
            .unwrap_or_default()
    }
}

#[derive(Default)]
pub struct ExperimentManager {
    experiments: Mutex<HashMap<String, Recorder>>,
}

impl ExperimentManager {
    pub fn new() -> Self {
        Self {
            experiments: Mutex::new(HashMap::new()),
        }
    }

    pub fn start(&self, name: impl Into<String>) -> Recorder {
        let name = name.into();
        let record = Recorder::new(ExperimentRecord::new(&name));
        self.experiments
            .lock()
            .expect("lock experiments")
            .insert(name.clone(), record.clone());
        log_event(
            file!(),
            "ExperimentManager",
            "start",
            "workflow.experiment",
            line!(),
            &format!("Started experiment {name}"),
            None,
            "post",
            "POST",
        );
        record
    }

    pub fn get(&self, name: &str) -> WorkflowResult<Recorder> {
        self.experiments
            .lock()
            .expect("lock experiments")
            .get(name)
            .cloned()
            .ok_or_else(|| WorkflowError::ExperimentMissing(name.to_string()))
    }

    pub fn list(&self) -> Vec<String> {
        self.experiments
            .lock()
            .expect("lock experiments")
            .keys()
            .cloned()
            .collect()
    }
}

type TaskCallback = dyn Fn(&Recorder) + Send + Sync;

#[derive(Default)]
pub struct TaskManager {
    tasks: Vec<Box<TaskCallback>>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(&mut self, task: F)
    where
        F: Fn(&Recorder) + Send + Sync + 'static,
    {
        self.tasks.push(Box::new(task));
    }

    pub fn run(&self, recorder: &Recorder) {
        for task in &self.tasks {
            task(recorder);
        }
    }
}

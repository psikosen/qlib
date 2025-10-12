use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
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

type TaskFn = dyn Fn(&Recorder) -> WorkflowResult<()> + Send + Sync;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskStatus {
    Waiting,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSummary {
    pub id: u64,
    pub name: String,
    pub status: TaskStatus,
    pub attempts: u32,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

struct TaskEntry {
    id: u64,
    name: String,
    status: TaskStatus,
    attempts: u32,
    last_error: Option<String>,
    callback: Arc<TaskFn>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl TaskEntry {
    fn new(id: u64, name: String, callback: Arc<TaskFn>) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            status: TaskStatus::Waiting,
            attempts: 0,
            last_error: None,
            callback,
            created_at: now,
            updated_at: now,
        }
    }

    fn summarize(&self) -> TaskSummary {
        TaskSummary {
            id: self.id,
            name: self.name.clone(),
            status: self.status,
            attempts: self.attempts,
            last_error: self.last_error.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

#[derive(Default)]
struct TaskStore {
    tasks: HashMap<u64, TaskEntry>,
    queue: VecDeque<u64>,
    next_id: u64,
}

#[derive(Clone, Default)]
pub struct TaskManager {
    inner: Arc<Mutex<TaskStore>>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TaskStore {
                tasks: HashMap::new(),
                queue: VecDeque::new(),
                next_id: 1,
            })),
        }
    }

    fn enqueue_internal<F>(&self, name: Option<String>, task: F) -> u64
    where
        F: Fn(&Recorder) -> WorkflowResult<()> + Send + Sync + 'static,
    {
        let mut store = self.inner.lock().expect("lock task store");
        let id = store.next_id;
        store.next_id += 1;
        let task_name = name.unwrap_or_else(|| format!("task-{id}"));
        let entry = TaskEntry::new(id, task_name.clone(), Arc::new(task));
        store.queue.push_back(id);
        store.tasks.insert(id, entry);
        drop(store);

        log_event(
            file!(),
            "TaskManager",
            "enqueue",
            "workflow.task",
            line!(),
            &format!("Enqueued task `{task_name}` with id {id}"),
            None,
            "none",
            "POST",
        );

        id
    }

    pub fn register<F>(&mut self, task: F) -> u64
    where
        F: Fn(&Recorder) + Send + Sync + 'static,
    {
        let task_arc = move |recorder: &Recorder| {
            task(recorder);
            Ok(())
        };
        self.enqueue_internal(None, task_arc)
    }

    pub fn enqueue<F>(&self, name: impl Into<String>, task: F) -> u64
    where
        F: Fn(&Recorder) -> WorkflowResult<()> + Send + Sync + 'static,
    {
        self.enqueue_internal(Some(name.into()), task)
    }

    fn claim_next(&self) -> Option<(u64, Arc<TaskFn>, String)> {
        let mut store = self.inner.lock().expect("lock task store");
        let id = store.queue.pop_front()?;
        if let Some(entry) = store.tasks.get_mut(&id) {
            entry.status = TaskStatus::Running;
            entry.attempts += 1;
            entry.updated_at = Utc::now();
            let name = entry.name.clone();
            let callback = Arc::clone(&entry.callback);
            drop(store);

            log_event(
                file!(),
                "TaskManager",
                "claim",
                "workflow.task",
                line!(),
                &format!("Claimed task `{name}` (id {id}) for execution"),
                None,
                "none",
                "GET",
            );

            Some((id, callback, name))
        } else {
            None
        }
    }

    fn mark_completed(&self, id: u64, name: &str) {
        if let Ok(mut store) = self.inner.lock() {
            if let Some(entry) = store.tasks.get_mut(&id) {
                entry.status = TaskStatus::Completed;
                entry.updated_at = Utc::now();
            }
        }

        log_event(
            file!(),
            "TaskManager",
            "complete",
            "workflow.task",
            line!(),
            &format!("Completed task `{name}` (id {id})"),
            None,
            "none",
            "PUT",
        );
    }

    fn mark_failed_internal(&self, id: u64, name: &str, error: String) {
        if let Ok(mut store) = self.inner.lock() {
            if let Some(entry) = store.tasks.get_mut(&id) {
                entry.status = TaskStatus::Failed;
                entry.last_error = Some(error.clone());
                entry.updated_at = Utc::now();
            }
        }

        log_event(
            file!(),
            "TaskManager",
            "fail",
            "workflow.task",
            line!(),
            &format!("Task `{name}` (id {id}) failed"),
            Some(&error),
            "none",
            "PUT",
        );
    }

    pub fn run(&self, recorder: &Recorder) {
        while let Some((id, callback, name)) = self.claim_next() {
            match callback(recorder) {
                Ok(()) => self.mark_completed(id, &name),
                Err(error) => self.mark_failed_internal(id, &name, error.to_string()),
            }
        }
    }

    pub fn requeue(&self, id: u64) -> bool {
        if let Ok(mut store) = self.inner.lock() {
            let task_name = store.tasks.get_mut(&id).map(|entry| {
                entry.status = TaskStatus::Waiting;
                entry.updated_at = Utc::now();
                entry.name.clone()
            });

            if let Some(name) = task_name {
                store.queue.push_back(id);
                drop(store);
                log_event(
                    file!(),
                    "TaskManager",
                    "requeue",
                    "workflow.task",
                    line!(),
                    &format!("Requeued task `{name}` (id {id})"),
                    None,
                    "none",
                    "POST",
                );
                return true;
            }
        }
        false
    }

    pub fn retry_failed(&self) -> usize {
        let mut requeued = Vec::new();
        if let Ok(mut store) = self.inner.lock() {
            for entry in store.tasks.values_mut() {
                if entry.status == TaskStatus::Failed {
                    entry.status = TaskStatus::Waiting;
                    entry.updated_at = Utc::now();
                    requeued.push((entry.id, entry.name.clone()));
                }
            }

            for (id, _) in &requeued {
                store.queue.push_back(*id);
            }
        }

        for (id, name) in &requeued {
            log_event(
                file!(),
                "TaskManager",
                "retry_failed",
                "workflow.task",
                line!(),
                &format!("Requeued failed task `{name}` (id {id})"),
                None,
                "none",
                "POST",
            );
        }

        requeued.len()
    }

    pub fn tasks(&self) -> Vec<TaskSummary> {
        self.inner
            .lock()
            .expect("lock task store")
            .tasks
            .values()
            .map(TaskEntry::summarize)
            .collect()
    }

    pub fn by_status(&self, status: TaskStatus) -> Vec<TaskSummary> {
        self.tasks()
            .into_iter()
            .filter(|task| task.status == status)
            .collect()
    }

    pub fn get(&self, id: u64) -> Option<TaskSummary> {
        self.inner
            .lock()
            .expect("lock task store")
            .tasks
            .get(&id)
            .map(TaskEntry::summarize)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("lock task store").tasks.len()
    }
}

use std::sync::OnceLock;

use anyhow::anyhow;
use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt};

use crate::Result;

static SUBSCRIBER: OnceLock<std::result::Result<(), String>> = OnceLock::new();

const SHERLOCK_PROMPT: &str = "[Continuous skepticism (Sherlock Protocol)] Could this change affect unexpected files/systems? | Any hidden dependencies or cascades? | What edge cases and failure modes are unhandled? | If stuck, work backward from the desired outcome.";

#[derive(Debug, Serialize)]
pub struct LogEvent<'a> {
    pub filename: &'a str,
    pub timestamp: DateTime<Utc>,
    pub classname: &'a str,
    pub function: &'a str,
    pub system_section: &'a str,
    pub line_num: u32,
    pub error: Option<&'a str>,
    pub db_phase: &'a str,
    pub method: &'a str,
    pub message: &'a str,
    pub derived: &'a str,
}

/// Initialize tracing subscriber emitting JSON records that follow the required schema.
///
/// Calling this function multiple times is safe; only the first invocation installs the
/// subscriber.
pub fn init_logging() -> Result<()> {
    let result = SUBSCRIBER.get_or_init(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        fmt()
            .with_env_filter(filter)
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .with_timer(fmt::time::UtcTime::rfc_3339())
            .with_target(false)
            .try_init()
            .map_err(|error| error.to_string())?;

        Ok(())
    });

    match result {
        Ok(()) => Ok(()),
        Err(message) => Err(anyhow!(message.clone())),
    }
}

/// Emit a structured log event conforming to the canonical schema alongside the
/// "Continuous skepticism" derived line required by the project guidelines.
#[allow(clippy::too_many_arguments)]
pub fn log_event(
    filename: &str,
    classname: &str,
    function: &str,
    system_section: &str,
    line_num: u32,
    message: &str,
    error: Option<&str>,
    db_phase: &str,
    method: &str,
) {
    let event = LogEvent {
        filename,
        timestamp: Utc::now(),
        classname,
        function,
        system_section,
        line_num,
        error,
        db_phase,
        method,
        message,
        derived: SHERLOCK_PROMPT,
    };

    if let Ok(serialized) = serde_json::to_string(&event) {
        info!(target: "qliber", json = %serialized, derived = SHERLOCK_PROMPT);
    } else {
        info!(target: "qliber", message, derived = SHERLOCK_PROMPT);
    }
}

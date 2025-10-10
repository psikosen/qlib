use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, NaiveDate, Utc};
use polars::io::SerReader;
use polars::io::json::JsonReader;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::provider::Instrument;

#[derive(Debug, Error)]
pub enum RemoteError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
}

pub type RemoteResult<T> = Result<T, RemoteError>;

pub trait RemoteTransport: Send + Sync {
    fn request(&self, event: &str, payload: &Value) -> RemoteResult<Value>;
}

pub struct RemoteDataClient<T: RemoteTransport> {
    transport: Arc<T>,
}

impl<T: RemoteTransport> RemoteDataClient<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    pub fn fetch_calendar(&self) -> RemoteResult<Vec<DateTime<Utc>>> {
        let response = self.transport.request("calendar.sessions", &Value::Null)?;
        let sessions: Vec<String> = serde_json::from_value(response)?;
        let mut parsed = Vec::with_capacity(sessions.len());
        for session in sessions {
            let dt = session
                .parse::<DateTime<Utc>>()
                .map_err(|err| RemoteError::Protocol(err.to_string()))?;
            parsed.push(dt);
        }
        Ok(parsed)
    }

    pub fn fetch_instruments(&self) -> RemoteResult<Vec<Instrument>> {
        let response = self.transport.request("instruments.list", &Value::Null)?;
        let records: Vec<InstrumentPayload> = serde_json::from_value(response)?;
        Ok(records.into_iter().map(Instrument::from).collect())
    }

    pub fn fetch_feature(
        &self,
        instrument: &str,
        feature: &str,
        as_of: DateTime<Utc>,
    ) -> RemoteResult<DataFrame> {
        let payload = json!({
            "instrument": instrument,
            "feature": feature,
            "as_of": as_of.to_rfc3339(),
        });
        let response = self.transport.request("feature.load", &payload)?;
        frame_from_value(&response)
    }

    pub fn fetch_pit_snapshot(
        &self,
        instrument: &str,
        as_of: DateTime<Utc>,
    ) -> RemoteResult<Option<DataFrame>> {
        let payload = json!({
            "instrument": instrument,
            "as_of": as_of.to_rfc3339(),
        });
        let response = self.transport.request("pit.snapshot", &payload)?;
        if response.is_null() {
            return Ok(None);
        }
        frame_from_value(&response).map(Some)
    }

    pub fn fetch_pit_range(
        &self,
        instrument: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> RemoteResult<Vec<DataFrame>> {
        let payload = json!({
            "instrument": instrument,
            "start": start.to_rfc3339(),
            "end": end.to_rfc3339(),
        });
        let response = self.transport.request("pit.range", &payload)?;
        let frames: Vec<Value> = serde_json::from_value(response)?;
        frames
            .into_iter()
            .map(|value| frame_from_value(&value))
            .collect()
    }
}

fn frame_from_value(value: &Value) -> RemoteResult<DataFrame> {
    let bytes = serde_json::to_vec(value)?;
    let cursor = Cursor::new(bytes);
    let frame = JsonReader::new(cursor).finish()?;
    Ok(frame)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstrumentPayload {
    symbol: String,
    start: String,
    end: Option<String>,
    #[serde(default)]
    metadata: serde_json::Map<String, Value>,
}

impl From<InstrumentPayload> for Instrument {
    fn from(value: InstrumentPayload) -> Self {
        let start = NaiveDate::parse_from_str(&value.start, "%Y-%m-%d")
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
        let end = value
            .end
            .and_then(|end| NaiveDate::parse_from_str(&end, "%Y-%m-%d").ok());
        let mut instrument = Instrument::new(value.symbol, start, end);
        for (key, value) in value.metadata {
            if let Some(text) = value.as_str() {
                instrument = instrument.with_metadata(key, text.to_string());
            }
        }
        instrument
    }
}

pub struct MockTransport {
    responses: Mutex<VecDeque<RemoteResult<Value>>>,
    events: Mutex<Vec<String>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn push_response(&self, response: RemoteResult<Value>) {
        self.responses
            .lock()
            .expect("lock queue")
            .push_back(response);
    }

    pub fn events(&self) -> Vec<String> {
        self.events.lock().expect("lock events").clone()
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteTransport for MockTransport {
    fn request(&self, event: &str, _: &Value) -> RemoteResult<Value> {
        self.events
            .lock()
            .expect("lock events")
            .push(event.to_string());
        let mut queue = self.responses.lock().expect("lock queue");
        queue
            .pop_front()
            .unwrap_or_else(|| Err(RemoteError::Protocol("no queued response".into())))
    }
}

impl From<std::sync::PoisonError<std::sync::MutexGuard<'_, VecDeque<RemoteResult<Value>>>>>
    for RemoteError
{
    fn from(
        err: std::sync::PoisonError<std::sync::MutexGuard<'_, VecDeque<RemoteResult<Value>>>>,
    ) -> Self {
        RemoteError::Transport(err.to_string())
    }
}

impl From<std::sync::PoisonError<std::sync::MutexGuard<'_, Vec<String>>>> for RemoteError {
    fn from(err: std::sync::PoisonError<std::sync::MutexGuard<'_, Vec<String>>>) -> Self {
        RemoteError::Transport(err.to_string())
    }
}

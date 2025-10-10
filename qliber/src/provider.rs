use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, NaiveDate, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::logging::log_event;

type PitMap = HashMap<String, BTreeMap<DateTime<Utc>, DataFrame>>;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("calendar operation failed: {0}")]
    Calendar(String),
    #[error("instrument error: {0}")]
    Instrument(String),
    #[error("feature backend error: {0}")]
    FeatureBackend(String),
    #[error("pit storage error: {0}")]
    Pit(String),
}

pub type ProviderResult<T> = Result<T, ProviderError>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Instrument {
    pub symbol: String,
    pub start: NaiveDate,
    pub end: Option<NaiveDate>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Instrument {
    pub fn new<S: Into<String>>(symbol: S, start: NaiveDate, end: Option<NaiveDate>) -> Self {
        Self {
            symbol: symbol.into(),
            start,
            end,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Clone, Default)]
pub struct InstrumentStore {
    inner: Arc<RwLock<HashMap<String, Instrument>>>,
}

impl InstrumentStore {
    pub fn register(&self, instrument: Instrument) -> ProviderResult<()> {
        let symbol = instrument.symbol.clone();
        let mut guard = self
            .inner
            .write()
            .map_err(|_| ProviderError::Instrument("lock poisoned".into()))?;
        guard.insert(symbol.clone(), instrument);
        log_event(
            file!(),
            "InstrumentStore",
            "register",
            "provider.instrument",
            line!(),
            &format!("Registered instrument {symbol}"),
            None,
            "none",
            "POST",
        );
        Ok(())
    }

    pub fn get(&self, symbol: &str) -> ProviderResult<Option<Instrument>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| ProviderError::Instrument("lock poisoned".into()))?;
        Ok(guard.get(symbol).cloned())
    }

    pub fn all(&self) -> ProviderResult<Vec<Instrument>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| ProviderError::Instrument("lock poisoned".into()))?;
        Ok(guard.values().cloned().collect())
    }
}

#[derive(Clone)]
pub struct TradingCalendar {
    sessions: Arc<RwLock<Vec<NaiveDate>>>,
}

impl TradingCalendar {
    pub fn new(mut sessions: Vec<NaiveDate>) -> Self {
        sessions.sort();
        Self {
            sessions: Arc::new(RwLock::new(sessions)),
        }
    }

    pub fn contains(&self, date: &NaiveDate) -> ProviderResult<bool> {
        let guard = self
            .sessions
            .read()
            .map_err(|_| ProviderError::Calendar("lock poisoned".into()))?;
        Ok(guard.binary_search(date).is_ok())
    }

    pub fn next(&self, date: &NaiveDate) -> ProviderResult<Option<NaiveDate>> {
        let guard = self
            .sessions
            .read()
            .map_err(|_| ProviderError::Calendar("lock poisoned".into()))?;
        match guard.binary_search(date) {
            Ok(idx) if idx + 1 < guard.len() => Ok(Some(guard[idx + 1])),
            Ok(_) | Err(_) => guard
                .iter()
                .find(|session| *session > date)
                .copied()
                .map_or(Ok(None), |d| Ok(Some(d))),
        }
    }

    pub fn previous(&self, date: &NaiveDate) -> ProviderResult<Option<NaiveDate>> {
        let guard = self
            .sessions
            .read()
            .map_err(|_| ProviderError::Calendar("lock poisoned".into()))?;
        match guard.binary_search(date) {
            Ok(idx) if idx > 0 => Ok(Some(guard[idx - 1])),
            Ok(_) | Err(_) => guard
                .iter()
                .rev()
                .find(|session| *session < date)
                .copied()
                .map_or(Ok(None), |d| Ok(Some(d))),
        }
    }

    pub fn sessions(&self) -> ProviderResult<Vec<NaiveDate>> {
        let guard = self
            .sessions
            .read()
            .map_err(|_| ProviderError::Calendar("lock poisoned".into()))?;
        Ok(guard.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq)]
pub struct FeatureKey {
    pub instrument: String,
    pub feature: String,
    pub as_of: DateTime<Utc>,
    pub horizon: Option<String>,
}

impl PartialEq for FeatureKey {
    fn eq(&self, other: &Self) -> bool {
        self.instrument == other.instrument
            && self.feature == other.feature
            && self.as_of.timestamp_nanos_opt() == other.as_of.timestamp_nanos_opt()
            && self.horizon == other.horizon
    }
}

impl Hash for FeatureKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.instrument.hash(state);
        self.feature.hash(state);
        self.as_of
            .timestamp_nanos_opt()
            .unwrap_or_default()
            .hash(state);
        self.horizon.hash(state);
    }
}

impl FeatureKey {
    pub fn new<S: Into<String>>(instrument: S, feature: S, as_of: DateTime<Utc>) -> Self {
        Self {
            instrument: instrument.into(),
            feature: feature.into(),
            as_of,
            horizon: None,
        }
    }

    pub fn with_horizon(mut self, horizon: impl Into<String>) -> Self {
        self.horizon = Some(horizon.into());
        self
    }
}

pub trait FeatureBackend: Send + Sync {
    fn get(&self, key: &FeatureKey) -> ProviderResult<Option<DataFrame>>;
    fn set(&self, key: FeatureKey, frame: DataFrame) -> ProviderResult<()>;
    fn invalidate(&self, key: &FeatureKey) -> ProviderResult<()>;
}

#[derive(Default)]
pub struct InMemoryFeatureBackend {
    store: RwLock<HashMap<FeatureKey, DataFrame>>,
}

impl InMemoryFeatureBackend {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

impl FeatureBackend for InMemoryFeatureBackend {
    fn get(&self, key: &FeatureKey) -> ProviderResult<Option<DataFrame>> {
        let guard = self
            .store
            .read()
            .map_err(|_| ProviderError::FeatureBackend("lock poisoned".into()))?;
        Ok(guard.get(key).cloned())
    }

    fn set(&self, key: FeatureKey, frame: DataFrame) -> ProviderResult<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|_| ProviderError::FeatureBackend("lock poisoned".into()))?;
        guard.insert(key.clone(), frame);
        log_event(
            file!(),
            "InMemoryFeatureBackend",
            "set",
            "provider.feature",
            line!(),
            &format!("Stored feature {} for {}", key.feature, key.instrument),
            None,
            "post",
            "PUT",
        );
        Ok(())
    }

    fn invalidate(&self, key: &FeatureKey) -> ProviderResult<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|_| ProviderError::FeatureBackend("lock poisoned".into()))?;
        guard.remove(key);
        log_event(
            file!(),
            "InMemoryFeatureBackend",
            "invalidate",
            "provider.feature",
            line!(),
            &format!("Invalidated feature {} for {}", key.feature, key.instrument),
            None,
            "post",
            "DELETE",
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct PitRecord {
    pub instrument: String,
    pub timestamp: DateTime<Utc>,
    pub payload: DataFrame,
}

impl PitRecord {
    pub fn new(
        instrument: impl Into<String>,
        timestamp: DateTime<Utc>,
        payload: DataFrame,
    ) -> Self {
        Self {
            instrument: instrument.into(),
            timestamp,
            payload,
        }
    }
}

#[derive(Default, Clone)]
pub struct PitStore {
    records: Arc<RwLock<PitMap>>,
}

impl PitStore {
    pub fn insert(&self, record: PitRecord) -> ProviderResult<()> {
        let mut guard = self
            .records
            .write()
            .map_err(|_| ProviderError::Pit("lock poisoned".into()))?;
        let entry = guard.entry(record.instrument.clone()).or_default();
        entry.insert(record.timestamp, record.payload.clone());
        log_event(
            file!(),
            "PitStore",
            "insert",
            "provider.pit",
            line!(),
            &format!(
                "Inserted PIT record for {} at {}",
                record.instrument, record.timestamp
            ),
            None,
            "post",
            "PUT",
        );
        Ok(())
    }

    pub fn snapshot(
        &self,
        instrument: &str,
        as_of: DateTime<Utc>,
    ) -> ProviderResult<Option<DataFrame>> {
        let guard = self
            .records
            .read()
            .map_err(|_| ProviderError::Pit("lock poisoned".into()))?;
        let Some(tree) = guard.get(instrument) else {
            return Ok(None);
        };
        let mut iter = tree.range(..=as_of);
        Ok(iter.next_back().map(|(_, frame)| frame.clone()))
    }

    pub fn range(
        &self,
        instrument: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ProviderResult<Vec<PitRecord>> {
        if end < start {
            return Err(ProviderError::Pit("end before start".into()));
        }
        let guard = self
            .records
            .read()
            .map_err(|_| ProviderError::Pit("lock poisoned".into()))?;
        let Some(tree) = guard.get(instrument) else {
            return Ok(vec![]);
        };
        Ok(tree
            .range(start..=end)
            .map(|(timestamp, frame)| PitRecord {
                instrument: instrument.to_string(),
                timestamp: *timestamp,
                payload: frame.clone(),
            })
            .collect())
    }
}

#[derive(Clone)]
pub struct DataProvider<B: FeatureBackend + 'static> {
    calendar: TradingCalendar,
    instruments: InstrumentStore,
    feature_backend: Arc<B>,
    pit_store: PitStore,
}

impl<B: FeatureBackend + 'static> DataProvider<B> {
    pub fn new(calendar: TradingCalendar, backend: Arc<B>) -> Self {
        Self {
            calendar,
            instruments: InstrumentStore::default(),
            feature_backend: backend,
            pit_store: PitStore::default(),
        }
    }

    pub fn with_components(
        calendar: TradingCalendar,
        instruments: InstrumentStore,
        backend: Arc<B>,
        pit_store: PitStore,
    ) -> Self {
        Self {
            calendar,
            instruments,
            feature_backend: backend,
            pit_store,
        }
    }

    pub fn calendar(&self) -> &TradingCalendar {
        &self.calendar
    }

    pub fn instruments(&self) -> &InstrumentStore {
        &self.instruments
    }

    pub fn pit_store(&self) -> &PitStore {
        &self.pit_store
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.feature_backend
    }

    pub fn store_instrument(&self, instrument: Instrument) -> ProviderResult<()> {
        self.instruments.register(instrument)
    }

    pub fn load_feature(&self, key: &FeatureKey) -> ProviderResult<Option<DataFrame>> {
        self.feature_backend.get(key)
    }

    pub fn store_feature(&self, key: FeatureKey, frame: DataFrame) -> ProviderResult<()> {
        self.feature_backend.set(key, frame)
    }

    pub fn invalidate_feature(&self, key: &FeatureKey) -> ProviderResult<()> {
        self.feature_backend.invalidate(key)
    }

    pub fn store_pit(&self, record: PitRecord) -> ProviderResult<()> {
        self.pit_store.insert(record)
    }

    pub fn pit_snapshot(
        &self,
        instrument: &str,
        as_of: DateTime<Utc>,
    ) -> ProviderResult<Option<DataFrame>> {
        self.pit_store.snapshot(instrument, as_of)
    }
}

pub type DefaultDataProvider = DataProvider<InMemoryFeatureBackend>;

impl DefaultDataProvider {
    pub fn with_calendar(calendar: TradingCalendar) -> Self {
        Self::new(calendar, Arc::new(InMemoryFeatureBackend::new()))
    }
}

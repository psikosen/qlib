use std::sync::{Arc, Once};

use chrono::{NaiveDate, TimeZone, Utc};
use polars::prelude::*;
use qliber::{
    DataProvider, FeatureBackend, FeatureKey, InMemoryFeatureBackend, Instrument, ProviderResult,
    TradingCalendar,
};

static INIT: Once = Once::new();

fn init_logging() {
    INIT.call_once(|| {
        let _ = qliber::logging::init_logging();
    });
}

fn sample_calendar() -> TradingCalendar {
    TradingCalendar::new(vec![
        NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
        NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
    ])
}

fn sample_frame() -> DataFrame {
    df!(
        "value" => &[1.0, 2.0, 3.0],
        "timestamp" => &[
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
        ]
    )
    .expect("frame construction")
}

#[test]
fn calendar_navigation() -> ProviderResult<()> {
    init_logging();
    let calendar = sample_calendar();
    assert!(calendar.contains(&NaiveDate::from_ymd_opt(2024, 1, 3).unwrap())?);
    assert!(!calendar.contains(&NaiveDate::from_ymd_opt(2024, 1, 5).unwrap())?);

    assert_eq!(
        calendar.next(&NaiveDate::from_ymd_opt(2024, 1, 2).unwrap())?,
        Some(NaiveDate::from_ymd_opt(2024, 1, 3).unwrap())
    );
    assert_eq!(
        calendar.previous(&NaiveDate::from_ymd_opt(2024, 1, 3).unwrap())?,
        Some(NaiveDate::from_ymd_opt(2024, 1, 2).unwrap())
    );
    Ok(())
}

#[test]
fn instrument_store_register_and_get() -> ProviderResult<()> {
    init_logging();
    let store = qliber::InstrumentStore::default();
    let instrument = Instrument::new(
        "SH600000",
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
        None,
    )
    .with_metadata("exchange", "SSE");
    store.register(instrument.clone())?;
    let fetched = store.get("SH600000")?.expect("instrument present");
    assert_eq!(fetched.symbol, instrument.symbol);
    assert_eq!(fetched.metadata.get("exchange"), Some(&"SSE".to_string()));
    Ok(())
}

#[test]
fn feature_backend_roundtrip() -> ProviderResult<()> {
    init_logging();
    let backend = InMemoryFeatureBackend::new();
    let frame = sample_frame();
    let key = FeatureKey::new(
        "SH600000",
        "close",
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
    );
    backend.set(key.clone(), frame.clone())?;
    let loaded = backend.get(&key)?.expect("feature present");
    assert_eq!(loaded.shape(), frame.shape());
    backend.invalidate(&key)?;
    assert!(backend.get(&key)?.is_none());
    Ok(())
}

#[test]
fn pit_store_snapshot_range() -> ProviderResult<()> {
    init_logging();
    let store = qliber::PitStore::default();
    let frame = sample_frame();
    store.insert(qliber::PitRecord::new(
        "SH600000",
        Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        frame.clone(),
    ))?;
    store.insert(qliber::PitRecord::new(
        "SH600000",
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
        frame.clone(),
    ))?;
    let snapshot = store
        .snapshot(
            "SH600000",
            Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
        )?
        .expect("snapshot available");
    assert_eq!(snapshot.shape(), frame.shape());
    let range = store.range(
        "SH600000",
        Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
    )?;
    assert_eq!(range.len(), 2);
    Ok(())
}

#[test]
fn data_provider_end_to_end() -> ProviderResult<()> {
    init_logging();
    let calendar = sample_calendar();
    let backend = Arc::new(InMemoryFeatureBackend::new());
    let provider = DataProvider::new(calendar.clone(), backend);
    provider.store_instrument(Instrument::new(
        "SH600000",
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
        None,
    ))?;

    let key = FeatureKey::new(
        "SH600000",
        "close",
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
    );
    let frame = sample_frame();
    provider.store_feature(key.clone(), frame.clone())?;
    let fetched = provider
        .load_feature(&key)?
        .expect("feature present via provider");
    assert_eq!(fetched.shape(), frame.shape());

    provider.store_pit(qliber::PitRecord::new(
        "SH600000",
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
        frame.clone(),
    ))?;
    let pit = provider.pit_snapshot(
        "SH600000",
        Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
    )?;
    assert!(pit.is_some());

    assert!(
        provider
            .calendar()
            .contains(&NaiveDate::from_ymd_opt(2024, 1, 2).unwrap())?
    );

    Ok(())
}

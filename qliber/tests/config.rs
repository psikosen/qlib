use chrono::{NaiveDate, TimeZone, Utc};
use polars::prelude::*;
use tracing::Level;

use qliber::config::reset_for_tests;
use qliber::provider::{DefaultDataProvider, FeatureKey, TradingCalendar};
use qliber::{DefaultConfig, InitOptions, Region, config_snapshot, init};

#[test]
fn init_establishes_client_defaults() -> anyhow::Result<()> {
    reset_for_tests();
    init(DefaultConfig::Client, InitOptions::default())?;
    let snapshot = config_snapshot().expect("config snapshot available");
    assert_eq!(snapshot.mode, DefaultConfig::Client);
    assert_eq!(snapshot.region, Region::China);
    assert!(snapshot.registered);
    Ok(())
}

#[test]
fn init_respects_skip_if_registered() -> anyhow::Result<()> {
    reset_for_tests();
    init(DefaultConfig::Client, InitOptions::default())?;

    let options = InitOptions {
        skip_if_registered: true,
        logging_level: Some(Level::DEBUG),
        ..Default::default()
    };
    init(DefaultConfig::Server, options)?;

    let snapshot = config_snapshot().expect("config snapshot available");
    assert_eq!(snapshot.mode, DefaultConfig::Client);
    assert_eq!(snapshot.region, Region::China);
    Ok(())
}

#[test]
fn init_clears_registered_feature_caches() -> anyhow::Result<()> {
    reset_for_tests();
    let calendar = TradingCalendar::new(vec![NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()]);
    let provider = DefaultDataProvider::with_calendar(calendar);
    let timestamp = Utc.from_utc_datetime(
        &NaiveDate::from_ymd_opt(2024, 1, 2)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    );
    let key = FeatureKey::new("TEST", "close", timestamp);

    let frame = df! { "close" => &[1.0_f64] }?;
    provider.store_feature(key.clone(), frame)?;
    assert!(provider.load_feature(&key)?.is_some());

    init(DefaultConfig::Client, InitOptions::default())?;

    assert!(provider.load_feature(&key)?.is_none());

    Ok(())
}

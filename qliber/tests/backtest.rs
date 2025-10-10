use chrono::{TimeZone, Utc};

use qliber::backtest::{
    Backtester, MarketEvent, MarketSnapshot, OrderSignal, SimpleExecutor, StaticRLStrategy,
    ThresholdInterpreter,
};
use qliber::portfolio::PortfolioSnapshot;

struct BuyAndHoldStrategy;

impl qliber::Strategy for BuyAndHoldStrategy {
    fn on_snapshot(
        &mut self,
        snapshot: &MarketSnapshot,
        portfolio: &PortfolioSnapshot,
    ) -> qliber::BacktestResult<Vec<OrderSignal>> {
        if portfolio.holdings.is_empty() {
            Ok(snapshot
                .events()
                .iter()
                .map(|event| OrderSignal::new(&event.instrument, 1.0))
                .collect())
        } else {
            Ok(vec![])
        }
    }
}

#[test]
fn backtester_executes_strategy() -> anyhow::Result<()> {
    let events = vec![
        MarketSnapshot::new(vec![MarketEvent {
            timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
            instrument: "TEST".to_string(),
            price: 100.0,
            volume: 1000.0,
        }]),
        MarketSnapshot::new(vec![MarketEvent {
            timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
            instrument: "TEST".to_string(),
            price: 102.0,
            volume: 1000.0,
        }]),
    ];

    let mut backtester = Backtester::new(BuyAndHoldStrategy, SimpleExecutor::new(0.0), 1000.0);
    let report = backtester.run(events)?;
    assert_eq!(report.executions.len(), 1);
    assert_eq!(report.equity_curve.len(), 2);
    Ok(())
}

#[test]
fn rl_strategy_collects_rewards() -> anyhow::Result<()> {
    let interpreter = std::sync::Arc::new(ThresholdInterpreter::new("price", "TEST", 0.5, 1.0));
    let strategy = StaticRLStrategy::new(interpreter);
    let events = vec![
        MarketSnapshot::new(vec![MarketEvent {
            timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
            instrument: "TEST".to_string(),
            price: 1.0,
            volume: 10.0,
        }]),
        MarketSnapshot::new(vec![MarketEvent {
            timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
            instrument: "TEST".to_string(),
            price: 2.0,
            volume: 10.0,
        }]),
    ];

    let mut backtester = Backtester::new(strategy, SimpleExecutor::new(0.0), 1000.0);
    let report = backtester.run_with_rl(events)?;
    assert_eq!(report.executions.len(), 2);
    Ok(())
}

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;
use crate::portfolio::{Holding, PortfolioSnapshot};

#[derive(Debug, Error)]
pub enum BacktestError {
    #[error("executor error: {0}")]
    Executor(String),
    #[error("strategy error: {0}")]
    Strategy(String),
    #[error("data error: {0}")]
    Data(String),
}

pub type BacktestResult<T> = Result<T, BacktestError>;

#[derive(Debug, Clone)]
pub struct MarketEvent {
    pub timestamp: DateTime<Utc>,
    pub instrument: String,
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    events: Vec<MarketEvent>,
}

impl MarketSnapshot {
    pub fn new(events: Vec<MarketEvent>) -> Self {
        Self { events }
    }

    pub fn events(&self) -> &[MarketEvent] {
        &self.events
    }

    pub fn price_map(&self) -> HashMap<String, f64> {
        self.events
            .iter()
            .map(|event| (event.instrument.clone(), event.price))
            .collect()
    }

    pub fn timestamp(&self) -> Option<DateTime<Utc>> {
        self.events.iter().map(|event| event.timestamp).max()
    }
}

#[derive(Debug, Clone)]
pub struct OrderSignal {
    pub instrument: String,
    pub quantity: f64,
}

impl OrderSignal {
    pub fn new(instrument: impl Into<String>, quantity: f64) -> Self {
        Self {
            instrument: instrument.into(),
            quantity,
        }
    }
}

pub trait Strategy {
    fn on_snapshot(
        &mut self,
        snapshot: &MarketSnapshot,
        portfolio: &PortfolioSnapshot,
    ) -> BacktestResult<Vec<OrderSignal>>;
}

pub trait RLStrategy: Strategy {
    fn on_reward(&mut self, reward: f64, done: bool);
}

pub trait SignalInterpreter: Send + Sync {
    fn interpret(&self, features: &DataFrame) -> Vec<OrderSignal>;
}

pub struct ThresholdInterpreter {
    column: String,
    instrument: String,
    threshold: f64,
    quantity: f64,
}

impl ThresholdInterpreter {
    pub fn new(
        column: impl Into<String>,
        instrument: impl Into<String>,
        threshold: f64,
        quantity: f64,
    ) -> Self {
        Self {
            column: column.into(),
            instrument: instrument.into(),
            threshold,
            quantity,
        }
    }
}

impl SignalInterpreter for ThresholdInterpreter {
    fn interpret(&self, features: &DataFrame) -> Vec<OrderSignal> {
        let Ok(series) = features.column(&self.column) else {
            return vec![];
        };
        let Ok(chunked) = series.f64() else {
            return vec![];
        };
        chunked
            .into_iter()
            .flatten()
            .filter_map(|value| {
                if value > self.threshold {
                    Some(OrderSignal::new(&self.instrument, self.quantity))
                } else if value < -self.threshold {
                    Some(OrderSignal::new(&self.instrument, -self.quantity))
                } else {
                    None
                }
            })
            .collect()
    }
}

pub trait Executor {
    fn execute(
        &self,
        orders: &[OrderSignal],
        snapshot: &MarketSnapshot,
    ) -> BacktestResult<Vec<ExecutionReport>>;
}

#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub instrument: String,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
}

pub struct SimpleExecutor {
    slippage: f64,
}

impl SimpleExecutor {
    pub fn new(slippage: f64) -> Self {
        Self { slippage }
    }
}

impl Executor for SimpleExecutor {
    fn execute(
        &self,
        orders: &[OrderSignal],
        snapshot: &MarketSnapshot,
    ) -> BacktestResult<Vec<ExecutionReport>> {
        let mut executions = Vec::new();
        let prices = snapshot.price_map();
        let timestamp = snapshot.timestamp().unwrap_or_else(Utc::now);

        for order in orders {
            if let Some(price) = prices.get(&order.instrument) {
                let adjusted_price = price * (1.0 + self.slippage * order.quantity.signum());
                executions.push(ExecutionReport {
                    instrument: order.instrument.clone(),
                    quantity: order.quantity,
                    price: adjusted_price,
                    timestamp,
                });
            } else {
                log_event(
                    file!(),
                    "SimpleExecutor",
                    "execute",
                    "backtest.execution",
                    line!(),
                    &format!(
                        "Skipped order for {} due to missing price",
                        order.instrument
                    ),
                    None,
                    "none",
                    "POST",
                );
            }
        }

        Ok(executions)
    }
}

#[derive(Default)]
pub struct BacktestReport {
    pub portfolio_history: BTreeMap<DateTime<Utc>, PortfolioSnapshot>,
    pub executions: Vec<ExecutionReport>,
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
}

pub struct Backtester<S: Strategy, E: Executor> {
    strategy: S,
    executor: E,
    initial_cash: f64,
}

impl<S: Strategy, E: Executor> Backtester<S, E> {
    pub fn new(strategy: S, executor: E, initial_cash: f64) -> Self {
        Self {
            strategy,
            executor,
            initial_cash,
        }
    }

    pub fn run<I>(&mut self, snapshots: I) -> BacktestResult<BacktestReport>
    where
        I: IntoIterator<Item = MarketSnapshot>,
    {
        let mut portfolio = PortfolioSnapshot::with_cash(self.initial_cash);
        let mut history = BTreeMap::new();
        let mut executions = Vec::new();
        let mut equity_curve = Vec::new();

        for snapshot in snapshots {
            let timestamp = snapshot.timestamp().unwrap_or_else(Utc::now);
            mark_to_market(&mut portfolio, &snapshot);
            let orders = self.strategy.on_snapshot(&snapshot, &portfolio)?;
            let fills = self.executor.execute(&orders, &snapshot)?;
            for fill in &fills {
                apply_fill(&mut portfolio, fill);
            }
            mark_to_market(&mut portfolio, &snapshot);
            let equity = portfolio_value(&portfolio);
            equity_curve.push((timestamp, equity));
            history.insert(timestamp, portfolio.clone());
            executions.extend(fills);
        }

        Ok(BacktestReport {
            portfolio_history: history,
            executions,
            equity_curve,
        })
    }
}

impl<S: RLStrategy, E: Executor> Backtester<S, E> {
    pub fn run_with_rl<I>(&mut self, snapshots: I) -> BacktestResult<BacktestReport>
    where
        I: IntoIterator<Item = MarketSnapshot>,
    {
        let mut portfolio = PortfolioSnapshot::with_cash(self.initial_cash);
        let mut history = BTreeMap::new();
        let mut executions = Vec::new();
        let mut equity_curve = Vec::new();
        let mut previous_value = self.initial_cash;

        let snapshots_vec: Vec<MarketSnapshot> = snapshots.into_iter().collect();
        let total = snapshots_vec.len();

        for (idx, snapshot) in snapshots_vec.into_iter().enumerate() {
            let timestamp = snapshot.timestamp().unwrap_or_else(Utc::now);
            mark_to_market(&mut portfolio, &snapshot);
            let orders = self.strategy.on_snapshot(&snapshot, &portfolio)?;
            let fills = self.executor.execute(&orders, &snapshot)?;
            for fill in &fills {
                apply_fill(&mut portfolio, fill);
            }
            mark_to_market(&mut portfolio, &snapshot);
            let equity = portfolio_value(&portfolio);
            let reward = equity - previous_value;
            previous_value = equity;
            let done = idx + 1 == total;
            self.strategy.on_reward(reward, done);
            equity_curve.push((timestamp, equity));
            history.insert(timestamp, portfolio.clone());
            executions.extend(fills);
        }

        Ok(BacktestReport {
            portfolio_history: history,
            executions,
            equity_curve,
        })
    }
}

fn mark_to_market(portfolio: &mut PortfolioSnapshot, snapshot: &MarketSnapshot) {
    for event in snapshot.events() {
        if let Some(holding) = portfolio.holdings.get_mut(&event.instrument) {
            holding.price = Some(event.price);
        }
    }
}

fn apply_fill(portfolio: &mut PortfolioSnapshot, fill: &ExecutionReport) {
    let entry = portfolio
        .holdings
        .entry(fill.instrument.clone())
        .or_insert_with(|| Holding::new(0.0, Some(fill.price)));
    entry.amount += fill.quantity;
    entry.price = Some(fill.price);
    portfolio.cash -= fill.quantity * fill.price;

    log_event(
        file!(),
        "Backtester",
        "apply_fill",
        "backtest.portfolio",
        line!(),
        &format!(
            "Applied fill {} {:.2} @ {:.4}",
            fill.instrument, fill.quantity, fill.price
        ),
        None,
        "post",
        "PUT",
    );
}

fn portfolio_value(portfolio: &PortfolioSnapshot) -> f64 {
    let mut value = portfolio.cash;
    for holding in portfolio.holdings.values() {
        if let Some(price) = holding.price {
            value += holding.amount * price;
        }
    }
    value
}

pub struct PipelineStrategy {
    interpreter: Arc<dyn SignalInterpreter>,
}

impl PipelineStrategy {
    pub fn new(interpreter: Arc<dyn SignalInterpreter>) -> Self {
        Self { interpreter }
    }

    pub fn from_features(&self, features: &DataFrame) -> Vec<OrderSignal> {
        self.interpreter.interpret(features)
    }
}

impl Strategy for PipelineStrategy {
    fn on_snapshot(
        &mut self,
        snapshot: &MarketSnapshot,
        _: &PortfolioSnapshot,
    ) -> BacktestResult<Vec<OrderSignal>> {
        let timestamps: Vec<String> = snapshot
            .events()
            .iter()
            .map(|event| event.timestamp.to_rfc3339())
            .collect();
        let instruments: Vec<String> = snapshot
            .events()
            .iter()
            .map(|event| event.instrument.clone())
            .collect();
        let prices: Vec<f64> = snapshot.events().iter().map(|event| event.price).collect();
        let frame = df! {
            "timestamp" => timestamps,
            "instrument" => instruments,
            "price" => prices,
        }
        .map_err(|err| BacktestError::Data(err.to_string()))?;

        Ok(self.interpreter.interpret(&frame))
    }
}

pub struct StaticRLStrategy {
    interpreter: Arc<dyn SignalInterpreter>,
    pub rewards: Vec<f64>,
}

impl StaticRLStrategy {
    pub fn new(interpreter: Arc<dyn SignalInterpreter>) -> Self {
        Self {
            interpreter,
            rewards: Vec::new(),
        }
    }
}

impl Strategy for StaticRLStrategy {
    fn on_snapshot(
        &mut self,
        snapshot: &MarketSnapshot,
        _: &PortfolioSnapshot,
    ) -> BacktestResult<Vec<OrderSignal>> {
        let prices: Vec<f64> = snapshot.events().iter().map(|event| event.price).collect();
        let instruments: Vec<String> = snapshot
            .events()
            .iter()
            .map(|event| event.instrument.clone())
            .collect();
        let frame = df! {
            "price" => prices,
            "instrument" => instruments,
        }
        .map_err(|err| BacktestError::Data(err.to_string()))?;
        Ok(self.interpreter.interpret(&frame))
    }
}

impl RLStrategy for StaticRLStrategy {
    fn on_reward(&mut self, reward: f64, _done: bool) {
        self.rewards.push(reward);
    }
}

//! qliber is a Rust-native port of core concepts from Microsoft's Qlib project.
//! It provides efficient dataset ingestion, feature engineering, metrics evaluation,
//! and structured logging tailored for quantitative research pipelines.

pub mod backtest;
pub mod config;
pub mod contrib;
pub mod dataset;
pub mod ensemble;
pub mod features;
pub mod interpret;
pub mod llm;
pub mod logging;
pub mod meta;
pub mod metrics;
pub mod ops;
pub mod portfolio;
pub mod provider;
pub mod remote;
pub mod riskmodel;
pub mod rl;
pub mod trainer;
pub mod workflow;

pub use backtest::{
    BacktestError, BacktestReport, BacktestResult, Backtester, ExecutionReport, MarketEvent,
    MarketSnapshot, OrderSignal, PipelineStrategy, RLStrategy, SignalInterpreter, SimpleExecutor,
    StaticRLStrategy, Strategy, ThresholdInterpreter,
};
pub use config::{
    ConfigSnapshot, DefaultConfig, InitOptions, REG_CN, REG_TW, REG_US, Region, config_snapshot,
    init, with_data_path,
};
pub use contrib::{Alpha158Processor, Alpha360Processor};
pub use dataset::{
    DataHandler, DataHandlerConfig, DataLoader, DatasetBatch, DatasetError, DropNullsProcessor,
    ExpressionProcessor, FillForwardProcessor, LoaderOptions, MarketData, Processor,
    ProcessorChain, ProcessorRef, RenameProcessor, SelectColumnsProcessor,
};
pub use ensemble::WeightedEnsemble;
pub use features::{with_daily_returns, with_moving_average, with_z_score};
pub use interpret::{FeatureImportance, FeatureInterpreter, PermutationFeatureInterpreter};
#[cfg(feature = "gguf")]
pub use llm::GgufRunner;
pub use llm::{GenerationOptions, LlmError, OllamaClient};
pub use meta::{MetaError, MetaLabelGenerator, WeightLearner};
pub use metrics::{
    AccumulationMode, AnalysisFrequency, FrequencyUnit, IndicatorMethod, MetricsError,
    MetricsResult, PerformanceMetrics, indicator_analysis, indicator_analysis_with_method,
    risk_analysis,
};
pub use ops::{Expression, OpsError, evaluate_expression};
pub use portfolio::{
    Holding, InterestMethod, PortfolioError, PortfolioResult, PortfolioSnapshot, PriceMatrix,
    alpha, annual_return_from_positions, annual_return_from_returns, beta, daily_return_series,
    information_coefficient, max_drawdown_from_returns, position_value, position_value_series,
    rank_information_coefficient, sharpe_ratio_from_returns, volatility,
};
pub use provider::{
    DataProvider, DefaultDataProvider, FeatureBackend, FeatureKey, InMemoryFeatureBackend,
    Instrument, InstrumentStore, PitRecord, PitStore, ProviderError, ProviderResult,
    TradingCalendar,
};
pub use remote::{MockTransport, RemoteDataClient, RemoteError, RemoteResult, RemoteTransport};
pub use riskmodel::{
    FactorRiskModel, PoetRiskModel, PoetThresholdMethod, RiskModelError, RiskModelResult,
    ShrinkageMethod, StructuredFactorModel, StructuredRiskModel,
};
pub use rl::{
    Agent, CounterEnvironment, Environment, IncrementAgent, RlError, RlResult, RlTrainer,
};
pub use trainer::{
    GLOBAL_TRAINER_REGISTRY, MeanModel, TrainableModel, Trainer, TrainerAdapter, TrainerRegistry,
    TrainerRequest, TrainingError, TrainingResult, XgBoostModel, XgBoostObjective,
    XgBoostParameters,
};
pub use workflow::{
    ExperimentManager, ExperimentRecord, Recorder, TaskManager, TaskStatus, TaskSummary,
    WorkflowError, WorkflowResult,
};

pub type Result<T> = anyhow::Result<T>;

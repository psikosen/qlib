use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use polars::lazy::dsl::{col, lit};
use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;
use crate::ops::{OpsError, evaluate_expression};

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("failed to load market data: {source}")]
    Load { source: PolarsError },
    #[error("failed to transform market data: {source}")]
    Transform { source: PolarsError },
    #[error("expression evaluation failed: {0}")]
    Expression(#[from] OpsError),
}

pub type DatasetResult<T> = Result<T, DatasetError>;

#[derive(Clone)]
pub struct MarketData {
    frame: LazyFrame,
}

impl MarketData {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> DatasetResult<Self> {
        let path_ref = path.as_ref();
        let lazy_reader = LazyCsvReader::new(path_ref)
            .has_header(true)
            .with_try_parse_dates(true)
            .with_infer_schema_length(Some(2048));

        let frame = lazy_reader.finish().map_err(|source| {
            log_event(
                file!(),
                "MarketData",
                "from_csv",
                "dataset.load",
                line!(),
                &format!("Failed to load {}", path_ref.display()),
                Some(&source.to_string()),
                "none",
                "GET",
            );
            DatasetError::Load { source }
        })?;

        log_event(
            file!(),
            "MarketData",
            "from_csv",
            "dataset.load",
            line!(),
            &format!("Loaded dataset from {}", path_ref.display()),
            None,
            "none",
            "GET",
        );

        Ok(Self { frame })
    }

    pub fn lazy(&self) -> LazyFrame {
        self.frame.clone()
    }

    pub fn filter_date_range(
        &self,
        column: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> DatasetResult<Self> {
        let mut filter_expr = col(column).is_not_null();

        if let Some(start) = start.map(|dt| dt.naive_utc()) {
            filter_expr = filter_expr.and(col(column).gt_eq(lit(start)));
        }

        if let Some(end) = end.map(|dt| dt.naive_utc()) {
            filter_expr = filter_expr.and(col(column).lt_eq(lit(end)));
        }

        let filtered = self.frame.clone().filter(filter_expr);

        log_event(
            file!(),
            "MarketData",
            "filter_date_range",
            "dataset.filter",
            line!(),
            &format!("Applied date filter on column {column}"),
            None,
            "none",
            "GET",
        );

        Ok(Self { frame: filtered })
    }

    pub fn select_columns(&self, columns: &[&str]) -> DatasetResult<Self> {
        let selection: Vec<Expr> = columns.iter().copied().map(col).collect();
        let selected = self.frame.clone().select(selection);

        log_event(
            file!(),
            "MarketData",
            "select_columns",
            "dataset.transform",
            line!(),
            &format!("Selected columns: {}", columns.join(", ")),
            None,
            "none",
            "GET",
        );

        Ok(Self { frame: selected })
    }

    pub fn collect(&self) -> DatasetResult<DataFrame> {
        self.frame
            .clone()
            .collect()
            .map_err(|source| DatasetError::Transform { source })
    }
}

pub trait Processor: Send + Sync {
    fn name(&self) -> &str;
    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame>;
}

pub type ProcessorRef = Arc<dyn Processor>;

#[derive(Clone, Default)]
pub struct ProcessorChain {
    steps: Vec<ProcessorRef>,
}

impl ProcessorChain {
    pub fn new(steps: Vec<ProcessorRef>) -> Self {
        Self { steps }
    }

    pub fn push(&mut self, processor: ProcessorRef) {
        self.steps.push(processor);
    }

    pub fn apply(&self, mut frame: DataFrame) -> DatasetResult<DataFrame> {
        for processor in &self.steps {
            log_event(
                file!(),
                "ProcessorChain",
                "apply",
                "dataset.processor",
                line!(),
                &format!("Applying processor {}", processor.name()),
                None,
                "none",
                "GET",
            );
            frame = processor.process(frame)?;
        }
        Ok(frame)
    }
}

pub struct SelectColumnsProcessor {
    columns: Vec<String>,
}

impl SelectColumnsProcessor {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }
}

impl Processor for SelectColumnsProcessor {
    fn name(&self) -> &str {
        "select_columns"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        let selected = frame
            .select(&self.columns)
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(selected)
    }
}

pub struct FillForwardProcessor {
    columns: Option<Vec<String>>,
}

impl FillForwardProcessor {
    pub fn new(columns: Option<Vec<String>>) -> Self {
        Self { columns }
    }
}

impl Processor for FillForwardProcessor {
    fn name(&self) -> &str {
        "fill_forward"
    }

    fn process(&self, mut frame: DataFrame) -> DatasetResult<DataFrame> {
        let columns: Vec<String> = self.columns.clone().unwrap_or_else(|| {
            frame
                .get_columns()
                .iter()
                .map(|c| c.name().to_string())
                .collect()
        });

        for column in columns {
            let filled = frame
                .column(&column)
                .map_err(|source| DatasetError::Transform { source })?
                .clone()
                .fill_null(FillNullStrategy::Forward(None))
                .map_err(|source| DatasetError::Transform { source })?;
            frame
                .replace(&column, filled)
                .map_err(|source| DatasetError::Transform { source })?;
        }
        Ok(frame)
    }
}

pub struct DropNullsProcessor {
    columns: Option<Vec<String>>,
}

impl DropNullsProcessor {
    pub fn new(columns: Option<Vec<String>>) -> Self {
        Self { columns }
    }
}

impl Processor for DropNullsProcessor {
    fn name(&self) -> &str {
        "drop_nulls"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        let dropped = match self.columns.clone() {
            Some(cols) => frame
                .drop_nulls::<String>(Some(&cols))
                .map_err(|source| DatasetError::Transform { source })?,
            None => frame
                .drop_nulls::<&str>(None)
                .map_err(|source| DatasetError::Transform { source })?,
        };
        Ok(dropped)
    }
}

pub struct RenameProcessor {
    from: String,
    to: String,
}

impl RenameProcessor {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
        }
    }
}

impl Processor for RenameProcessor {
    fn name(&self) -> &str {
        "rename"
    }

    fn process(&self, mut frame: DataFrame) -> DatasetResult<DataFrame> {
        frame
            .rename(&self.from, &self.to)
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(frame)
    }
}

pub struct ExpressionProcessor {
    expression: String,
    output: String,
}

impl ExpressionProcessor {
    pub fn new(expression: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            expression: expression.into(),
            output: output.into(),
        }
    }
}

impl Processor for ExpressionProcessor {
    fn name(&self) -> &str {
        "expression"
    }

    fn process(&self, mut frame: DataFrame) -> DatasetResult<DataFrame> {
        let mut series = evaluate_expression(&self.expression, &frame)?;
        series.rename(&self.output);
        frame
            .with_column(series)
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(frame)
    }
}

#[derive(Clone, Default)]
pub struct DataHandlerConfig {
    pub feature_processors: Vec<ProcessorRef>,
    pub label_processors: Vec<ProcessorRef>,
    pub feature_columns: Vec<String>,
    pub label_columns: Vec<String>,
}

pub struct DataHandler {
    market: MarketData,
    config: DataHandlerConfig,
}

impl DataHandler {
    pub fn new(market: MarketData, config: DataHandlerConfig) -> Self {
        Self { market, config }
    }

    pub fn loader(&self, options: LoaderOptions) -> DatasetResult<DataLoader> {
        let mut source = self.market.clone();
        if let Some(column) = options.date_column.as_deref() {
            source = source.filter_date_range(column, options.start, options.end)?;
        }

        Ok(DataLoader {
            source,
            feature_chain: ProcessorChain::new(self.config.feature_processors.clone()),
            label_chain: ProcessorChain::new(self.config.label_processors.clone()),
            feature_columns: self.config.feature_columns.clone(),
            label_columns: self.config.label_columns.clone(),
            limit: options.limit,
        })
    }
}

#[derive(Clone, Default)]
pub struct LoaderOptions {
    pub date_column: Option<String>,
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

pub struct DatasetBatch {
    pub features: DataFrame,
    pub labels: DataFrame,
}

pub struct DataLoader {
    source: MarketData,
    feature_chain: ProcessorChain,
    label_chain: ProcessorChain,
    feature_columns: Vec<String>,
    label_columns: Vec<String>,
    limit: Option<usize>,
}

impl DataLoader {
    fn base_frame(&self) -> DatasetResult<DataFrame> {
        let mut frame = self.source.collect()?;
        if let Some(limit) = self.limit
            && frame.height() > limit
        {
            frame = frame.head(Some(limit));
        }
        Ok(frame)
    }

    pub fn load_features(&self) -> DatasetResult<DataFrame> {
        let base = self.base_frame()?;
        let processed = self.feature_chain.apply(base)?;
        Self::select_columns(processed, &self.feature_columns)
    }

    pub fn load_labels(&self) -> DatasetResult<DataFrame> {
        let base = self.base_frame()?;
        let processed = self.label_chain.apply(base)?;
        Self::select_columns(processed, &self.label_columns)
    }

    pub fn load(&self) -> DatasetResult<DatasetBatch> {
        let base = self.base_frame()?;
        let feature_frame = self
            .feature_chain
            .apply(base.clone())
            .and_then(|frame| Self::select_columns(frame, &self.feature_columns))?;
        let label_frame = self
            .label_chain
            .apply(base)
            .and_then(|frame| Self::select_columns(frame, &self.label_columns))?;
        Ok(DatasetBatch {
            features: feature_frame,
            labels: label_frame,
        })
    }

    fn select_columns(frame: DataFrame, columns: &[String]) -> DatasetResult<DataFrame> {
        if columns.is_empty() {
            return Ok(frame);
        }
        let selected = frame
            .select(columns)
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(selected)
    }
}

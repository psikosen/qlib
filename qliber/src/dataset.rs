use std::path::Path;

use chrono::{DateTime, Utc};
use polars::lazy::dsl::{col, lit};
use polars::prelude::*;
use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("failed to load market data: {source}")]
    Load { source: PolarsError },
    #[error("failed to transform market data: {source}")]
    Transform { source: PolarsError },
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

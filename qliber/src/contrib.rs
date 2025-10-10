use polars::prelude::*;

use crate::dataset::{DatasetError, DatasetResult, Processor};
use crate::features::{with_daily_returns, with_moving_average, with_z_score};

pub struct Alpha158Processor {
    price_column: String,
}

impl Alpha158Processor {
    pub fn new(price_column: impl Into<String>) -> Self {
        Self {
            price_column: price_column.into(),
        }
    }
}

impl Processor for Alpha158Processor {
    fn name(&self) -> &str {
        "alpha158"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        let returns = with_daily_returns(&frame, &self.price_column, "return")
            .map_err(|source| DatasetError::Transform { source })?;
        let ma5 = with_moving_average(&returns, &self.price_column, 5, "ma_5")
            .map_err(|source| DatasetError::Transform { source })?;
        let enriched = with_moving_average(&ma5, &self.price_column, 20, "ma_20")
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(enriched)
    }
}

pub struct Alpha360Processor {
    price_column: String,
}

impl Alpha360Processor {
    pub fn new(price_column: impl Into<String>) -> Self {
        Self {
            price_column: price_column.into(),
        }
    }
}

impl Processor for Alpha360Processor {
    fn name(&self) -> &str {
        "alpha360"
    }

    fn process(&self, frame: DataFrame) -> DatasetResult<DataFrame> {
        let returns = with_daily_returns(&frame, &self.price_column, "return")
            .map_err(|source| DatasetError::Transform { source })?;
        let ma = with_moving_average(&returns, &self.price_column, 10, "ma_10")
            .map_err(|source| DatasetError::Transform { source })?;
        let zscore = with_z_score(&ma, &self.price_column, 15, "z_price")
            .map_err(|source| DatasetError::Transform { source })?;

        let mut frame = zscore.clone();
        let return_series = frame
            .column("return")
            .map_err(|source| DatasetError::Transform { source })?
            .clone();
        let squared: Float64Chunked = return_series
            .f64()
            .map_err(|_| DatasetError::Transform {
                source: PolarsError::ComputeError("expected float return".into()),
            })?
            .into_iter()
            .map(|value| value.map(|v| v * v))
            .collect();
        let mut squared_series = squared.into_series();
        squared_series.rename("return_sq");
        frame
            .with_column(squared_series)
            .map_err(|source| DatasetError::Transform { source })?;
        Ok(frame)
    }
}

use std::collections::BTreeMap;

use chrono::NaiveDate;
use polars::prelude::*;
use qliber::portfolio::{PortfolioSnapshot, PriceMatrix};
use qliber::{
    Holding, InterestMethod, alpha, annual_return_from_positions, annual_return_from_returns, beta,
    daily_return_series, information_coefficient, max_drawdown_from_returns, position_value,
    position_value_series, rank_information_coefficient, sharpe_ratio_from_returns, volatility,
};

fn sample_price_matrix() -> PriceMatrix {
    let frame = df! {
        "date" => &["2024-01-02", "2024-01-03", "2024-01-04"],
        "AAA" => &[10.0, 11.0, 10.5],
        "BBB" => &[20.0, 19.5, 20.5],
    }
    .unwrap();

    PriceMatrix::from_dataframe(&frame, "date").expect("build price matrix")
}

fn sample_positions() -> BTreeMap<NaiveDate, PortfolioSnapshot> {
    let mut positions = BTreeMap::new();
    let mut snapshot = PortfolioSnapshot::with_cash(60.0);
    snapshot.insert_holding("AAA", Holding::new(5.0, None));
    snapshot.insert_holding("BBB", Holding::new(2.0, None));
    positions.insert(
        NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        snapshot.clone(),
    );

    let mut snapshot_next = PortfolioSnapshot::with_cash(60.0);
    snapshot_next.insert_holding("AAA", Holding::new(5.0, None));
    snapshot_next.insert_holding("BBB", Holding::new(2.0, None));
    positions.insert(
        NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
        snapshot_next.clone(),
    );

    let mut snapshot_final = PortfolioSnapshot::with_cash(60.0);
    snapshot_final.insert_holding("AAA", Holding::new(5.0, None));
    snapshot_final.insert_holding("BBB", Holding::new(2.0, None));
    positions.insert(
        NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
        snapshot_final.clone(),
    );

    positions
}

#[test]
fn test_price_matrix_and_position_values() {
    let prices = sample_price_matrix();
    let positions = sample_positions();

    let first_date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let first_value = position_value(&positions[&first_date], &prices, first_date).unwrap();
    let expected_first = 5.0 * 10.0 + 2.0 * 20.0 + 60.0;
    assert!((first_value - expected_first).abs() < 1e-9);

    let series = position_value_series(&positions, &prices).unwrap();
    let values: Vec<f64> = series.values().copied().collect();
    assert_eq!(values.len(), 3);
    let expected_second = 5.0 * 11.0 + 2.0 * 19.5 + 60.0;
    let expected_third = 5.0 * 10.5 + 2.0 * 20.5 + 60.0;
    assert!((values[0] - expected_first).abs() < 1e-9);
    assert!((values[1] - expected_second).abs() < 1e-9);
    assert!((values[2] - expected_third).abs() < 1e-9);
}

#[test]
fn test_return_series_and_annualization() {
    let prices = sample_price_matrix();
    let positions = sample_positions();
    let returns = daily_return_series(&positions, &prices, 150.0).unwrap();
    assert_eq!(returns.len(), 3);
    assert!((returns[0].1 - 0.0).abs() < 1e-9);

    let second_value = 5.0 * 11.0 + 2.0 * 19.5 + 60.0;
    let third_value = 5.0 * 10.5 + 2.0 * 20.5 + 60.0;
    let expected_second = second_value / 150.0 - 1.0;
    let expected_third = third_value / second_value - 1.0;
    assert!((returns[1].1 - expected_second).abs() < 1e-9);
    assert!((returns[2].1 - expected_third).abs() < 1e-9);

    let annual = annual_return_from_positions(&positions, &prices, 150.0, 252.0).unwrap();
    let final_value = third_value;
    let expected = (final_value / 150.0).powf(252.0 / 3.0) - 1.0;
    assert!((annual - expected).abs() < 1e-9);
}

#[test]
fn test_risk_helpers_from_returns() {
    let returns = vec![0.01, -0.005, 0.02, 0.015, -0.01];
    let annual_compound = annual_return_from_returns(&returns, InterestMethod::Compound, 252.0);
    let annual_simple = annual_return_from_returns(&returns, InterestMethod::Simple, 252.0);
    let mean = returns.iter().copied().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| {
            let diff = r - mean;
            diff * diff
        })
        .sum::<f64>()
        / (returns.len() as f64 - 1.0);
    let std = variance.sqrt();

    let expected_compound = (1.0 + mean).powf(252.0) - 1.0;
    let expected_simple = mean * 252.0;
    assert!((annual_compound - expected_compound).abs() < 1e-9);
    assert!((annual_simple - expected_simple).abs() < 1e-9);

    let sharpe = sharpe_ratio_from_returns(&returns, 0.02, InterestMethod::Compound, 252.0);
    let expected_sharpe = (annual_compound - 0.02) / std / 252f64.sqrt();
    assert!((sharpe - expected_sharpe).abs() < 1e-9);

    let mdd = max_drawdown_from_returns(&returns);
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut expected_mdd = 0.0;
    for value in &returns {
        cumulative *= 1.0 + value;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (cumulative / peak) - 1.0;
        if drawdown < expected_mdd {
            expected_mdd = drawdown;
        }
    }
    assert!((mdd - expected_mdd).abs() < 1e-9);

    let vol = volatility(&returns);
    assert!((vol - std).abs() < 1e-9);
}

#[test]
fn test_beta_and_alpha_and_ic() {
    let strategy = vec![0.01, 0.015, 0.0, 0.02, -0.005];
    let benchmark = vec![0.005, 0.007, -0.002, 0.01, 0.0];

    let beta_val = beta(&strategy, &benchmark).unwrap();
    let mean_strategy = strategy.iter().copied().sum::<f64>() / strategy.len() as f64;
    let mean_benchmark = benchmark.iter().copied().sum::<f64>() / benchmark.len() as f64;
    let cov = strategy
        .iter()
        .zip(benchmark.iter())
        .map(|(s, b)| (s - mean_strategy) * (b - mean_benchmark))
        .sum::<f64>()
        / (strategy.len() as f64 - 1.0);
    let var_b = benchmark
        .iter()
        .map(|b| {
            let diff = b - mean_benchmark;
            diff * diff
        })
        .sum::<f64>()
        / (benchmark.len() as f64 - 1.0);
    let expected_beta = cov / var_b;
    assert!((beta_val - expected_beta).abs() < 1e-9);

    let alpha_val = alpha(&strategy, &benchmark, 0.02, InterestMethod::Compound, 252.0).unwrap();
    let annual_strategy = annual_return_from_returns(&strategy, InterestMethod::Compound, 252.0);
    let annual_benchmark = annual_return_from_returns(&benchmark, InterestMethod::Compound, 252.0);
    let expected_alpha = annual_strategy - 0.02 - expected_beta * (annual_benchmark - 0.02);
    assert!((alpha_val - expected_alpha).abs() < 1e-9);

    let rank_ic = rank_information_coefficient(&strategy, &benchmark).unwrap();
    let mut strat_ranks: Vec<(usize, f64)> = strategy.iter().copied().enumerate().collect();
    strat_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut bench_ranks: Vec<(usize, f64)> = benchmark.iter().copied().enumerate().collect();
    bench_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut strat_rank_vec = vec![0.0; strategy.len()];
    let mut bench_rank_vec = vec![0.0; benchmark.len()];
    for (rank, (idx, _)) in strat_ranks.iter().enumerate() {
        strat_rank_vec[*idx] = (rank + 1) as f64;
    }
    for (rank, (idx, _)) in bench_ranks.iter().enumerate() {
        bench_rank_vec[*idx] = (rank + 1) as f64;
    }
    let diffs: Vec<f64> = strat_rank_vec
        .iter()
        .zip(bench_rank_vec.iter())
        .map(|(a, b)| a - b)
        .collect();
    let sum_diff_sq: f64 = diffs.iter().map(|d| d * d).sum();
    let n = strategy.len() as f64;
    let expected_rank_ic = 1.0 - 6.0 * sum_diff_sq / (n * (n * n - 1.0));
    assert!((rank_ic - expected_rank_ic).abs() < 1e-9);

    let normal_ic = information_coefficient(&strategy, &benchmark).unwrap();
    let numerator = strategy
        .iter()
        .zip(benchmark.iter())
        .map(|(s, b)| (s - mean_strategy) * (b - mean_benchmark))
        .sum::<f64>();
    let denom_strategy = strategy
        .iter()
        .map(|s| {
            let diff = s - mean_strategy;
            diff * diff
        })
        .sum::<f64>();
    let denom_benchmark = benchmark
        .iter()
        .map(|b| {
            let diff = b - mean_benchmark;
            diff * diff
        })
        .sum::<f64>();
    let expected_normal_ic = numerator / (denom_strategy.sqrt() * denom_benchmark.sqrt());
    assert!((normal_ic - expected_normal_ic).abs() < 1e-9);
}

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn qrun_cli_train_produces_metrics() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let config_path = dir.path().join("config.json");
    let config = serde_json::json!({
        "task": "train",
        "label_column": "label",
        "epochs": 2
    });
    fs::write(&config_path, config.to_string())?;

    let mut cmd = Command::cargo_bin("qrun")?;
    cmd.arg("--config").arg(&config_path);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Training complete"))
        .stdout(predicate::str::contains("epoch_0_mse"));

    Ok(())
}

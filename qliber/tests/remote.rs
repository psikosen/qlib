use chrono::Utc;
use serde_json::json;

use qliber::remote::{MockTransport, RemoteDataClient};

#[test]
fn remote_client_fetches_resources() -> anyhow::Result<()> {
    let transport = std::sync::Arc::new(MockTransport::new());
    transport.push_response(Ok(json!(["2024-01-01T00:00:00Z"])));
    transport.push_response(Ok(json!([
        {
            "symbol": "TEST",
            "start": "2024-01-01",
            "end": null,
            "metadata": {"exchange": "SIM"}
        }
    ])));
    transport.push_response(Ok(json!([
        {"timestamp": "2024-01-01T00:00:00Z", "value": 1.0},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 2.0}
    ])));

    let client = RemoteDataClient::new(transport.clone());
    let sessions = client.fetch_calendar()?;
    assert_eq!(sessions.len(), 1);

    let instruments = client.fetch_instruments()?;
    assert_eq!(instruments.len(), 1);
    assert_eq!(instruments[0].symbol, "TEST");

    let frame = client.fetch_feature("TEST", "value", Utc::now())?;
    assert_eq!(frame.height(), 2);

    Ok(())
}

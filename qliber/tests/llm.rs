use httpmock::prelude::*;
use serde_json::json;

use qliber::{GenerationOptions, OllamaClient};

#[test]
fn ollama_client_generates_with_options() {
    let server = MockServer::start();
    let expected_body = json!({
        "model": "test",
        "prompt": "hello",
        "stream": false,
        "options": {"num_predict": 16}
    });

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/api/generate")
            .json_body(expected_body.clone());
        then.status(200)
            .json_body(json!({"model": "test", "done": true, "response": "world"}));
    });

    let client = OllamaClient::new(server.base_url(), "test").expect("client");
    let options = GenerationOptions {
        max_tokens: Some(16),
        ..Default::default()
    };
    let response = client
        .generate_with_options("hello", &options)
        .expect("generation");
    assert_eq!(response, "world");
    mock.assert();
}

#[test]
fn ollama_client_supports_system_prompt() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/api/generate").json_body(json!({
            "model": "phi",
            "prompt": "question",
            "stream": false,
            "system": "assistant",
        }));
        then.status(200)
            .json_body(json!({"model": "phi", "done": true, "response": "answer"}));
    });

    let client = OllamaClient::new(server.base_url(), "phi").expect("client");
    let options = GenerationOptions {
        system_prompt: Some("assistant".to_string()),
        ..Default::default()
    };
    let response = client
        .generate_with_options("question", &options)
        .expect("generation");
    assert_eq!(response, "answer");
    mock.assert();
}

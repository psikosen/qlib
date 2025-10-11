use reqwest::Url;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::logging::log_event;

#[cfg(feature = "gguf")]
use llama_cpp::{
    LlamaModel, LlamaParams, LlamaSession, SessionParams, standard_sampler::StandardSampler,
};
#[cfg(feature = "gguf")]
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

/// Configuration options shared by the supported language model backends.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Optional system prompt to prepend to the provided user prompt.
    pub system_prompt: Option<String>,
    /// Maximum number of tokens to decode when generating a completion.
    pub max_tokens: Option<usize>,
    /// Backend-specific options expressed as a JSON object (e.g. temperature).
    pub options: Option<Value>,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("invalid base url: {0}")]
    InvalidBaseUrl(String),
    #[error("ollama options must be a JSON object")]
    InvalidOptions,
    #[error("ollama returned no response body")]
    EmptyResponse,
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[cfg(feature = "gguf")]
    #[error("failed to load gguf model: {0}")]
    Load(#[from] llama_cpp::model::LlamaLoadError),
    #[cfg(feature = "gguf")]
    #[error("gguf session error: {0}")]
    Session(#[from] llama_cpp::session::LlamaContextError),
}

/// Blocking client for the Ollama HTTP API.
#[derive(Clone)]
pub struct OllamaClient {
    base_url: Url,
    model: String,
    client: Client,
}

impl OllamaClient {
    /// Create a new Ollama client targeting the provided server and model name.
    pub fn new(base_url: impl AsRef<str>, model: impl Into<String>) -> Result<Self, LlmError> {
        let mut url = Url::parse(base_url.as_ref())
            .map_err(|_| LlmError::InvalidBaseUrl(base_url.as_ref().into()))?;
        if url.path().is_empty() || url.path() == "/" {
            url.set_path("/api/generate");
        } else if !url.path().ends_with("/api/generate") {
            let mut parts: Vec<String> = url
                .path_segments()
                .map(|segments| segments.map(|s| s.to_string()).collect())
                .unwrap_or_else(Vec::new);
            if parts.last().map(|s| s != "api").unwrap_or(true) {
                parts.push("api".into());
            }
            parts.push("generate".into());
            let mut path = url
                .path_segments_mut()
                .map_err(|_| LlmError::InvalidBaseUrl(base_url.as_ref().into()))?;
            path.clear();
            for part in parts {
                path.push(&part);
            }
        }

        let client = Client::builder().build()?;
        Ok(Self {
            base_url: url,
            model: model.into(),
            client,
        })
    }

    /// Generate a completion using the provided prompt.
    pub fn generate(&self, prompt: &str) -> Result<String, LlmError> {
        self.generate_with_options(prompt, &GenerationOptions::default())
    }

    /// Generate a completion with explicit decoding options.
    pub fn generate_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<String, LlmError> {
        let mut options_map = match options.options.clone() {
            Some(Value::Object(map)) => map,
            Some(_) => return Err(LlmError::InvalidOptions),
            None => Map::new(),
        };
        if let Some(max_tokens) = options.max_tokens {
            options_map.insert("num_predict".into(), Value::from(max_tokens as i64));
        }
        let payload = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            system: options.system_prompt.clone(),
            options: if options_map.is_empty() {
                None
            } else {
                Some(Value::Object(options_map))
            },
        };

        log_event(
            file!(),
            "OllamaClient",
            "generate",
            "llm.ollama",
            line!(),
            &format!("prompt_len={} model={}", prompt.len(), self.model),
            None,
            "pre",
            "POST",
        );

        let response = self
            .client
            .post(self.base_url.clone())
            .json(&payload)
            .send()?
            .error_for_status()?;
        let decoded: OllamaGenerateResponse = response.json()?;
        let body = decoded.response.ok_or(LlmError::EmptyResponse)?;

        log_event(
            file!(),
            "OllamaClient",
            "generate",
            "llm.ollama",
            line!(),
            &format!("generated_len={} model={}", body.len(), self.model),
            None,
            "post",
            "POST",
        );

        Ok(body)
    }
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    #[serde(default)]
    response: Option<String>,
}

#[cfg(feature = "gguf")]
pub struct GgufRunner {
    model: Arc<LlamaModel>,
    session_params: SessionParams,
    session_cache: Mutex<Option<LlamaSession>>,
}

#[cfg(feature = "gguf")]
impl GgufRunner {
    /// Load a GGUF model from disk with default llama.cpp parameters.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, LlmError> {
        Self::with_params(model_path, LlamaParams::default(), SessionParams::default())
    }

    /// Load a GGUF model with explicit model and session parameters.
    pub fn with_params(
        model_path: impl AsRef<Path>,
        model_params: LlamaParams,
        session_params: SessionParams,
    ) -> Result<Self, LlmError> {
        let model = LlamaModel::load_from_file(model_path, model_params.clone())?;
        Ok(Self {
            model: Arc::new(model),
            session_params,
            session_cache: Mutex::new(None),
        })
    }

    fn borrow_session(&self) -> Result<LlamaSession, LlmError> {
        let mut guard = self.session_cache.lock().expect("lock session cache");
        if let Some(session) = guard.take() {
            Ok(session)
        } else {
            Ok(self.model.create_session(self.session_params.clone())?)
        }
    }

    fn store_session(&self, session: LlamaSession) {
        if let Ok(mut guard) = self.session_cache.lock() {
            *guard = Some(session);
        }
    }

    /// Generate a completion from a GGUF model using llama.cpp.
    pub fn generate_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<String, LlmError> {
        let mut session = self.borrow_session()?;
        let mut combined_prompt = String::new();
        if let Some(system) = &options.system_prompt {
            combined_prompt.push_str(system);
            if !system.ends_with('\n') {
                combined_prompt.push('\n');
            }
        }
        combined_prompt.push_str(prompt);
        session.advance_context(combined_prompt.as_bytes())?;

        let max_tokens = options.max_tokens.unwrap_or(256);
        log_event(
            file!(),
            "GgufRunner",
            "generate",
            "llm.gguf",
            line!(),
            &format!(
                "prompt_len={} max_tokens={}",
                combined_prompt.len(),
                max_tokens
            ),
            None,
            "pre",
            "NONE",
        );

        let completion = session
            .start_completing_with(StandardSampler::default(), max_tokens)
            .map_err(LlmError::from)?
            .into_string();

        log_event(
            file!(),
            "GgufRunner",
            "generate",
            "llm.gguf",
            line!(),
            &format!("generated_len={}", completion.len()),
            None,
            "post",
            "NONE",
        );

        self.store_session(session);

        Ok(completion)
    }

    /// Convenience helper mirroring [`OllamaClient::generate`].
    pub fn generate(&self, prompt: &str) -> Result<String, LlmError> {
        self.generate_with_options(prompt, &GenerationOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_object_options() {
        let client = OllamaClient::new("http://localhost:11434", "qwen").unwrap();
        let options = GenerationOptions {
            options: Some(Value::from("temperature")),
            ..Default::default()
        };
        let err = client.generate_with_options("hi", &options).unwrap_err();
        assert!(matches!(err, LlmError::InvalidOptions));
    }
}

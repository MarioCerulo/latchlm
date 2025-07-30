// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module implements a client for interacting with the Gemini API,
//! including support for model variants and structured error handling.

use std::{future::ready, sync::Arc};

use latchlm_core::{AiModel, AiProvider, AiRequest, AiResponse, BoxFuture, Error, ModelId};

use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

mod response;
pub use response::*;

/// Variants representing supported Gemini models.
///
/// These variants map to the actual model identifiers used by the Gemini API.
#[derive(Debug, EnumIter, Deserialize, Clone, PartialEq, Eq, Hash)]
#[serde(try_from = "&str")]
#[non_exhaustive]
pub enum GeminiModel {
    Flash20,
    Flash20Lite,
    Flash25,
    Pro25,
    FlashThinking,
}

impl std::fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl TryFrom<&str> for GeminiModel {
    type Error = Error;
    fn try_from(value: &str) -> latchlm_core::Result<Self> {
        match value {
            "gemini-2.0-flash" => Ok(Self::Flash20),
            "gemini-2.0-flash-lite" => Ok(Self::Flash20Lite),
            "gemini-2.0-flash-thinking-exp-01-21" => Ok(Self::FlashThinking),
            "gemini-2.5-flash" => Ok(Self::Flash25),
            "gemini-2.5-pro" => Ok(Self::Pro25),
            invalid_model => Err(Error::InvalidModelError(invalid_model.to_owned())),
        }
    }
}

impl AsRef<str> for GeminiModel {
    fn as_ref(&self) -> &str {
        match self {
            GeminiModel::Flash20 => "gemini-2.0-flash",
            GeminiModel::Flash20Lite => "gemini-2.0-flash-lite",
            GeminiModel::Pro25 => "gemini-2.5-pro",
            GeminiModel::FlashThinking => "gemini-2.0-flash-thinking-exp-01-21",
            GeminiModel::Flash25 => "gemini-2.5-flash",
        }
    }
}
impl AiModel for GeminiModel {}

impl GeminiModel {
    /// Get the [`ModelId`] of the specified model
    ///
    /// [`ModelId`]: latchlm_core::ModelId
    pub fn model_id(&self) -> ModelId {
        match self {
            Self::Flash20 => ModelId {
                id: self.as_ref().to_string(),
                name: "Gemini 2.0 Flash".to_string(),
            },
            Self::Pro25 => ModelId {
                id: self.as_ref().to_string(),
                name: "Gemini 2.5 Pro".to_string(),
            },
            Self::Flash20Lite => ModelId {
                id: self.as_ref().to_string(),
                name: "Gemini 2.0 Flash Lite".to_string(),
            },
            Self::FlashThinking => ModelId {
                id: self.as_ref().to_string(),
                name: "Gemini 2.0 Flash Thinking".to_string(),
            },
            Self::Flash25 => ModelId {
                id: self.as_ref().to_string(),
                name: "Gemini 2.5 Flash".to_string(),
            },
        }
    }

    /// Returns all supported Gemini model variants as a vector of [`ModelId`].
    pub fn variants() -> Vec<ModelId> {
        Self::iter().map(|variant| variant.model_id()).collect()
    }
}

#[derive(Debug)]
pub enum GeminiError {
    /// Returned when no API key is provided
    MissingApiKeyError,
    /// Returned when no HTTP client is provided
    MissingClientError,
}

impl From<GeminiError> for Error {
    fn from(value: GeminiError) -> Self {
        match value {
            GeminiError::MissingApiKeyError => Error::ProviderError {
                provider: "Gemini".into(),
                error: "Missing API key".into(),
            },
            GeminiError::MissingClientError => Error::ProviderError {
                provider: "Gemini".into(),
                error: "Missing reqwest::Client".into(),
            },
        }
    }
}

impl std::error::Error for GeminiError {}

impl std::fmt::Display for GeminiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeminiError::MissingApiKeyError => write!(f, "API key is required"),
            GeminiError::MissingClientError => write!(f, "HTTP client is required"),
        }
    }
}

/// Builder for constructing a [`Gemini`] client instance
#[derive(Default)]
pub struct GeminiBuilder {
    client: Option<reqwest::Client>,
    api_key: Option<SecretString>,
}

impl GeminiBuilder {
    /// Create a new builder instance with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a custom HTTP client
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the API key
    pub fn api_key(mut self, api_key: SecretString) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Loads the API key from the `GEMINI_API_KEY` environment variable
    pub fn api_key_from_env(mut self) -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("GEMINI_API_KEY")?;

        self.api_key = Some(SecretString::from(api_key));
        Ok(self)
    }

    /// Constructs a [`Gemini`] instance
    ///
    /// # Errors
    /// Returns an error if the client or API key are missing
    ///
    /// # Panics
    /// Panics if the base URL is not set, which should never happen since it has a default.
    pub fn build(self) -> latchlm_core::Result<Gemini> {
        let client = self.client.ok_or(GeminiError::MissingClientError)?;
        let api_key = self.api_key.ok_or(GeminiError::MissingApiKeyError)?;

        Ok(Gemini::new(client, api_key))
    }
}

/// A client for interacting with the Gemini API.
pub struct Gemini {
    client: reqwest::Client,
    base_url: reqwest::Url,
    api_key: Arc<SecretString>,
}

const GEMINI_API_URL: &str = "https://generativelanguage.googleapis.com";
// The HTTP header used for authentication with the gemini api
const X_GOOG_API_KEY: &str = "x-goog-api-key";

impl Gemini {
    /// Creates a new `Gemini` client instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to a preconfigured [`reqwest::Client`].
    /// * `api_key` - The API key wrapped in `SecretString` for secure handling
    pub fn new(client: reqwest::Client, api_key: SecretString) -> Self {
        Self {
            client,
            base_url: reqwest::Url::parse(GEMINI_API_URL).expect("Failed to parse base url"),
            api_key: Arc::new(api_key),
        }
    }

    /// Creates a new `Gemini` client instance with a custom base URL.
    ///
    /// This constructor is intended exclusively for testing and mocking scenarios
    /// and should **never** be used in production code.
    ///
    /// # Feature
    /// Requires the `test-utils` feature flag.
    #[cfg(feature = "test-utils")]
    pub fn new_with_base_url(
        client: reqwest::Client,
        base_url: reqwest::Url,
        api_key: SecretString,
    ) -> Self {
        Self {
            client,
            base_url,
            api_key: Arc::new(api_key),
        }
    }

    /// Sends a request to the Gemini API.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails, the response status is not successful,
    /// or if the response cannot be parsed.
    ///
    /// [`Error`]: latchlm_core::Error
    pub async fn request(
        &self,
        model: GeminiModel,
        request: AiRequest,
    ) -> latchlm_core::Result<GeminiResponse> {
        let url = self
            .base_url
            .join(&format!(
                "/v1beta/models/{}:generateContent",
                model.as_ref()
            ))
            .expect("Failed to parse the URL");

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            X_GOOG_API_KEY,
            self.api_key
                .expose_secret()
                .parse()
                .expect("Failed to parse header"),
        );

        let payload = serde_json::json!({"contents": [{"parts": {"text": request.text}}]});

        let response = self
            .client
            .post(url)
            .headers(headers)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError {
                status: response.status().as_u16(),
                message: response.text().await?,
            });
        }

        let bytes = response.bytes().await?;

        let response: GeminiResponse = serde_json::from_slice(&bytes)?;

        Ok(response)
    }
}

impl AiProvider for Gemini {
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, latchlm_core::Result<AiResponse>> {
        let Ok(model) = GeminiModel::try_from(model.as_ref()) else {
            let model_name = model.as_ref().to_owned();
            return Box::pin(ready(Err(Error::InvalidModelError(model_name))));
        };

        Box::pin(async move { self.request(model, request).await.map(Into::into) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_gemini_model_try_from_valid(model in prop_oneof![
            Just(GeminiModel::Flash20),
            Just(GeminiModel::Flash20Lite),
            Just(GeminiModel::Flash25),
            Just(GeminiModel::Pro25),
            Just(GeminiModel::FlashThinking),
        ]) {
            let model_str = model.as_ref();
            let parsed_model = GeminiModel::try_from(model_str).unwrap();
            prop_assert_eq!(model, parsed_model);
        }

        #[test]
        fn test_gemini_model_try_from_invalid(model_str in "\\PC*") {
            let valid_ids: Vec<_> = GeminiModel::variants()
                .iter()
                .map(|v| v.clone().id)
                .collect();

            prop_assume!(!valid_ids.contains(&model_str));

            let err = GeminiModel::try_from(model_str.as_str()).unwrap_err();
            prop_assert_eq!(err.to_string(), format!("Invalid model name: {}", model_str));
        }
    }
}

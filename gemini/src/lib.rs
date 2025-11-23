// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module implements a client for interacting with the Gemini API,
//! including support for model variants and structured error handling.

use std::{future::ready, sync::Arc};

use eventsource_stream::Eventsource;
use futures::{FutureExt, StreamExt, stream::BoxStream};
use latchlm_core::{AiModel, AiProvider, AiRequest, AiResponse, BoxFuture, Error, Result};
use latchlm_macros::AiModel;

use secrecy::{ExposeSecret, SecretString};

mod response;
pub use response::*;

/// Variants representing supported Gemini models.
///
/// These variants map to the actual model identifiers used by the Gemini API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, AiModel)]
#[non_exhaustive]
pub enum GeminiModel {
    #[model(id = "gemini-2.0-flash", name = "Gemini 2.0 Flash")]
    Flash20,
    #[model(id = "gemini-2.0-flash-lite", name = "Gemini 2.0 Flash Lite")]
    Flash20Lite,
    #[model(id = "gemini-2.5-flash", name = "Gemini 2.5 Flash")]
    Flash25,
    #[model(id = "gemini-2.5-pro", name = "Gemini 2.5 Pro")]
    Pro25,
    #[model(
        id = "gemini-2.0-flash-thinking-exp-01-21",
        name = "Gemini 2.0 Flash Thinking"
    )]
    FlashThinking,
}

impl std::fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

/// Errors that can occur when building a [`Gemini`] client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiError {
    /// Returned when no API key is provided
    MissingApiKeyError,
    /// Returned when no HTTP client is provided
    MissingClientError,
}

impl From<GeminiError> for Error {
    fn from(value: GeminiError) -> Self {
        match value {
            GeminiError::MissingApiKeyError => Self::ProviderError {
                provider: "Gemini".into(),
                error: "Missing API key".into(),
            },
            GeminiError::MissingClientError => Self::ProviderError {
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
            Self::MissingApiKeyError => write!(f, "API key is required"),
            Self::MissingClientError => write!(f, "HTTP client is required"),
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a custom HTTP client
    #[must_use]
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the API key
    #[must_use]
    pub fn api_key(mut self, api_key: SecretString) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Loads the API key from the `GEMINI_API_KEY` environment variable
    pub fn api_key_from_env(mut self) -> std::result::Result<Self, std::env::VarError> {
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
    pub fn build(self) -> Result<Gemini> {
        let client = self.client.ok_or(GeminiError::MissingClientError)?;
        let api_key = self.api_key.ok_or(GeminiError::MissingApiKeyError)?;

        Ok(Gemini::new(client, api_key))
    }
}

/// A client for interacting with the Gemini API.
#[derive(Debug, Clone)]
pub struct Gemini {
    client: reqwest::Client,
    base_url: reqwest::Url,
    api_key: Arc<SecretString>,
}

impl Gemini {
    const BASE_URL: &str = "https://generativelanguage.googleapis.com";

    // The HTTP header used for authentication with the gemini api
    const X_GOOG_API_KEY: &str = "x-goog-api-key";

    /// Creates a new `Gemini` client instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to a preconfigured [`reqwest::Client`].
    /// * `api_key` - The API key wrapped in `SecretString` for secure handling
    #[allow(clippy::expect_used)]
    #[must_use]
    pub fn new(client: reqwest::Client, api_key: SecretString) -> Self {
        Self {
            client,
            base_url: reqwest::Url::parse(Self::BASE_URL).expect("Failed to parse base url"),
            api_key: Arc::new(api_key),
        }
    }

    /// Creates a new `Gemini` client instance with a custom base URL.
    ///
    /// This constructor is intended exclusively for testing and mocking scenarios
    /// and should **never** be used in production code.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to a preconfigured [`reqwest::Client`].
    /// * `base_url` - The base URL for the Gemini API.
    /// * `api_key` - The API key wrapped in `SecretString` for secure handling
    ///
    /// # Feature
    /// Requires the `test-utils` feature flag.
    #[cfg(feature = "test-utils")]
    #[must_use]
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

    /// Creates a new [`GeminiBuilder`] instance.
    #[must_use]
    pub fn builder() -> GeminiBuilder {
        GeminiBuilder::new()
    }

    /// Sends a request to the Gemini API to generate content.
    ///
    /// This method constructs a request to Google's Gemini API, handles authentication
    /// and returns the parsed response containing the generated content.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for the request.
    /// * `request` - The request to send.
    ///
    /// # Returns
    ///
    /// Returns a [`GeminiResponse`] if the request is successful.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if:
    /// - The HTTP request fails (network issues, timeout, etc.)
    /// - The API returns a non-success status code
    /// - The response body cannot be parsed as valid JSON
    /// - The API key is invalid or missing required permissions
    ///
    /// # Example
    ///
    /// ```toml
    /// [dependencies]
    /// latchlm = { version = "*", features = ["gemini"] }
    /// secrecy = "*"
    /// reqwest = "*"
    /// tokio = { version = "*", features = ["full"] }
    /// ```
    ///
    /// ```no_run
    /// use secrecy::SecretString;
    /// use latchlm_core::AiRequest;
    /// use latchlm_gemini::{Gemini, GeminiModel};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let gemini = Gemini::builder()
    ///         .client(reqwest::Client::new())
    ///         .api_key(SecretString::new("your-api-key".into()))
    ///         .build()?;
    ///
    ///     let response = gemini.request(
    ///         GeminiModel::Flash25,
    ///         AiRequest {
    ///             text: "Hello".into(),
    ///         }
    ///     ).await?;
    ///
    ///     println!("{}", response.extract_text());
    ///     Ok(())
    /// }
    /// ```
    /// [`Error`]: latchlm_core::Error
    #[allow(clippy::expect_used)]
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub async fn request(&self, model: GeminiModel, request: AiRequest) -> Result<GeminiResponse> {
        let url = self
            .base_url
            .join(&format!(
                "/v1beta/models/{}:generateContent",
                model.as_ref()
            ))
            .expect("Failed to parse the URL");

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            Self::X_GOOG_API_KEY,
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
            #[cfg(feature = "tracing")]
            tracing::error!("API error: {}", response.status());

            return Err(Error::ApiError {
                status: response.status().as_u16(),
                message: response.text().await?,
            });
        }

        let bytes = response.bytes().await?;

        #[cfg(feature = "tracing")]
        tracing::debug!("Received bytes: {:?}", bytes);

        let response: GeminiResponse = serde_json::from_slice(&bytes)?;

        Ok(response)
    }

    /// Sends a streaming request to the Gemini API and prints chunks as they arrive.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for the request.
    /// * `request` - The request to send.
    #[allow(clippy::expect_used)]
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub async fn streaming_request(
        &self,
        model: GeminiModel,
        request: AiRequest,
    ) -> Result<BoxStream<'_, Result<GeminiResponse>>> {
        let url = self
            .base_url
            .join(&format!(
                "/v1beta/models/{}:streamGenerateContent?alt=sse",
                model.as_ref()
            ))
            .expect("Failed to parse URL");

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            Self::X_GOOG_API_KEY,
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

        // WARNING: this implementation assumes that each event from the API
        // is delivered as a single line starting with 'data: ', followed by a complete JSON object.
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event| async {
                let event = match event {
                    Ok(event) => {
                        #[cfg(feature = "tracing")]
                        tracing::debug!("Received event: {:?}", event);

                        event
                    }
                    Err(err) => {
                        #[cfg(feature = "tracing")]
                        tracing::error!("Error receiving event: {:?}", err);

                        return Some(Err(Error::ProviderError {
                            provider: "Gemini".to_string(),
                            error: err.to_string(),
                        }));
                    }
                };

                let data = event.data;

                Some(serde_json::from_str::<GeminiResponse>(&data).map_err(Into::into))
            });

        Ok(Box::pin(stream))
    }
}

impl AiProvider for Gemini {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, model)))]
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        let Some(model) = model.downcast::<GeminiModel>() else {
            let model_name = model.as_ref().to_owned();

            #[cfg(feature = "tracing")]
            tracing::error!("Invalid model type: {}", model_name);

            return Box::pin(ready(Err(Error::InvalidModelError(model_name))));
        };

        Box::pin(async move { self.request(model, request).await.map(Into::into) })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, model)))]
    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>> {
        let Some(model) = model.downcast::<GeminiModel>() else {
            let model_name = model.as_ref().to_owned();

            #[cfg(feature = "tracing")]
            tracing::error!("Invalid model type: {}", model_name);

            return Box::pin(futures::stream::once(async move {
                Err(Error::InvalidModelError(model_name))
            }));
        };

        Box::pin(
            async move {
                match self.streaming_request(model, request).await {
                    Ok(stream) => stream.map(|res| res.map(Into::into)).boxed(),
                    Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
                }
            }
            .flatten_stream(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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
                .map(|v| v.id.clone())
                .collect();

            prop_assume!(!valid_ids.contains(&model_str.clone().into()));

            let err = GeminiModel::try_from(model_str.as_str()).unwrap_err();
            prop_assert_eq!(err.to_string(), format!("Invalid model name: {}", model_str));
        }
    }
}

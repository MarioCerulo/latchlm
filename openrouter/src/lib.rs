// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! Provider implementation for OpenRouter.
//!
//! This crate implements a client for interacting with the OpenRouter API.

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use latchlm_core::{AiModel, AiProvider, AiRequest, AiResponse, BoxFuture, Error, ModelId, Result};
use reqwest::{Client, Url};
use secrecy::{ExposeSecret, SecretString};
use std::{borrow::Cow, env::VarError, future::ready, sync::Arc};

mod response;
pub use response::*;

/// OpenRouter model identifier.
#[derive(Debug, Clone)]
pub struct OpenrouterModel(String);

impl AsRef<str> for OpenrouterModel {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl AiModel for OpenrouterModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn model_id(&self) -> ModelId<'_> {
        ModelId {
            id: Cow::Borrowed(&self.0),
            name: Cow::Borrowed(&self.0),
        }
    }
}

impl OpenrouterModel {
    pub fn new<S: Into<String>>(model_name: S) -> Self {
        Self(model_name.into())
    }
}

/// Errors that can occur while using the [`Openrouter`] client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenrouterError {
    MissingClientError,
    MissingApiKeyError,
    HeaderParseError(String),
}

impl std::fmt::Display for OpenrouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenrouterError::MissingClientError => write!(f, "HTTP client is required"),
            OpenrouterError::MissingApiKeyError => write!(f, "API key is required"),
            OpenrouterError::HeaderParseError(err) => write!(f, "Failed to parse header: {err}"),
        }
    }
}

impl From<OpenrouterError> for Error {
    fn from(value: OpenrouterError) -> Self {
        match value {
            OpenrouterError::MissingClientError => Error::ProviderError {
                provider: "OpenRouter".to_owned(),
                error: "Missing reqwest::Client".to_owned(),
            },
            OpenrouterError::MissingApiKeyError => Error::ProviderError {
                provider: "OpenRouter".to_owned(),
                error: "Missing API key".to_owned(),
            },
            OpenrouterError::HeaderParseError(err) => Error::ProviderError {
                provider: "OpenRouter".to_owned(),
                error: format!("Failed to parse header: {err}"),
            },
        }
    }
}

impl std::error::Error for OpenrouterError {}

/// A builder for creating an [`Openrouter`] client.
#[derive(Debug, Clone, Default)]
pub struct OpenrouterBuilder {
    client: Option<Client>,
    api_key: Option<SecretString>,
    http_referer: Option<String>,
    x_title: Option<String>,
}

impl OpenrouterBuilder {
    /// Creates a new OpenRouter client builder.
    ///
    /// # Returns
    /// An new [`OpenrouterBuilder`] instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the HTTP client to use for making requests.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use.
    ///
    /// # Returns
    ///
    /// The updated [`OpenrouterBuilder`] instance.
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the API key to use for authentication.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key to use.
    ///
    /// # Returns
    ///
    /// The updated [`OpenrouterBuilder`] instance.
    pub fn api_key(mut self, api_key: SecretString) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Sets the API key to be used by the [`Openrouter`] client from the environment variable `OPENROUTER_API_KEY`.
    pub fn api_key_from_env(mut self) -> std::result::Result<Self, VarError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")?;

        self.api_key = Some(SecretString::from(api_key));
        Ok(self)
    }

    /// Sets the `HTTP-Referer` header to be used by the [`Openrouter`] client.
    ///
    /// # Arguments
    ///
    /// * `http_referer` - The `HTTP-Referer` header to use.
    ///
    /// # Returns
    ///
    /// The updated [`OpenrouterBuilder`] instance.
    pub fn http_referer(mut self, http_referer: String) -> Self {
        self.http_referer = Some(http_referer);
        self
    }

    /// Sets the `X-Title` header to be used by the [`Openrouter`] client.
    ///
    /// # Arguments
    ///
    /// * `x_title` - The `X-Title` header to use.
    ///
    /// # Returns
    ///
    /// The updated [`OpenrouterBuilder`] instance.
    pub fn x_title(mut self, x_title: String) -> Self {
        self.x_title = Some(x_title);
        self
    }

    /// Builds the [`Openrouter`] client.
    ///
    /// # Returns
    ///
    /// The [`Openrouter`] client.
    pub fn build(self) -> Result<Openrouter> {
        let client = self.client.ok_or(OpenrouterError::MissingClientError)?;
        let api_key = self.api_key.ok_or(OpenrouterError::MissingApiKeyError)?;

        Ok(Openrouter::new(
            client,
            api_key,
            self.http_referer,
            self.x_title,
        ))
    }
}

/// A client for the OpenRouter API.
#[derive(Debug, Clone)]
pub struct Openrouter {
    base_url: Url,
    client: Client,
    api_key: Arc<SecretString>,
    http_referer: Option<String>,
    x_title: Option<String>,
}

impl Openrouter {
    const BASE_URL: &str = "https://openrouter.ai/api/v1/";

    /// Creates a new [`Openrouter`] client.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use.
    /// * `api_key` - The API key to use.
    /// * `http_referer` - The HTTP referer to use.
    /// * `x_title` - The X-Title header to use.
    ///
    /// # Returns
    ///
    /// The [`Openrouter`] client.
    pub fn new(
        client: Client,
        api_key: SecretString,
        http_referer: Option<String>,
        x_title: Option<String>,
    ) -> Self {
        Self {
            base_url: Url::parse(Self::BASE_URL).expect("Invalid base URL"),
            client,
            api_key: Arc::new(api_key),
            http_referer,
            x_title,
        }
    }

    /// Creates a new [`Openrouter`] client with a custom base URL for testing.
    ///
    /// This constructor is intended for testing and mocking scenarios and should **never**
    /// be used in production code.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use.
    /// * `base_url` - The base URL to use.
    /// * `api_key` - The API key to use.
    ///
    /// # Returns
    ///
    /// The [`Openrouter`] client.
    #[cfg(feature = "test-utils")]
    pub fn new_with_base_url(client: Client, base_url: Url, api_key: SecretString) -> Self {
        Self {
            base_url,
            client,
            api_key: Arc::new(api_key),
            http_referer: None,
            x_title: None,
        }
    }

    /// Creates a new [`Openrouter`] client builder.
    pub fn builder() -> OpenrouterBuilder {
        OpenrouterBuilder::new()
    }

    /// Sends a request to the OpenRouter API to generate content.
    ///
    /// This method constructs a request to OpenRouter's API, handles authentication
    /// and returns the parsed response containing the generated content.
    ///
    /// # Arguments
    ///
    /// * `model` - The [`OpenrouterModel`] to use for the request.
    /// * `request` - The [`AiRequest`] containing the request prompt and settings to send to the API.
    ///
    /// # Returns
    ///
    /// The [`OpenrouterResponse`] containing the generated content.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if:
    /// - The HTTP request fails (network issues, timeouts, etc.)
    /// - The API returns a non-success status code
    /// - The response body cannot be parsed as valid JSON
    /// - The API key is invalid or missing required permissions
    ///
    /// # Example
    ///
    /// ```toml
    /// [dependencies]
    /// latchlm_core = "*"
    /// latchlm_openrouter = "*"
    /// secrecy = "*"
    /// tokio = { version = "*", features = ["full"] }
    /// ```
    ///
    /// ```no_run
    /// use latchlm_core::AiRequest;
    /// use latchlm_openrouter::{Openrouter, OpenrouterModel};
    /// use secrecy::SecretString;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let openrouter = Openrouter::builder()
    ///         .client(reqwest::Client::new())
    ///         .api_key(SecretString::new("your-api-key".into()))
    ///         .build()?;
    ///
    ///     let response = openrouter.request(
    ///         &OpenrouterModel::new("openai/gpt-oss-20b"),
    ///         AiRequest {
    ///             text: "Hello".into(),
    ///         }
    ///     ).await?;
    ///
    ///     println!("Generated: {}", response.extract_text());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`AiRequest`]: latchlm_core::AiRequest
    /// [`Error`]: latchlm_core::Error
    pub async fn request(
        &self,
        model: OpenrouterModel,
        request: AiRequest,
    ) -> Result<OpenrouterResponse> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .expect("Failed to parse content-type"),
        );

        if let Some(http_referer) = &self.http_referer {
            headers.insert(
                "HTTP-Referer",
                http_referer.parse().expect("Failed to parse http-referer"),
            );
        }

        if let Some(x_title) = &self.x_title {
            headers.insert("X-Title", x_title.parse().expect("Failed to parse x-title"));
        }

        let request = serde_json::json!({
            "model": model.as_ref(),
            "messages": [
                {
                    "role": "user",
                    "content": request.text
                }
            ],
        });

        let url = self
            .base_url
            .join("chat/completions")
            .expect("Failed to join URL");

        let response = self
            .client
            .post(url)
            .headers(headers)
            .bearer_auth(self.api_key.expose_secret())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError {
                status: response.status().as_u16(),
                message: response.text().await?,
            });
        }

        let bytes = response.bytes().await?;

        let response = serde_json::from_slice(&bytes)?;

        Ok(response)
    }

    /// Sends a streaming request to the OpenRouter and returns a stream of responses.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for the request.
    /// * `request` - The request to send.
    pub async fn streaming_request(
        &self,
        model: OpenrouterModel,
        request: AiRequest,
    ) -> Result<BoxStream<'_, Result<OpenrouterStreamResponse>>> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .expect("Failed to parse content-type"),
        );

        if let Some(http_referer) = &self.http_referer {
            headers.insert(
                "HTTP-Referer",
                http_referer.parse().expect("Failed to parse http-referer"),
            );
        }

        if let Some(x_title) = &self.x_title {
            headers.insert("X-Title", x_title.parse().expect("Failed to parse x-title"));
        }

        let request = serde_json::json!({
            "model": model.as_ref(),
            "messages": [
                {
                    "role": "user",
                    "content": request.text
                }
            ],
            "stream": true
        });

        let url = self
            .base_url
            .join("chat/completions")
            .expect("Failed to join URL");

        let response = self
            .client
            .post(url)
            .headers(headers)
            .bearer_auth(self.api_key.expose_secret())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError {
                status: response.status().as_u16(),
                message: response.text().await?,
            });
        }

        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|result| async {
                let event = match result {
                    Ok(event) => event,
                    Err(err) => {
                        return Some(Err(Error::ProviderError {
                            provider: "OpenRouter".to_string(),
                            error: err.to_string(),
                        }));
                    }
                };
                let data = event.data;

                if data.contains("[DONE]") {
                    return None;
                }

                Some(serde_json::from_str::<OpenrouterStreamResponse>(&data).map_err(Into::into))
            });

        Ok(Box::pin(stream))
    }

    /// Returns a list of available models.
    ///
    /// This function fetches the list of available models from the OpenRouter API.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if:
    /// - The API request fails.
    /// - The response is not successful.
    /// - The response cannot be parsed.
    ///
    /// [`Error`]: latchlm_core::Error
    pub async fn models(&self) -> Result<Vec<ModelId<'_>>> {
        let url = self.base_url.join("models").expect("Failed to join URL");
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(Error::ApiError {
                status: response.status().as_u16(),
                message: response.text().await?,
            });
        }

        let response: ModelsList = response.json().await?;

        Ok(response.into())
    }
}

impl AiProvider for Openrouter {
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        let Some(model) = model.downcast::<OpenrouterModel>() else {
            let model_name = model.as_ref();
            return Box::pin(ready(Err(Error::InvalidModelError(model_name.into()))));
        };

        let model = model.clone();
        Box::pin(async move { self.request(model, request).await.map(Into::into) })
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module implements a client for interacting with the OpenAI API,
//! including support for model variants and structured error handling.

use std::{future::ready, sync::Arc};

use latchlm_core::{AiModel, AiProvider, AiRequest, Error};
use latchlm_macros::AiModel;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};

mod response;
pub use response::*;

/// Variants representing supported OpenAI models.
///
/// These variants map to the actual model identifiers used by the OpenAI API.
#[derive(Debug, Clone, PartialEq, Eq, Hash, AiModel)]
pub enum OpenaiModel {
    #[model(id = "o3", name = "GPT-o3")]
    Gpto3,
    #[model(id = "o3-pro", name = "GPT-o3 Pro")]
    Gpto3Pro,
    #[model(id = "o3-mini", name = "GPT-o3 Mini")]
    Gpto3Mini,
    #[model(id = "o4-mini", name = "GPT-o4 Mini")]
    Gpto4Mini,
    #[model(id = "gpt-5", name = "GPT-5")]
    Gpt5,
    #[model(id = "gpt-5-mini", name = "GPT-5 Mini")]
    Gpt5Mini,
    #[model(id = "gpt-5-nano", name = "GPT-5 Nano")]
    Gpt5Nano,
    #[model(id = "gpt-5-chat-latest", name = "GPT-5 Chat")]
    Gpt5Chat,
    #[model(id = "gpt-4.1", name = "GPT-4.1")]
    Gpt41,
    #[model(id = "gpt-4.1-mini", name = "GPT-4.1 Mini")]
    Gpt41Mini,
    #[model(id = "gpt-4.1-nano", name = "GPT-4.1 Nano")]
    Gpt41Nano,
    #[model(id = "gpt-4o", name = "GPT-4o")]
    Gpt4o,
    #[model(id = "gpt-4o-mini", name = "GPT-4o Mini")]
    Gpt4oMini,
}

impl std::fmt::Display for OpenaiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

/// Errors that can occur when building a [`Openai`] client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenaiError {
    /// Returned when no HTTP client is provided
    MissingClientError,
    /// Returned when no API key is provided
    MissingApiKeyError,
}

impl std::fmt::Display for OpenaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenaiError::MissingClientError => write!(f, "HTTP client is required"),
            OpenaiError::MissingApiKeyError => write!(f, "API key is required"),
        }
    }
}

impl From<OpenaiError> for Error {
    fn from(value: OpenaiError) -> Self {
        match value {
            OpenaiError::MissingApiKeyError => Error::ProviderError {
                provider: "OpenAI".into(),
                error: "Missing API Key".into(),
            },
            OpenaiError::MissingClientError => Error::ProviderError {
                provider: "OpenAI".into(),
                error: "Missing request::Client".into(),
            },
        }
    }
}

/// A builder for creating an [`Openai`] client.
#[derive(Debug, Clone, Default)]
pub struct OpenaiBuilder {
    client: Option<reqwest::Client>,
    api_key: Option<SecretString>,
}

impl OpenaiBuilder {
    /// Create a new OpenAI builder.
    ///
    /// # Returns
    ///
    /// A new `OpenaiBuilder` instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTTP client to use for making requests.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use for making requests.
    ///
    /// # Returns
    ///
    /// The updated `OpenaiBuilder` instance.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Set the API key to use for authentication.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key to use for authentication.
    ///
    /// # Returns
    ///
    /// The updated `OpenaiBuilder` instance.
    pub fn api_key(mut self, api_key: SecretString) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Loads the API key from the `OPENAI_API_KEY` environment variable.
    pub fn api_key_from_env(mut self) -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("OPENAI_API_KEY")?;

        self.api_key = Some(SecretString::from(api_key));
        Ok(self)
    }

    /// Build the OpenAI client.
    ///
    /// # Returns
    ///
    /// A new `Openai` client.
    pub fn build(self) -> latchlm_core::Result<Openai> {
        let client = self.client.ok_or(OpenaiError::MissingClientError)?;
        let api_key = self.api_key.ok_or(OpenaiError::MissingApiKeyError)?;
        Ok(Openai::new(client, api_key))
    }
}

/// A client for interacting with the OpenAI API.
#[derive(Debug, Clone)]
pub struct Openai {
    client: reqwest::Client,
    base_url: reqwest::Url,
    api_key: Arc<SecretString>,
}

impl Openai {
    const BASE_URL: &str = "https://api.openai.com/v1/responses";

    /// Create a new OpenAI client.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use for making requests.
    /// * `api_key` - The API key to use for authentication.
    ///
    /// # Returns
    ///
    /// A new `Openai` client.
    pub fn new(client: reqwest::Client, api_key: SecretString) -> Self {
        Self {
            client,
            base_url: reqwest::Url::parse(Self::BASE_URL).expect("Failed to parse base URL"),
            api_key: Arc::new(api_key),
        }
    }

    /// Create a new OpenAI client with a custom base URL for testing.
    ///
    /// This constructor is intended exclusively for testing and mocking scenarios
    /// and should **never** be used in production code.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use for making requests.
    /// * `base_url` - The base URL to use for making requests.
    /// * `api_key` - The API key to use for authentication.
    ///
    /// # Returns
    ///
    /// A new `Openai` client.
    #[cfg(feature = "test-utils")]
    pub fn new_with_base_url(
        client: reqwest::Client,
        base_url: reqwest::Url,
        api_key: SecretString,
    ) -> Self {
        Self {
            client,
            api_key: Arc::new(api_key),
            base_url,
        }
    }

    /// Create a new [`OpenaiBuilder`] instance.
    pub fn builder() -> OpenaiBuilder {
        OpenaiBuilder::new()
    }

    /// Sends a request to the OpenAI API to generate content.
    ///
    /// This method constructs a request to OpenAI's API, handles authentication,
    /// and returns the parsed response containing the generated content.
    ///
    /// # Arguments
    ///
    /// * `model` - The OpenAI model to use for content generation
    /// * `request` - The input request containing the text prompt
    ///
    /// # Returns
    ///
    /// Returns an [`OpenaiResponse`] containing the generated content and metadata
    /// from the API response.
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
    /// latchlm_core = "*"
    /// latchlm_openai = "*"
    /// secrecy = "*"
    /// tokio = { version = "*", features = ["full"] }
    /// ```
    ///
    /// ```no_run
    /// use latchlm_core::AiRequest;
    /// use latchlm_openai::{Openai, OpenaiModel};
    /// use secrecy::SecretString;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let openai = Openai::builder()
    ///         .client(reqwest::Client::new())
    ///         .api_key(SecretString::new("your-api-key".into()))
    ///         .build()?;
    ///
    ///     let response = openai.request(
    ///         OpenaiModel::Gpt4o,
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
    /// [`Error`]: latchlm_core::Error
    pub async fn request(
        &self,
        model: OpenaiModel,
        request: AiRequest,
    ) -> latchlm_core::Result<OpenaiResponse> {
        let mut header_map = HeaderMap::new();
        header_map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        header_map.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .expect("Invalid header value for Authorization"),
        );

        let request = serde_json::json!({"model": model.as_ref(), "input": request.text});

        let response = self
            .client
            .post(self.base_url.clone())
            .headers(header_map)
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

        let response: OpenaiResponse = serde_json::from_slice(&bytes)?;

        Ok(response)
    }
}

impl AiProvider for Openai {
    fn send_request(
        &self,
        model: &dyn latchlm_core::AiModel,
        request: AiRequest,
    ) -> latchlm_core::BoxFuture<'_, latchlm_core::Result<latchlm_core::AiResponse>> {
        let Ok(model) = model.as_ref().parse() else {
            let model_name = model.as_ref();
            return Box::pin(ready(Err(Error::InvalidModelError(model_name.into()))));
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
        fn test_openai_model_try_from_valid(model in prop_oneof![
            Just(OpenaiModel::Gpto3),
            Just(OpenaiModel::Gpto3Pro),
            Just(OpenaiModel::Gpto3Mini),
            Just(OpenaiModel::Gpto4Mini),
            Just(OpenaiModel::Gpt5),
            Just(OpenaiModel::Gpt5Mini),
            Just(OpenaiModel::Gpt5Nano),
            Just(OpenaiModel::Gpt5Chat),
            Just(OpenaiModel::Gpt41),
            Just(OpenaiModel::Gpt41Mini),
            Just(OpenaiModel::Gpt41Nano),
            Just(OpenaiModel::Gpt4o),
            Just(OpenaiModel::Gpt4oMini),
        ]) {
            let model_str = model.as_ref();
            let parsed_model = OpenaiModel::try_from(model_str).unwrap();
            prop_assert_eq!(model, parsed_model);
        }

        #[test]
        fn test_openai_model_try_from_invalid(model_str in "\\PC*") {
            let valid_ids: Vec<_> = OpenaiModel::variants()
                .iter()
                .map(|v| v.id.clone())
                .collect();

            prop_assume!(!valid_ids.contains(&model_str.clone().into()));

            let err = OpenaiModel::try_from(model_str.as_str()).unwrap_err();
            prop_assert_eq!(err.to_string(), format!("Invalid model name: {}", model_str));
        }
    }
}

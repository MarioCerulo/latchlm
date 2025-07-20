// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module implements a client for interacting with the Gemini API,
//! including support for model variants and structured error handling.

use std::{future::ready, sync::Arc};

use latchlm_core::{AiModel, AiProvider, AiRequest, AiResponse, BoxFuture, Error, ModelId, Result};

use reqwest::header::HeaderValue;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::GeminiResponse;

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
    fn try_from(value: &str) -> Result<Self> {
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

    /// Get a `Vec` of all the variants of the model
    pub fn variants() -> Vec<ModelId> {
        Self::iter().map(|variant| variant.model_id()).collect()
    }
}

/// A client for interacting with the Gemini API.
pub struct Gemini {
    client: reqwest::Client,
    base_url: String,
    api_key: Arc<SecretString>,
}

// The HTTP header used for authentication with the gemini api
static X_GOOG_API_KEY: &str = "x-goog-api-key";

impl Gemini {
    /// Create a new `Gemini` client instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to a preconfigured [`reqwest::Client`].
    /// * `base_url` - The base url for the gemini API.
    /// * `api_key` - The API key wrapped in `SecretString` for secure handling
    pub fn new(client: reqwest::Client, base_url: &str, api_key: SecretString) -> Self {
        Self {
            client,
            base_url: base_url.to_owned(),
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
    pub async fn request(&self, model: GeminiModel, request: AiRequest) -> Result<GeminiResponse> {
        let url = format!("{}/{}:generateContent", self.base_url, model.as_ref());
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            X_GOOG_API_KEY,
            HeaderValue::from_str(self.api_key.expose_secret()).expect("Failed to parse header"),
        );

        let payload = serde_json::json!({"contents": [{"parts": {"text": request.text}}]});

        let response = self
            .client
            .post(&url)
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
    ) -> BoxFuture<'_, Result<AiResponse>> {
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

    #[test]
    fn test_gemini_model_try_from_valid() {
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash").unwrap(),
            GeminiModel::Flash20
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.5-pro").unwrap(),
            GeminiModel::Pro25
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash-lite").unwrap(),
            GeminiModel::Flash20Lite
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash-thinking-exp-01-21").unwrap(),
            GeminiModel::FlashThinking
        );
    }

    #[test]
    fn test_gemini_model_try_from_invalid() {
        let err = GeminiModel::try_from("nonexistent-model").unwrap_err();
        assert_eq!(err.to_string(), "Invalid model name: nonexistent-model");
    }
}

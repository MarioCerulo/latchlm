// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module implements a client for interacting with the Gemini API,
//! including support for model variants and structured error handling.

use std::sync::Arc;

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
    Flash,
    FlashPro,
    FlashLite,
    FlashThinking,
    Flash25Preview,
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
            "gemini-2.0-flash" => Ok(Self::Flash),
            "gemini-2.0-flash-lite" => Ok(Self::FlashLite),
            "gemini-2.0-flash-thinking-exp-01-21" => Ok(Self::FlashThinking),
            "gemini-2.5-pro-exp-03-25" => Ok(Self::FlashPro),
            "gemini-2.5-flash-preview-04-17" => Ok(Self::Flash25Preview),
            invalid_model => Err(Error::InvalidModelError(invalid_model.to_owned())),
        }
    }
}

impl AsRef<str> for GeminiModel {
    fn as_ref(&self) -> &str {
        match self {
            GeminiModel::Flash => "gemini-2.0-flash",
            GeminiModel::FlashPro => "gemini-2.5-pro-exp-03-25",
            GeminiModel::FlashLite => "gemini-2.0-flash-lite",
            GeminiModel::FlashThinking => "gemini-2.0-flash-thinking-exp-01-21",
            GeminiModel::Flash25Preview => "gemini-2.5-flash-preview-04-17",
        }
    }
}
impl AiModel for GeminiModel {}

impl GeminiModel {
    pub fn model_id(&self) -> ModelId {
        match self {
            Self::Flash => ModelId {
                id: "gemini-2.0-flash".to_string(),
                name: "Gemini 2.0 Flash".to_string(),
            },
            Self::FlashPro => ModelId {
                id: "gemini-2.5-pro-exp-03-25".to_string(),
                name: "Gemini 2.5 Pro".to_string(),
            },
            Self::FlashLite => ModelId {
                id: "gemini-2.0-flash-lite".to_string(),
                name: "Gemini 2.0 Flash Lite".to_string(),
            },
            Self::FlashThinking => ModelId {
                id: "gemini-2.0-flash-thinking-exp-01-21".to_string(),
                name: "Gemini 2.0 Flash Thinking".to_string(),
            },
            Self::Flash25Preview => ModelId {
                id: "gemini-2.5-flash-preview-04-17".to_string(),
                name: "Gemini 2.5 Flash Preview".to_string(),
            },
        }
    }

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

impl Gemini {
    /// Create a new `Gemini` client instance.
    ///
    /// # Arguments
    ///
    /// * `client` - A reference to a preconfigured `reqwest::Client`.
    /// * `base_url` - The base url for the gemini API.
    /// * `api_key` - The API key wrapped in `SecretString` for secure handling
    pub fn new(client: reqwest::Client, base_url: &str, api_key: SecretString) -> Self {
        Self {
            client,
            base_url: base_url.to_owned(),
            api_key: Arc::new(api_key),
        }
    }
}

impl AiProvider for Gemini {
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<Result<AiResponse>> {
        let url = format!("{}/{}:generateContent", self.base_url, model.as_ref());

        let client = reqwest::Client::clone(&self.client);
        let api_key = Arc::clone(&self.api_key);
        let message = request.text.to_string();

        Box::pin(async move {
            const X_GOOG_API_KEY: &str = "x-goog-api-key";
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                X_GOOG_API_KEY,
                HeaderValue::from_str(api_key.expose_secret()).expect("Failed to parse header"),
            );

            let payload = serde_json::json!({"contents": [{"parts": {"text": message}}]});

            let mut response = client
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

            let mut buffer = Vec::new();

            while let Some(chunk) = response.chunk().await? {
                buffer.extend_from_slice(&chunk);
            }

            let response: GeminiResponse = serde_json::from_slice(&buffer)?;

            Ok(AiResponse {
                text: response.extract_text(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_model_try_from_valid() {
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash").unwrap(),
            GeminiModel::Flash
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.5-pro-exp-03-25").unwrap(),
            GeminiModel::FlashPro
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash-lite").unwrap(),
            GeminiModel::FlashLite
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.0-flash-thinking-exp-01-21").unwrap(),
            GeminiModel::FlashThinking
        );
        assert_eq!(
            GeminiModel::try_from("gemini-2.5-flash-preview-04-17").unwrap(),
            GeminiModel::Flash25Preview
        );
    }

    #[test]
    fn test_gemini_model_try_from_invalid() {
        let err = GeminiModel::try_from("nonexistent-model").unwrap_err();
        assert_eq!(err.to_string(), "Invalid model name: nonexistent-model");
    }
}

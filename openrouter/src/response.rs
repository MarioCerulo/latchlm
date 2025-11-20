// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module contains the structs used to deserialize
//! the OpenRouter API responses

use latchlm_core::{AiResponse, ModelId, TokenUsage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: String,
    refusal: Option<serde_json::Value>,
    reasoning: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Choice {
    logprobs: Option<serde_json::Value>,
    finish_reason: String,
    native_finish_reason: String,
    index: u64,
    message: Message,
}

/// Represents the response from the OpenRouter API.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenrouterResponse {
    id: String,
    provider: String,
    model: String,
    object: String,
    created: u64,
    choices: Vec<Choice>,
    usage: Usage,
}

impl OpenrouterResponse {
    #[must_use]
    pub fn extract_text(&self) -> String {
        self.choices
            .iter()
            .map(|choice| choice.message.content.clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl From<OpenrouterResponse> for AiResponse {
    fn from(response: OpenrouterResponse) -> Self {
        let text = response.extract_text();

        Self {
            text,
            token_usage: TokenUsage {
                input_tokens: Some(response.usage.prompt_tokens),
                output_tokens: Some(response.usage.completion_tokens),
                total_tokens: Some(response.usage.total_tokens),
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelsItem {
    id: String,
    name: String,
}

/// Represents a list of available models.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelsList {
    data: Vec<ModelsItem>,
}

impl From<ModelsList> for Vec<ModelId<'_>> {
    fn from(value: ModelsList) -> Self {
        let mut list = vec![];
        for model in value.data {
            list.push(ModelId {
                id: model.id.into(),
                name: model.name.into(),
            });
        }
        list
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u64,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
    pub native_finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// Represents a single streaming chunk from the OpenRouter API.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenrouterStreamResponse {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

impl OpenrouterStreamResponse {
    #[must_use]
    pub fn extract_text(&self) -> String {
        self.choices
            .iter()
            .map(|choice| choice.delta.content.clone().unwrap_or_default())
            .collect::<Vec<String>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text() {
        let test_response = OpenrouterResponse {
            choices: vec![Choice {
                message: Message {
                    content: "Test Content".to_string(),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(test_response.extract_text(), "Test Content");
    }

    #[test]
    fn test_extract_text_with_multiple_choices() {
        let test_response = OpenrouterResponse {
            choices: vec![
                Choice {
                    message: Message {
                        content: "First Choice".to_string(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Choice {
                    message: Message {
                        content: "Second Choice".to_string(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_eq!(test_response.extract_text(), "First Choice Second Choice");
    }

    #[test]
    fn test_extract_text_with_empty_choices() {
        let test_response = OpenrouterResponse {
            choices: vec![],
            ..Default::default()
        };

        assert_eq!(test_response.extract_text(), "");
    }

    #[test]
    fn test_extract_text_empty_response() {
        let test_response = OpenrouterResponse::default();

        assert_eq!(test_response.extract_text(), "");
    }
}

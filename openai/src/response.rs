// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module contains the structs used to deserialize
//! the OpenAI API responses

use latchlm_core::{AiResponse, TokenUsage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Content {
    #[serde(rename = "type")]
    kind: String,
    text: String,
    annotations: Vec<String>,
    logprobs: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum Output {
    Content {
        #[serde(rename = "type")]
        kind: String,
        id: String,
        status: Option<String>,
        role: Option<String>,
        content: Vec<Content>,
    },
    Summary {
        id: String,
        summary: Vec<serde_json::Value>,
        #[serde(rename = "type")]
        kind: String,
    },
}

impl Default for Output {
    fn default() -> Self {
        Output::Content {
            kind: String::new(),
            id: String::new(),
            status: None,
            role: None,
            content: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Reasoning {
    effort: Option<String>,
    summary: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Format {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Text {
    format: Format,
    verbosity: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct InputTokensDetails {
    cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputTokensDetails {
    reasoning_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Usage {
    input_tokens: u64,
    input_tokens_details: InputTokensDetails,
    output_tokens: u64,
    output_tokens_details: OutputTokensDetails,
    total_tokens: u64,
}

/// Represents the response from the OpenAI API.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct OpenaiResponse {
    id: String,
    object: String,
    created_at: u32,
    status: String,
    background: Option<bool>,
    error: Option<String>,
    incomplete_details: Option<String>,
    instructions: Option<String>,
    max_output_tokens: Option<u64>,
    model: String,
    output: Vec<Output>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    prompt_cache_key: Option<String>,
    reasoning: Reasoning,
    safety_identifier: Option<String>,
    service_tier: Option<String>,
    store: bool,
    temperature: f32,
    text: Text,
    tool_choice: String,
    tools: Vec<String>,
    top_logprobe: Option<u64>,
    top_p: f32,
    truncation: String,
    usage: Usage,
    user: Option<String>,
    metadata: serde_json::Value,
}

impl From<OpenaiResponse> for AiResponse {
    fn from(value: OpenaiResponse) -> Self {
        Self {
            text: value.extract_text(),
            token_usage: TokenUsage {
                input_tokens: Some(value.usage.input_tokens),
                output_tokens: Some(value.usage.output_tokens),
                total_tokens: Some(value.usage.total_tokens),
            },
        }
    }
}

impl OpenaiResponse {
    pub fn extract_text(&self) -> String {
        self.output
            .iter()
            .flat_map(|output| match output {
                Output::Content {
                    content: contents, ..
                } => contents.iter().map(|c| c.text.clone()).collect::<Vec<_>>(),
                _ => vec![],
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text() {
        let response = OpenaiResponse {
            output: vec![Output::Content {
                kind: "content".to_string(),
                id: "id1".to_string(),
                status: None,
                role: None,
                content: vec![
                    Content {
                        text: "Hello".to_string(),
                        ..Default::default()
                    },
                    Content {
                        text: "World".to_string(),
                        ..Default::default()
                    },
                ],
            }],
            ..Default::default()
        };

        assert_eq!(response.extract_text(), "Hello World");
    }

    #[test]
    fn test_extract_empty_response() {
        let test_response = OpenaiResponse::default();
        assert_eq!(test_response.extract_text(), "");
    }
}

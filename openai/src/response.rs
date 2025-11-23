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
        Self::Content {
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
    input_tokens_details: Option<InputTokensDetails>,
    output_tokens: u64,
    output_tokens_details: Option<OutputTokensDetails>,
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
    error: Option<serde_json::Value>,
    incomplete_details: Option<serde_json::Value>,
    input: Option<Vec<serde_json::Value>>,
    instructions: Option<String>,
    max_output_tokens: Option<u64>,
    model: String,
    output: Vec<Output>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    prompt_cache_key: Option<String>,
    reasoning: Option<Reasoning>,
    reasoning_effort: Option<String>,
    safety_identifier: Option<String>,
    service_tier: Option<String>,
    store: bool,
    temperature: f32,
    text: Text,
    tool_choice: String,
    tools: Vec<serde_json::Value>,
    top_logprobs: Option<u64>,
    top_p: f32,
    truncation: String,
    usage: Option<Usage>,
    user: Option<serde_json::Value>,
    metadata: serde_json::Value,
}

impl From<OpenaiResponse> for AiResponse {
    fn from(value: OpenaiResponse) -> Self {
        let token_usage = TokenUsage {
            input_tokens: value.usage.as_ref().map(|usage| usage.input_tokens),
            output_tokens: value.usage.as_ref().map(|usage| usage.output_tokens),
            total_tokens: value.usage.as_ref().map(|usage| usage.total_tokens),
        };
        Self {
            text: value.extract_text(),
            token_usage,
        }
    }
}

impl OpenaiResponse {
    #[must_use]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputText {
    Delta {
        #[serde(rename = "type")]
        kind: String,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    Done {
        #[serde(rename = "type")]
        kind: String,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Part {
    #[serde(rename = "type")]
    kind: String,
    text: String,
    annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContentPart {
    Added {
        #[serde(rename = "type")]
        kind: String,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: Part,
    },
    Done {
        #[serde(rename = "type")]
        kind: String,
        item_id: u64,
        output_index: u64,
        content_index: u64,
        part: Part,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    status: Option<String>,
    role: Option<String>,
    content: Option<Vec<Content>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputItem {
    Added {
        #[serde(rename = "type")]
        kind: String,
        output_index: u64,
        item: Item,
    },
    Done {
        #[serde(rename = "type")]
        kind: String,
        output_index: u64,
        item: Item,
    },
}

/// Represents a single streaming chunk from the OpenAI API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenaiStreamResponse {
    #[serde(rename = "response.created")]
    ResponseCreated {
        response: OpenaiResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        response: OpenaiResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: OpenaiResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        response_id: Option<String>,
        output_index: u64,
        item: Item,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        response_id: Option<String>,
        output_index: u64,
        item: Item,
        sequence_number: u64,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        sequence_number: u64,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: Part,
        sequence_number: u64,
    },
    #[serde(rename = "response.text.delta")]
    TextDelta {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.text.done")]
    TextDone {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        response_id: Option<String>,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
        sequence_number: u64,
    },
}

impl From<OpenaiStreamResponse> for AiResponse {
    fn from(response: OpenaiStreamResponse) -> Self {
        match response {
            OpenaiStreamResponse::TextDelta { delta, .. }
            | OpenaiStreamResponse::OutputTextDelta { delta, .. } => Self {
                text: delta,
                token_usage: TokenUsage::default(),
            },
            OpenaiStreamResponse::ResponseCompleted { response, .. } => Self {
                text: "".to_string(),
                token_usage: TokenUsage {
                    input_tokens: response.usage.as_ref().map(|usage| usage.input_tokens),
                    output_tokens: response.usage.as_ref().map(|usage| usage.output_tokens),
                    total_tokens: response.usage.as_ref().map(|usage| usage.total_tokens),
                },
            },
            _ => Self {
                text: String::new(),
                token_usage: TokenUsage::default(),
            },
        }
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

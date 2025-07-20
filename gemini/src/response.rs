// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module contains the structs used to deserialize
//! the Gemini API responses

use latchlm_core::{AiResponse, TokenUsage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Text {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub parts: Vec<Text>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    content: Content,
    finish_reason: String,
    index: Option<u64>,
    avg_log_probs: Option<i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
struct PromptTokensDetails {
    modality: String,
    token_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
struct CandidatesTokensDetails {
    modality: String,
    token_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    prompt_token_count: u64,
    candidates_token_count: u64,
    total_token_count: u64,
    prompt_tokens_details: Vec<PromptTokensDetails>,
    thoughts_token_count: Option<u64>,
    candidates_tokens_details: Option<Vec<CandidatesTokensDetails>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    candidates: Vec<Candidate>,
    usage_metadata: UsageMetadata,
    model_version: String,
    response_id: String,
}

impl From<GeminiResponse> for AiResponse {
    fn from(value: GeminiResponse) -> AiResponse {
        AiResponse {
            text: value.extract_text(),
            token_usage: TokenUsage {
                input_tokens: Some(value.usage_metadata.prompt_token_count),
                output_tokens: Some(value.usage_metadata.candidates_token_count),
                total_tokens: Some(value.usage_metadata.total_token_count),
            },
        }
    }
}

impl GeminiResponse {
    pub fn extract_text(&self) -> String {
        self.candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts
                    .iter()
                    .map(|text| text.text.as_str())
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_extract_text() {
        let test_response = GeminiResponse {
            candidates: vec![
                Candidate {
                    content: Content {
                        parts: vec![
                            Text {
                                text: "First part. ".to_string(),
                            },
                            Text {
                                text: "Second part.".to_string(),
                            },
                        ],
                    },
                    ..Default::default()
                },
                Candidate {
                    content: Content {
                        parts: vec![Text {
                            text: "Another candidate.".to_string(),
                        }],
                    },
                    finish_reason: String::new(),
                    index: Some(0),
                    avg_log_probs: None,
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            test_response.extract_text(),
            "First part. Second part.Another candidate."
        );
    }

    #[test]
    fn test_extract_text_empty_response() {
        let test_response = GeminiResponse::default();

        assert_eq!(test_response.extract_text(), "");
    }

    #[test]
    fn test_extract_text_empty_parts() {
        let test_response = GeminiResponse {
            candidates: vec![Candidate {
                content: Content { parts: vec![] },
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(test_response.extract_text(), "");
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! This module contains the structs used to deserialize
//! the Gemini API responses

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Text {
    text: String,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    parts: Vec<Text>,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    content: Content,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    candidates: Vec<Candidate>,
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
                },
                Candidate {
                    content: Content {
                        parts: vec![Text {
                            text: "Another candidate.".to_string(),
                        }],
                    },
                },
            ],
        };

        assert_eq!(
            test_response.extract_text(),
            "First part. Second part.Another candidate."
        );
    }

    #[test]
    fn test_extract_text_empty_response() {
        let test_response = GeminiResponse { candidates: vec![] };

        assert_eq!(test_response.extract_text(), "");
    }

    #[test]
    fn test_extract_text_empty_parts() {
        let test_response = GeminiResponse {
            candidates: vec![Candidate {
                content: Content { parts: vec![] },
            }],
        };

        assert_eq!(test_response.extract_text(), "");
    }
}

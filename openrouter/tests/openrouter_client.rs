// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use latchlm_core::{AiProvider, AiRequest, Error};
use latchlm_openrouter::{Openrouter, OpenrouterModel};
use secrecy::{ExposeSecret, SecretString};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{bearer_token, body_partial_json, method},
};

#[tokio::test]
async fn test_request_response() {
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let model = OpenrouterModel::new("openai/gpt-oss-20b:free");
    let mock_response_body = serde_json::json!({
        "id": "gen-123",
        "provider": "Google AI Studio",
        "model": "google/gemma-3n-e2b-it:free",
        "object": "chat.completion",
        "created": 1754828429,
        "choices": [
            {
                "logprobs": null,
                "finish_reason": "stop",
                "native_finish_reason": "STOP",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Artificial intelligence refers to systems that can perform tasks usually needing human intelligence, such as visual perception, language understanding, decision-making, and problem solving.",
                    "refusal": null,
                    "reasoning": null
                }
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 85,
            "total_tokens": 93
        }
    });

    let test_api_key = SecretString::from("test-api-key");

    let _mock_guard = Mock::given(method("POST"))
        .and(bearer_token(test_api_key.expose_secret()))
        .and(body_partial_json(
            serde_json::json!({"model": "openai/gpt-oss-20b:free"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_body))
        .expect(1)
        .mount(&mock_server)
        .await;

    let test_client = Openrouter::new_with_base_url(
        reqwest::Client::new(),
        mock_base_url.parse().expect("Failed to parse URL"),
        test_api_key,
    );

    let response = test_client
        .send_request(
            &model,
            AiRequest {
                text: "Test Message".to_owned(),
            },
        )
        .await
        .expect("Failed to send request");

    let expected = "Artificial intelligence refers to systems that can perform tasks usually needing human intelligence, such as visual perception, language understanding, decision-making, and problem solving.";
    assert_eq!(response.text, expected);
}

#[tokio::test]
async fn test_error_unauthenticated() {
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let model = OpenrouterModel::new("openai/gpt-oss-20b:free");

    let _mock_guard = Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": {
                "message": "No auth credentials found",
                "code": 401
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let test_client = Openrouter::new_with_base_url(
        reqwest::Client::new(),
        mock_base_url.parse().expect("Failed to parse URL"),
        SecretString::from("api-key"),
    );

    let err = test_client
        .send_request(
            &model,
            AiRequest {
                text: "Test Message".to_owned(),
            },
        )
        .await
        .expect_err("Expected error");

    match err {
        Error::ApiError { status, message } => {
            assert_eq!(status, 401);
            assert!(message.contains("No auth credentials found"));
        }
        _ => panic!("Unexpected error"),
    }
}

#[tokio::test]
async fn test_error_invalid_model() {
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let invalid_model = OpenrouterModel::new("invalid/model");
    let test_client = Openrouter::new_with_base_url(
        reqwest::Client::new(),
        mock_base_url.parse().expect("Failed to parse URL"),
        SecretString::from("api-key"),
    );

    let _mock_guard = Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "message": "invalid/model is not a valid model ID",
                "code": 400
            },
            "user_id": ""
        })))
        .mount_as_scoped(&mock_server)
        .await;

    let err = test_client
        .send_request(
            &invalid_model,
            AiRequest {
                text: "Test Message".to_owned(),
            },
        )
        .await
        .expect_err("Expected error");

    match err {
        Error::ApiError { status, message } => {
            assert_eq!(status, 400);
            assert!(message.contains("invalid/model is not a valid model ID"));
        }
        _ => panic!("Unexpected error"),
    }
}

#[tokio::test]
async fn test_models_endpoint() {
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let test_client = Openrouter::new_with_base_url(
        reqwest::Client::new(),
        mock_base_url.parse().expect("Failed to parse URL"),
        SecretString::from("api-key"),
    );

    let _mock_guard = Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {
                    "id": "openai/gpt-5",
                    "name": "OpenAI: GPT-5",
                    "created": 1741818122,
                    "description": "string",
                    "architecture": {
                        "input_modalities": [
                            "text",
                            "image"
                        ],
                        "output_modalities": [
                            "text"
                        ],
                        "tokenizer": "GPT",
                        "instruct_type": "string"
                    },
                    "top_provider": {
                        "is_moderated": true,
                        "context_length": 128000,
                        "max_completion_tokens": 16384
                    },
                    "pricing": {
                        "prompt": "0.0000007",
                        "completion": "0.0000007",
                        "image": "0",
                        "request": "0",
                        "web_search": "0",
                        "internal_reasoning": "0",
                        "input_cache_read": "0",
                        "input_cache_write": "0"
                    },
                    "canonical_slug": "string",
                    "context_length": 128000,
                    "hugging_face_id": "string",
                    "per_request_limits": {},
                    "supported_parameters": [
                        "string"
                    ]
                }
            ]
        })))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    let models = test_client.models().await.expect("Failed to list models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "openai/gpt-5");
    assert_eq!(models[0].name, "OpenAI: GPT-5");
}

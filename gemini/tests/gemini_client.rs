// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

use latchlm_core::{AiModel, AiProvider, AiRequest, Error};
use latchlm_gemini::{Gemini, GeminiModel};
use secrecy::{ExposeSecret, SecretString};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path_regex},
};

#[tokio::test]
async fn test_gemini_request_response() {
    // Setup mock server
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let model = GeminiModel::Flash25;

    // Replace direct struct instantiation with a JSON string
    let mock_response_body = serde_json::json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a mock response"
                        }
                    ]
                },
                "finishReason": "STOP",
                "index": 0
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
            "promptTokensDetails": []
        },
        "modelVersion": "",
        "responseId": ""
    });

    let test_api_key = SecretString::from("test_api_key");

    // Setup the mock
    let _mock_guard = Mock::given(method("POST"))
        .and(path_regex(r".+:generateContent$"))
        .and(header("x-goog-api-key", test_api_key.expose_secret()))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_body))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    // Create client with mock server URL
    let test_client = Gemini::new(reqwest::Client::new(), &mock_base_url, test_api_key);

    // Make the request
    let response = test_client
        .send_request(
            &model,
            AiRequest {
                text: "Test Message".to_owned(),
            },
        )
        .await
        .expect("Failed to send request");

    // Verify response content
    assert_eq!(response.text, "This is a mock response");
}

#[tokio::test]
async fn test_gemini_error_unhautenticated() {
    // Setup mock server
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let model = GeminiModel::Flash25;
    let error_response_body = serde_json::json!({
        "error": {
            "code": 401,
            "message": "UNAUTHENTICATED"
        }
    });

    // Setup the mock to return 400 error
    let _mock_guard = Mock::given(method("POST"))
        .and(path_regex(r".+:generateContent$"))
        .respond_with(ResponseTemplate::new(401).set_body_json(error_response_body))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    // Create client with mock server URL and invalid API key
    let test_client = Gemini::new(
        reqwest::Client::new(),
        &mock_base_url,
        SecretString::from("invalid"),
    );
    // Make the request that should fail
    let err = test_client
        .send_request(
            &model,
            AiRequest {
                text: "Test message".to_owned(),
            },
        )
        .await
        .expect_err("Expected an error but got a successful response");

    // Match on the error variant and check status and message
    match err {
        Error::ApiError { status, message } => {
            assert_eq!(status, 401);
            assert!(message.contains("UNAUTHENTICATED"));
        }
        _ => panic!("Expected ApiError variant"),
    }
}

#[tokio::test]
async fn test_gemini_error_invalid_model() {
    struct InvalidModel;

    impl AsRef<str> for InvalidModel {
        fn as_ref(&self) -> &str {
            "Invalid Model"
        }
    }

    impl AiModel for InvalidModel {}

    let gemini = Gemini::new(
        reqwest::Client::new(),
        "test-url",
        SecretString::from("api-key"),
    );

    let request = AiRequest {
        text: "Test Request".to_owned(),
    };

    let err = gemini
        .send_request(&InvalidModel, request)
        .await
        .expect_err("Expected an error but got a successful response");

    match err {
        Error::InvalidModelError(invalid_model) => {
            assert_eq!(invalid_model, InvalidModel.as_ref())
        }
        _ => panic!("Expected InvalidModelError"),
    }
}

use latchlm_core::{AiProvider, AiRequest, Error};
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

    let model = GeminiModel::Flash;
    let mock_response_body = r#"{
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a mock response"
                }]
            }
        }]
    }"#;

    let test_api_key = SecretString::from("test_api_key");

    // Setup the mock
    Mock::given(method("POST"))
        .and(path_regex(r".+:generateContent$"))
        .and(header("x-goog-api-key", test_api_key.expose_secret()))
        .respond_with(ResponseTemplate::new(200).set_body_string(mock_response_body))
        .expect(1)
        .mount(&mock_server)
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
async fn test_gemini_error_handling() {
    // Setup mock server
    let mock_server = MockServer::start().await;
    let mock_base_url = mock_server.uri();

    let model = GeminiModel::Flash;
    let error_response_body = r#"{
        "error": {
            "code": 401,
            "message": "UNAUTHENTICATED"
        }
    }"#;

    // Setup the mock to return 400 error
    Mock::given(method("POST"))
        .and(path_regex(r".+:generateContent$"))
        .respond_with(ResponseTemplate::new(401).set_body_string(error_response_body))
        .expect(1)
        .mount(&mock_server)
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

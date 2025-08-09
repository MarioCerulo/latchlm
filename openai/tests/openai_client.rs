use latchlm_core::{AiProvider, AiRequest};
use latchlm_openai::{Openai, OpenaiModel};
use secrecy::{ExposeSecret, SecretString};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{bearer_token, method},
};

#[tokio::test]
async fn test_openai_request_response() {
    // Setup mock server
    let mock_server = MockServer::start().await;
    let mock_base_url = reqwest::Url::parse(&mock_server.uri()).expect("Failed to parse URL");

    let model = OpenaiModel::Gpt41Nano;

    let mock_response_body = serde_json::json!({
      "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
      "object": "response",
      "created_at": 1741476542,
      "status": "completed",
      "error": null,
      "incomplete_details": null,
      "instructions": null,
      "max_output_tokens": null,
      "model": "gpt-4.1-2025-04-14",
      "output": [
        {
          "type": "message",
          "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
          "status": "completed",
          "role": "assistant",
          "content": [
            {
              "type": "output_text",
              "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
              "annotations": []
            }
          ]
        }
      ],
      "parallel_tool_calls": true,
      "previous_response_id": null,
      "reasoning": {
        "effort": null,
        "summary": null
      },
      "store": true,
      "temperature": 1.0,
      "text": {
        "format": {
          "type": "text"
        }
      },
      "tool_choice": "auto",
      "tools": [],
      "top_p": 1.0,
      "truncation": "disabled",
      "usage": {
        "input_tokens": 36,
        "input_tokens_details": {
          "cached_tokens": 0
        },
        "output_tokens": 87,
        "output_tokens_details": {
          "reasoning_tokens": 0
        },
        "total_tokens": 123
      },
      "user": null,
      "metadata": {}
    }
    );

    let test_api_key = SecretString::from("test_api_key");

    // Setup the mock
    let _mock_guard = Mock::given(method("POST"))
        .and(bearer_token(test_api_key.expose_secret()))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_body))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    // Create client with mock server URL
    let test_client =
        Openai::new_with_base_url(reqwest::Client::new(), mock_base_url, test_api_key);

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

    let expected = "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.";

    // Verify response content
    assert_eq!(response.text, expected);
}

#[tokio::test]
async fn test_openai_gpt5_nano_response_format() {
    let mock_server = MockServer::start().await;
    let mock_base_url = reqwest::Url::parse(&mock_server.uri()).expect("Failed to parse URL");

    let model = OpenaiModel::Gpt5Nano;

    let mock_response = serde_json::json!({
        "id": "resp_6895cc06975881a0bfa0a29cc84b19aa08c1c8e0f30ad9dd",
        "object": "response",
        "created_at": 1754647558,
        "status": "completed",
        "background": false,
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "max_tool_calls": null,
        "model": "gpt-5-nano-2025-08-07",
        "output": [
            {
                "id": "rs_6895cc07378481a0860a65a15a2edc3708c1c8e0f30ad9dd",
                "type": "reasoning",
                "summary": []
            },
            {
                "id": "msg_6895cc09a74881a0a3fb2cd04a3be83308c1c8e0f30ad9dd",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "AI, or artificial intelligence, refers to computer systems that can perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images or sounds, solving problems, learning from data, and making decisions."
                    }
                ],
                "role": "assistant"
            }
        ],
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "prompt_cache_key": null,
        "reasoning": {
            "effort": "medium",
            "summary": null
        },
        "safety_identifier": null,
        "service_tier": "default",
        "store": true,
        "temperature": 1.0,
        "text": {
            "format": {
                "type": "text"
            },
            "verbosity": "medium"
        },
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 0
            },
            "output_tokens": 934,
            "output_tokens_details": {
                "reasoning_tokens": 640
            },
            "total_tokens": 944
        },
        "user": null,
        "metadata": {}
    });

    let test_api_key = SecretString::from("test_api_key");

    let _mock_guard = Mock::given(method("POST"))
        .and(bearer_token(test_api_key.expose_secret()))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    let client = Openai::new_with_base_url(reqwest::Client::new(), mock_base_url, test_api_key);

    let response = client
        .send_request(
            &model,
            AiRequest {
                text: "What is AI?".to_owned(),
            },
        )
        .await
        .map_err(|e| panic!("Error: {e}"));

    let expected = "AI, or artificial intelligence, refers to computer systems that can perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images or sounds, solving problems, learning from data, and making decisions.";

    assert_eq!(response.unwrap().text, expected);
}

#[tokio::test]
async fn test_openai_error_unhautenticated() {
    let mock_server = MockServer::start().await;
    let mock_server_url =
        reqwest::Url::parse(&mock_server.uri()).expect("Failed to parse mock URL");

    let model = OpenaiModel::Gpt4o;
    let api_key = SecretString::from("incorrect_api_key");

    let error_response_body = serde_json::json!({
      "error": {
        "message": "Incorrect API key provided: incorrect_api_key. You can find your API key at https://platform.openai.com/account/api-keys.",
        "type": "invalid_request_error",
        "param": null,
        "code": "invalid_api_key"
      }
    });

    let openai = Openai::new_with_base_url(reqwest::Client::new(), mock_server_url, api_key);

    let _mock_guard = Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&error_response_body))
        .expect(1)
        .mount_as_scoped(&mock_server)
        .await;

    let res = openai
        .send_request(
            &model,
            AiRequest {
                text: "test".into(),
            },
        )
        .await;

    assert!(res.is_err())
}

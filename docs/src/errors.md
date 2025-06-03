# Errors

LatchLM uses a unified error type to simplify error handling across all providers and models.

## Error Type

All fallible operations in LatchLM return a `Result<T, Error>`, where `Error` is an enum defined in the core crate. This makes it easy to handle errors in a consistent way, regardless of the provider or model you are using.

## Error Variants

The main error variants are:

- **RequestError**:
  Occurs when an HTTP request fails (e.g., network issues, timeouts, invalid URLs).
  Wraps a `reqwest::Error`.

- **ApiError**:
  Represents an error returned by the API provider itself, such as invalid API keys, quota exceeded, or unsupported operations.
  Contains the HTTP status code and a message.

- **ParseError**:
  Indicates a failure to parse the response from the provider (e.g., invalid JSON).
  Wraps a `serde_json::Error`.

- **InvalidModelError**:
  Returned when an invalid or unsupported model name is used

## Example
```rust
use latchlm::{AiProvider, AiModel, AiRequest, Error};

async fn call_model (
    provider: &dyn AiProvider,
    model: &dyn AiModel,
    prompt: &str,
) -> Result<String, Error> {
    let request = AiRequest { text: prompt.to_string() };
    let response = provider.send_request(model, request).await?;
    Ok(response.text)
}
```
### Handling Errors
You can match on the `Error` type to handle different error cases
```rust
match result {
    Ok(response) => println!("AI response: {}", response.text),
    Err(Error::RequestError(e)) => eprintln!("Network error: {e}"),
    Err(Error::ApiError { status, message }) => {
        eprintln!("API error (status {status}): {message}")
    },
    Err(Error::ParseError(e)) => eprintln!("Failed to parse response: {e}"),
    Err(Error::InvalidModelError(name)) => eprintln!("Invalid model: {name}"),
}
```

## Extensibility
The `Error` enum is marked as `#[non_exhaustive]`, which means additional variants may be added in the future.

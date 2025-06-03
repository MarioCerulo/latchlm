# Providers

In LatchLM a provider is a struct that implements the `AiProvider` trait, encapsulating the logic to interact with a given API.

## First-party Providers

LatchLM currently provides, with experimental support, a provider for the Gemini APi. More providers will follow.

## Implementing a provider

Below is an example of how you can implement a custom provider.

```rust
use latchlm::{AiModel, AiProvider, AiResponse, BoxFuture, Result};
use reqwest::Client;
use secrecy::SecretString;

pub struct MyProvider {
    client: reqwest::Client,
    api_key: SecretString,
}

impl AiProvider for MyProvider {
    fn send_request(&self, model: &dyn AiModel, message: &str) -> BoxFuture<Result<AiResponse>> {
        // Your implementation goes here.
        // For instance, you might use an async block like:
        // Box::pin(async move {
        //      // Build and send an HTTP request with self.client.
        //      // Use model.as_ref() to obtain the identifier.
        //      // Process the response and construct an AiResponse.
        //      OK(AiResponse { text: "Example response text".into() })
        // })
    }
}
```

# Implement Your Own Provider

Below is an example of how you can implement a custom provider.

```rust
use futures::stream::BoxStream;
use latchlm::{AiModel, AiProvider, AiResponse, BoxFuture, Result};
use reqwest::Client;
use secrecy::SecretString;

pub struct MyProvider {
    client: reqwest::Client,
    api_key: SecretString,
}

impl AiProvider for MyProvider {
    fn send_request(&self, model: &dyn AiModel, message: AiRequest) -> BoxFuture<'_, Result<AiResponse>> {
        // Your implementation goes here.
        // For instance, you might use an async block like:
        // Box::pin(async move {
        //      // Build and send an HTTP request with self.client.
        //      // Use model.as_ref() to obtain the identifier.
        //      // Process the response and construct an AiResponse.
        //      OK(AiResponse { text: "Example response text".into() })
        // })
    }
    
    fn send_streaming(&self, model: &dyn AiModel, request: AiRequest) -> BoxStream<'_, Result<AiResponse>> {
        // Your implementation goes here.
        // For instance, you might use an async block like:
        // Box::pin(async move {
        //      // Build and send an HTTP request with self.client.
        //      // Use model.as_ref() to obtain the identifier.
        //      // Process the response and construct an AiResponse.
        //      futures::stream::once(async { Ok(AiResponse { text: "Example response text".into() }) })
        // })
    }
}
```

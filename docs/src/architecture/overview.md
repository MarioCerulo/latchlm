# Overview

LatchLM is designed with a focus on flexibility, extensibility, and type safety. This section explains key architectural decisions and patterns used throughout the library.

## Core Design Patterns

### Dynamic Dispatch

LatchLM is built around dynamic dispatch to enable runtime model selection. The `AiProvider` trait uses `&dyn AiModel` parameters, allowing any implementor of the `AiModel` trait to be passed to provider methods. This design enables you to:

- Switch between different AI models at runtime
- Use different providers interchangeably with the same interface
- Create provider-agnostic abstraction layers in your application

Example of dynamic model selection:
```rust
async fn get_response(
    provider: &dyn AiProvider,
    model: &dyn AiModel,
    prompt: &str,
) -> Result<String> {
    let response = provider.send_request(model, prompt).await?;
    Ok(response.text)
}
```

### Asynchronous Interface

LatchLM uses Rust's async/await system to provide non-blocking operations.
The `BoxFuture` type alias simplifies returning futures from trait methods:

```rust
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;
```

### Thread Safety

Key traits in LatchLM include `Send + Sync` bounds to ensure thread safety:

```rust
pub trait AiModel: AsRef<str> + Send + Sync {}

pub trait AiProvider: Send + Sync { /* ... */ }
```

### Smart Pointers Support

LatchLM provides blanket implementations for the `AiProvider` trait for references (`&T`, `&mut T`, `Box<T>` and `Arc<T>`) allowing for flexible ownership and sharing patterns.

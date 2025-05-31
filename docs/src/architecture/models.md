# Models

In LatchLM, a model is an implementor of the marker trait `AiModel`.
Models serve as unique identifiers for the different variants supported by AI providers.

## Implementing a Model

Below is an example of how you can implement a custom model family.

```rust
use latchlm::AiModel;

// Custom AI model variants
pub enum MyModel {
    Fast,
    Advanced,
}

impl AsRef<str> for MyModel {
    fn as_ref(&self) -> &str {
        match self {
            MyModel::Fast => "mymodel-fast",
            MyModel::Advanced => "mymodel-advanced",
        }
    }
}

impl AiModel for MyModel {}
```

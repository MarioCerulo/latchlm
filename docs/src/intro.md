# Introduction

> **⚠️ Warning:**
> The documentation is still in development. Some sections and examples might change as the library evolves.

LatchLM is a provider-agnostic client library for AI models. Its goal is to provide a uniform, modular, interface for interacting with different providers.
By abstracting away provider-specific details, LatchLM allows you to integrate with multiple AI APIs seamlessly.

At the core of LatchLM is the `AiProvider` trait, which defines an asynchronous interface to send requests to an LLM. This trait simplifies the development of non-blocking applications that interact with AI models.

Key components include:
- **AiProvider Trait:** Abstracts the underlying API call, enabling provider-agnostic interactions.
- **AiModel Marker Trait:** Represents specific AI model variants. Implementing this trait allows you to convert model variants into string identifiers (via `AsRef<str>`), ensuring reliable model referencing.
- **ModelId:** encapsulates metadata about an Ai model, such as unique identifier and a human-readable name.
- **AiResponse:** Wraps the response from a model, allowing you to handle responses in a uniform way.

LatchLM is built with extensibility in mind, meaning that adding support for a new provider often requires only implementing the core traits. This design allows you to focus on building AI-driven applications without being locked into a single vendor.

## Why LatchLM?

_There's plenty of rust crates out there that wraps LLMs APIs, why create yet another one?_

Of the ones I tried, none of them suited my use case: swapping the model at runtime.
I needed a way to just select a model and treat it for what it is, an AI model. I need dynamic dispatch for different providers, and installing a crate for each one of them, was not gonna cut it. I could have used a crate that supported different providers and add dynamic dispatch on top in my specific app, but that would have been less fun and I wouldn't have learned as much.

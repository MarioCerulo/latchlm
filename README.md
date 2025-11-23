# LatchLM

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://mozilla.org/MPL/2.0/)

**LatchLM** is a modular Rust ecosystem for interacting with multiple AI model providers through a unified, provider-agnostic interface.

## Features

- **Provider-agnostic:** Swap between different AI/LLM providers at runtime.
- **Extensible:** Add new providers and models by implementing simple traits.
- **Unified API:** Consistent request/response types across providers.

## Supported Providers

- Google Gemini
- OpenAI
- OpenRouter

## Getting Started

Add LatchLM to your `Cargo.toml`:

```toml
[dependencies]
latchlm = "0.2"
```

## License

LatchLM source code and documentation are licensed under [Mozilla Public License v2.0](./LICENSE.md)

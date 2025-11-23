# Providers

In LatchLM a provider is a struct that implements the `AiProvider` trait, encapsulating the logic to interact with a given API.
In addition to some providers implemented by LatchLM and available through feature flags, users can also implement their own providers by implementing the `AiProvider` trait.

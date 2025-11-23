// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! # LatchLM Core
//!
//! Core traits and types for the LatchLM ecosystem.
//!
//! This crate provides the foundation for the LatchLM ecosystem by defining
//! the core abstractions used across all provider implementations.

pub mod error;
pub use error::*;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, future::Future, pin::Pin, sync::Arc};

/// A `Future` type used by the `AiProvider` trait.
///
/// This type alias represents a boxed, pinned future that is `Send`,
/// which allows to be returned from async traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A trait representing a specific AI model for a provider.
///
/// Implementors of this trait represent specific model variants supported by an LLM provider.
/// Each model must be convertible to a string identifier that can be used in API requests.
///
/// # Example
/// ```
/// use latchlm_core::{AiModel, ModelId};
///
/// pub enum MyModel {
///     Variant1,
///     Variant2,
/// }
///
/// impl AsRef<str> for MyModel {
///     fn as_ref(&self) -> &str {
///         match self {
///             MyModel::Variant1 => "mymodel-variant-1",
///             MyModel::Variant2 => "mymodel-variant-2",
///         }
///     }
/// }
///
/// impl AiModel for MyModel {
///     fn as_any(&self) -> &dyn std::any::Any {
///         self
///     }
///
///     fn model_id(&self) -> ModelId {
///         match self {
///             MyModel::Variant1 => ModelId { id: "mymodel-variant-1".into(), name: "My Model Variant 1".into()},
///             MyModel::Variant2 => ModelId { id: "mymodel-variant-2".into(), name: "My Model Variant 2".into()},
///         }
///     }
/// }
/// ```
pub trait AiModel: AsRef<str> + Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn model_id(&self) -> ModelId<'_>;
}

impl dyn AiModel {
    pub fn downcast<M: 'static + Clone>(&self) -> Option<M> {
        self.as_any().downcast_ref::<M>().cloned()
    }
}

/// A unique identifier for an LLM model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelId<'a> {
    /// The technical identifier used in API requests
    pub id: Cow<'a, str>,
    /// A human-readable name
    pub name: Cow<'a, str>,
}

impl std::fmt::Display for ModelId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A request for an LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiRequest {
    /// The input text to be processed by the model
    pub text: String,
}

/// Response from an LLM API provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AiResponse {
    /// The text response
    pub text: String,
    /// Token usage data
    pub token_usage: TokenUsage,
}

/// Token usage information returned by LLM providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Number of tokens in the input prompt
    pub input_tokens: Option<u64>,
    /// Number of tokens in the output response
    pub output_tokens: Option<u64>,
    /// Total tokens used during the interaction
    pub total_tokens: Option<u64>,
}

/// A trait representing an LLM API provider.
///
/// Implementors of this trait provide the functionality to interact with specific
/// LLM API providers through a unified interface.
///
/// Blanket implementations are provided for `&T`, `&mut T`, `Box<T>` and `Arc<T>`
/// where `T: AiProvider`.
pub trait AiProvider: Send + Sync {
    /// Sends a message to the specified model and returns the AI's response.
    ///
    /// # Arguments
    ///
    /// * `model` - The identifier of the model to use.
    /// * `request` - The request to send to the model.
    ///
    /// # Returns
    ///
    /// A future yielding either a `Response` or an `Error`
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the request fails, the response status is not successful,
    /// or if the response cannot be parsed.
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>>;

    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>>;
}

impl<T> AiProvider for &T
where
    T: AiProvider + ?Sized,
{
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        (**self).send_request(model, request)
    }

    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>> {
        (**self).send_streaming(model, request)
    }
}

impl<T> AiProvider for &mut T
where
    T: AiProvider + ?Sized,
{
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        (**self).send_request(model, request)
    }

    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>> {
        (**self).send_streaming(model, request)
    }
}

impl<T> AiProvider for Box<T>
where
    T: AiProvider + ?Sized,
{
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        (**self).send_request(model, request)
    }

    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>> {
        (**self).send_streaming(model, request)
    }
}

impl<T> AiProvider for Arc<T>
where
    T: AiProvider + ?Sized,
{
    fn send_request(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxFuture<'_, Result<AiResponse>> {
        (**self).send_request(model, request)
    }

    fn send_streaming(
        &self,
        model: &dyn AiModel,
        request: AiRequest,
    ) -> BoxStream<'_, Result<AiResponse>> {
        (**self).send_streaming(model, request)
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! # LatchLM
//!
//! LatchLM is a provider-agnostic, modular ecosystem for interacting with AI APIs.
//! This crate serves as the main entrypoint, re-exporting traits and types for building
//! applications and libraries that work with various LLM providers.

pub use latchlm_core::*;

#[cfg(feature = "gemini")]
pub use latchlm_gemini as gemini;

#[cfg(feature = "openai")]
pub use latchlm_openai as openai;

#[cfg(feature = "openrouter")]
pub use latchlm_openrouter as openrouter;

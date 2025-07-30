// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

//! Crate-wide error definitions.
//!
//! This module defines a unified `Error` enum used throughout the crate.

#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Api error: {status} - {message} ")]
    ApiError { status: u16, message: String },

    #[error("Failed to parse the response")]
    ParseError(#[from] serde_json::Error),

    #[error("Invalid model name: {0}")]
    InvalidModelError(String),

    #[error("Provider settings error: {provider} : {error}")]
    ProviderError { provider: String, error: String },
}

pub type Result<T> = std::result::Result<T, Error>;

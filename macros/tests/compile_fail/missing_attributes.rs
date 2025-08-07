#![allow(unused)]

use latchlm_core::AiModel;
use latchlm_macros::AiModel;

#[derive(AiModel)]
enum Model {
    Variant,
}

fn main() {}

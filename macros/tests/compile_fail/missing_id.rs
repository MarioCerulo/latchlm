#![allow(unused)]

use latchlm_core::AiModel;
use latchlm_macros::AiModel;

#[derive(AiModel)]
enum Model {
    #[model(name = "Variant")]
    Variant,
}

fn main() {
    Model::Variant.as_ref();
}

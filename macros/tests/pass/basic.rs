#![allow(unused)]

use latchlm_core::AiModel;
use latchlm_macros::AiModel;

#[derive(AiModel)]
enum Model {
    #[model(id = "variant-1", name = "V1")]
    Variant1,
    #[model(id = "variant-2", name = "V2")]
    Variant2,
}

fn main() {
    Model::Variant1.as_ref();
}

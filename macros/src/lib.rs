// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input};

/// Derive macro for the `AiModel` trait.
///
/// This macro can be used to automatically implement the `AiModel` trait for enums representing model variants.
/// Each enum variant must be annotated with a `#[model(id = "...", name = "...")] attribute specifying the model's
/// technical identifier and human-readable name.
///
/// The macro will implement:
/// - `AiModel` (with `model_id`)
/// - `AsRef<str>`
/// - `TryFrom<&str>`
/// - `serde::Serialize` and `serde::Deserialize`
/// - A static `variants()` method returning all model IDs.
///
/// # Errors
/// - Only enums are supported.
/// - Each variant must have both `id` and `name` specified in the `#[model]` attribute.
/// - Only `id` and `name` are supported keys in the attribute.
///
/// # Example
/// ```
/// use latchlm_core::AiModel;
/// use latchlm_macros::AiModel;
///
/// #[derive(AiModel)]
/// pub enum MyModel {
///     #[model(id = "mymodel-variant-1", name = "My Model Variant 1")]
///     Variant1,
///     #[model(id = "mymodel-variant-2", name = "My Model Variant 2")]
///     Variant2,
/// }
/// ```
#[proc_macro_derive(AiModel, attributes(model))]
pub fn ai_model_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match ai_model_derive_impl(input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

fn ai_model_derive_impl(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;

    let Data::Enum(data_enum) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "AiModel can only be derived for enums",
        ));
    };

    let variant_infos = data_enum
        .variants
        .iter()
        .map(|variant| {
            let variant_name = &variant.ident;
            let (id_value, name_value) = extract_model_attributes(variant)?;
            Ok((variant_name, id_value, name_value))
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let set: std::collections::HashSet<_> = variant_infos.iter().map(|(_, id, _)| id).collect();
    if set.len() != variant_infos.len() {
        return Err(syn::Error::new_spanned(
            input,
            "Repeated Id: model id must be unique",
        ));
    }

    let as_ref_arms = variant_infos.iter().map(|(variant_name, id_value, _)| {
        quote! {
            #name::#variant_name => #id_value,
        }
    });

    let try_from_arms = variant_infos
        .iter()
        .map(|(variant_name, id_value, _)| {
            quote! {
                #id_value => Ok(#name::#variant_name),
            }
        })
        .collect::<Vec<_>>();

    let serde_serialize_arms = variant_infos
        .iter()
        .map(|(variant_name, id_value, _)| {
            quote! {
                #name::#variant_name => serializer.serialize_str(#id_value),
            }
        })
        .collect::<Vec<_>>();

    let serde_deserialize_arms = variant_infos
        .iter()
        .map(|(variant_name, id_value, _)| {
            quote! {
                #id_value => Ok(#name::#variant_name),
            }
        })
        .collect::<Vec<_>>();

    let valid_variants = variant_infos
        .iter()
        .map(|(_, id_value, _)| id_value.as_str())
        .collect::<Vec<_>>();

    let expecting_message = format!("one of: {}", valid_variants.join(", "));

    let model_id_arms = variant_infos
        .iter()
        .map(|(variant_name, id, model_name)| {
            quote! {
                #name::#variant_name => ::latchlm_core::ModelId {
                    id: #id,
                    name: #model_name,
                }
            }
        })
        .collect::<Vec<_>>();

    let array_arms = variant_infos
        .iter()
        .map(|(_, id, model_name)| {
            quote! {
                ::latchlm_core::ModelId {
                    id: #id,
                    name: #model_name
                }
            }
        })
        .collect::<Vec<_>>();

    let array_len = array_arms.len();

    let expanded = quote! {
        impl AiModel for #name {
            fn model_id(&self) -> ::latchlm_core::ModelId {
                match self {
                    #(#model_id_arms),*
                }
            }

        }

        impl ::core::convert::AsRef<str> for #name {
            fn as_ref(&self) -> &str {
                match self {
                    #(#as_ref_arms)*
                }
            }
        }

        impl ::core::convert::TryFrom<&str> for #name {
            type Error = ::latchlm_core::Error;
            fn try_from(value: &str) -> ::latchlm_core::Result<Self> {
                match value {
                    #(#try_from_arms)*
                    invalid_model => Err(::latchlm_core::Error::InvalidModelError(invalid_model.to_string())),
                }
            }
        }

        impl ::serde::Serialize for #name {
            fn serialize<S>(&self, serializer: S) -> ::std::result::Result<S::Ok, S::Error>
            where
                S: ::serde::Serializer,
            {
                match self {
                    #(#serde_serialize_arms)*
                }
            }
        }

        impl<'de> ::serde::Deserialize<'de> for #name {
            fn deserialize<D>(deserializer: D) -> ::std::result::Result<Self, D::Error>
            where
                D: ::serde::Deserializer<'de>,
            {
                struct ModelVisitor;

                impl<'de> ::serde::de::Visitor<'de> for ModelVisitor {
                    type Value = #name;

                    fn expecting(&self, formatter: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                        formatter.write_str(#expecting_message)
                    }

                    fn visit_str<E>(self, value: &str) -> ::std::result::Result<Self::Value, E>
                    where
                        E: ::serde::de::Error,
                    {
                        match value {
                            #(#serde_deserialize_arms)*
                            invalid_model => Err(::serde::de::Error::unknown_variant(value, &[#(#valid_variants),*]))
                        }
                    }
                }

                deserializer.deserialize_str(ModelVisitor)
            }
        }

        impl #name {
            pub fn variants() -> &'static [::latchlm_core::ModelId] {
                const VARS: [::latchlm_core::ModelId; #array_len] = [
                    #(#array_arms),*
                ];

                &VARS
            }
        }
    };

    Ok(expanded.into())
}

fn extract_model_attributes(variant: &syn::Variant) -> syn::Result<(String, String)> {
    use syn::{Error, Expr, Lit, Meta};

    let mut model_id = None;
    let mut model_name = None;

    for attr in &variant.attrs {
        if !attr.path().is_ident("model") {
            continue;
        }

        let args = attr
            .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
            .map_err(|_| syn::Error::new_spanned(attr, "Invalid model attribute syntax"))?;

        for meta in args {
            match meta {
                Meta::NameValue(name_value) if name_value.path.is_ident("id") => {
                    match &name_value.value {
                        Expr::Lit(expr_lit) => match &expr_lit.lit {
                            Lit::Str(lit_str) => {
                                model_id = Some(lit_str.value());
                            }
                            _ => {
                                return Err(Error::new_spanned(
                                    &name_value.value,
                                    "Model id must be a string literal",
                                ));
                            }
                        },
                        _ => {
                            return Err(Error::new_spanned(
                                &name_value.value,
                                "Model id must be a string literal",
                            ));
                        }
                    }
                }
                Meta::NameValue(name_value) if name_value.path.is_ident("name") => {
                    match &name_value.value {
                        Expr::Lit(expr_lit) => match &expr_lit.lit {
                            Lit::Str(lit_str) => {
                                model_name = Some(lit_str.value());
                            }
                            _ => {
                                return Err(Error::new_spanned(
                                    &name_value.value,
                                    "Model name must be a string literal",
                                ));
                            }
                        },
                        _ => {
                            return Err(Error::new_spanned(
                                &name_value.value,
                                "Model name must be a string literal",
                            ));
                        }
                    }
                }
                Meta::NameValue(name_value) => {
                    return Err(Error::new_spanned(
                        &name_value.path,
                        "Only 'id' and 'name' are supported in #[model] attribute",
                    ));
                }
                _ => {
                    return Err(Error::new_spanned(
                        meta,
                        "Expected #[model(id = \"...\", name = \"...\")]",
                    ));
                }
            }
        }
    }

    let id = model_id
        .ok_or_else(|| Error::new_spanned(&variant.ident, "missing #[model] attribute with id"))?;

    let name = model_name.ok_or_else(|| {
        Error::new_spanned(&variant.ident, "missing #[model] attribute with name")
    })?;

    Ok((id, name))
}

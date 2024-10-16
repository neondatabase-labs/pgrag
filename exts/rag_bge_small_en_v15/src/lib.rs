use pgrx::prelude::*;

mod chunk;
mod errors;

pg_module_magic!();

#[pg_schema]
mod rag_bge_small_en_v15 {
    use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
    use tokenizers::{
        tokenizer::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams},
        DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, TokenizerImpl,
    };
    use tract_ndarray::s;
    use tract_onnx::prelude::*;

    use super::errors::*;
    use pgrx::prelude::*;
    use std::{cell::OnceCell, io::Cursor};

    macro_rules! local_tokenizer_files {
        // NOTE: macro assumes /unix/style/paths
        ($folder:literal) => {
            TokenizerFiles {
                tokenizer_file: include_bytes!(concat!($folder, "/tokenizer.json")).to_vec(),
                config_file: include_bytes!(concat!($folder, "/config.json")).to_vec(),
                special_tokens_map_file: include_bytes!(concat!($folder, "/special_tokens_map.json")).to_vec(),
                tokenizer_config_file: include_bytes!(concat!($folder, "/tokenizer_config.json")).to_vec(),
            }
        };
    }

    macro_rules! local_model {
        // NOTE: macro assumes /unix/style/paths
        ($model:ident, $folder:literal) => {
            $model {
                onnx_file: include_bytes!(concat!($folder, "/model.onnx")).to_vec(),
                tokenizer_files: local_tokenizer_files!($folder),
            }
        };
    }

    // NOTE. It might be nice to expose this function directly, but as at 2024-07-08 pgrx
    // doesn't support Vec<Vec<_>>: https://github.com/pgcentralfoundation/pgrx/issues/1762.

    // #[pg_extern(immutable, strict, name = "_embedding")]
    pub fn embeddings(input: Vec<&str>) -> Vec<Vec<f32>> {
        thread_local! {
            static CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let model = cell.get_or_init(|| {
                let user_def_model = local_model!(UserDefinedEmbeddingModel, "../../../lib/bge_small_en_v15");
                TextEmbedding::try_new_from_user_defined(user_def_model, Default::default())
                    .expect_or_pg_err("Couldn't load model bge_small_en_v15")
            });

            model
                .embed(input, None)
                .expect_or_pg_err("Unable to generate bge_small_en_v15 embeddings")
        })
    }

    pub fn normalize(v: &[f32]) -> Vec<f32> {
        let norm = (v.iter().map(|val| val * val).sum::<f32>()).sqrt();
        let epsilon = 1e-12; // add a super-small epsilon to avoid dividing by zero
        v.iter().map(|&val| val / (norm + epsilon)).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn embedding_tract(input: &str) -> Vec<f32> {
        thread_local! {
            static TCELL: OnceCell<(
                SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>>,
                TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>
            )> = const { OnceCell::new() };
        }
        TCELL.with(|cell| {
            let (model, tokenizer) = cell.get_or_init(|| {
                let raw_model = include_bytes!("../../../lib/bge_small_en_v15/model.onnx");
                let mut cursor = Cursor::new(raw_model);
                let model = tract_onnx::onnx()
                    .model_for_read(&mut cursor)
                    .expect_or_pg_err("Couldn't read ONNX model")
                    .into_optimized()
                    .expect_or_pg_err("Couldn't optimize ONNX model")
                    .into_runnable()
                    .expect_or_pg_err("Couldn't make ONNX model runnable");

                let mut tokenizer: tokenizers::Tokenizer =
                    tokenizers::Tokenizer::from_bytes(include_bytes!("../../../lib/bge_small_en_v15/tokenizer.json"))
                        .expect_or_pg_err("Couldn't load tokenizer.json");

                let config: serde_json::Value =
                    serde_json::from_slice(include_bytes!("../../../lib/bge_small_en_v15/config.json"))
                        .expect_or_pg_err("Couldn't load config.json");

                let special_tokens_map: serde_json::Value =
                    serde_json::from_slice(include_bytes!("../../../lib/bge_small_en_v15/special_tokens_map.json"))
                        .expect_or_pg_err("Couldn't load special_tokens_map.json");

                let tokenizer_config: serde_json::Value =
                    serde_json::from_slice(include_bytes!("../../../lib/bge_small_en_v15/tokenizer_config.json"))
                        .expect_or_pg_err("Couldn't load tokenizer_config.json");

                let max_length = tokenizer_config["model_max_length"]
                    .as_f64()
                    .unwrap_or_pg_err("Error reading model_max_length from tokenizer_config.json")
                    as usize;

                let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
                let pad_token = tokenizer_config["pad_token"]
                    .as_str()
                    .unwrap_or_pg_err("Error reading pad_token from tokenizer_config.json")
                    .into();

                let mut tokenizer = tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .with_truncation(Some(TruncationParams {
                        max_length,
                        ..Default::default()
                    }))
                    .expect_or_pg_err("Couldn't set truncation parameters")
                    .clone();

                if let serde_json::Value::Object(root_object) = special_tokens_map {
                    for (_, value) in root_object.iter() {
                        if value.is_string() {
                            tokenizer.add_special_tokens(&[AddedToken {
                                content: value.as_str().unwrap().into(),
                                special: true,
                                ..Default::default()
                            }]);
                        } else if value.is_object() {
                            tokenizer.add_special_tokens(&[AddedToken {
                                content: value["content"].as_str().unwrap().into(),
                                special: true,
                                single_word: value["single_word"].as_bool().unwrap(),
                                lstrip: value["lstrip"].as_bool().unwrap(),
                                rstrip: value["rstrip"].as_bool().unwrap(),
                                normalized: value["normalized"].as_bool().unwrap(),
                            }]);
                        }
                    }
                }
                (model, tokenizer)
            });

            let tokenizer_output = tokenizer
                .encode(input, true)
                .expect_or_pg_err("Couldn't tokenize string");

            let input_ids = tokenizer_output.get_ids();
            let attention_mask = tokenizer_output.get_attention_mask();
            let token_type_ids = tokenizer_output.get_type_ids();
            let length = input_ids.len();

            let input_ids: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, length), input_ids.iter().map(|&x| x as i64).collect())
                    .expect_or_pg_err("Couldn't create input IDs")
                    .into();

            let attention_mask: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, length), attention_mask.iter().map(|&x| x as i64).collect())
                    .expect_or_pg_err("Couldn't create attention mask")
                    .into();

            let token_type_ids: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, length), token_type_ids.iter().map(|&x| x as i64).collect())
                    .expect_or_pg_err("Couldn't create token type IDs")
                    .into();

            let outputs = model
                .run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into()))
                .expect_or_pg_err("Couldn't run embedding model");

            let output = &outputs[0];
            let logits: Vec<f32> = output
                .to_array_view::<f32>()
                .expect_or_pg_err("Couldn't make results into array")
                .slice(s![.., -1, ..])
                .iter()
                .map(|&x| x)
                .collect();

            normalize(&logits)
        })
    }

    #[pg_extern(immutable, strict)]
    pub fn _embedding(input: &str) -> Vec<f32> {
        let vectors = embeddings(vec![input]);
        vectors
            .into_iter()
            .next()
            .unwrap_or_pg_err("Unexpectedly empty result vector")
    }

    extension_sql!(
        "CREATE FUNCTION rag_bge_small_en_v15.embedding_for_passage(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding(input)::vector(384);
        $$;
        CREATE FUNCTION rag_bge_small_en_v15.embedding_for_query(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding('Represent this sentence for searching relevant passages: ' || input)::vector(384);
        $$;",
        name = "embeddings",
    );
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_bge_small_en_v15::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_embedding_length() {
        assert_eq!(_embedding("hello world!").len(), 384);
    }

    #[pg_test]
    fn test_embedding_immutability() {
        assert_eq!(_embedding("hello world!"), _embedding("hello world!"));
    }

    #[pg_test]
    fn test_embedding_variability() {
        assert_ne!(_embedding("hello world!"), _embedding("bye moon!"));
    }

    #[pg_test]
    fn test_embedding_tract_length() {
        assert_eq!(embedding_tract("hello world!").len(), 384);
    }

    fn approx_equal(a: Vec<f32>, b: Vec<f32>, epsilon: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }

    #[pg_test]
    fn test_embedding_tract_eq() {
        assert!(approx_equal(
            embedding_tract("hello world!"),
            _embedding("hello world!"),
            0.0001
        ));
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}

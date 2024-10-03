use pgrx::prelude::*;

mod errors;

pg_module_magic!();

#[pg_schema]
mod rag_embed_bge_small_en_v15 {
    use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
    use pgrx::prelude::*;
    use std::cell::OnceCell;
    use super::errors::*;

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

    // #[pg_extern(immutable, strict, name = "embedding")]
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

    #[pg_extern(immutable, strict)]
    pub fn _embedding(input: &str) -> Vec<f32> {
        let vectors = embeddings(vec![input]);
        vectors
            .into_iter()
            .next()
            .unwrap_or_pg_err("Unexpectedly empty result vector")
    }

    extension_sql!(
        "CREATE FUNCTION rag_embed_bge_small_en_v15.embedding_for_passage(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_embed_bge_small_en_v15._embedding(input)::vector(384);
        $$;
        CREATE FUNCTION rag_embed_bge_small_en_v15.embedding_for_query(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_embed_bge_small_en_v15._embedding('Represent this sentence for searching relevant passages: ' || input)::vector(384);
        $$;",
        name = "embedding_bge_small_en_v15",
    );
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_embed_bge_small_en_v15::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_embedding_bge_small_en_v15_length() {
        assert_eq!(_embedding("hello world!").len(), 384);
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_immutability() {
        assert_eq!(
            _embedding("hello world!"),
            _embedding("hello world!")
        );
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_variability() {
        assert_ne!(
            _embedding("hello world!"),
            _embedding("bye moon!")
        );
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

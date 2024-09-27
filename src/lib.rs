use pgrx::prelude::*;

mod errors;
mod json_api;

mod chunk;
mod docx;
mod fireworks;
mod markdown;
mod openai;
mod pdf;

pg_module_magic!();

#[pg_schema]
mod neon_ai {
    use fastembed::{
        RerankResult, TextEmbedding, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
    };
    use pgrx::prelude::*;
    use std::cell::OnceCell;

    use super::errors::*;

    extension_sql!(
        "CREATE TABLE neon_ai.config(name text PRIMARY KEY, value text);
        REVOKE ALL ON TABLE neon_ai.config FROM PUBLIC;",
        name = "config",
    );

    // === embedding embeddings ===

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

    // === local embeddings ===

    // NOTE. It might be nice to expose this function directly, but as at 2024-07-08 pgrx
    // doesn't support Vec<Vec<_>>: https://github.com/pgcentralfoundation/pgrx/issues/1762.

    // #[pg_extern(immutable, strict, name = "embedding_bge_small_en_v15")]
    pub fn embeddings_bge_small_en_v15(input: Vec<&str>) -> Vec<Vec<f32>> {
        thread_local! {
            static CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let model = cell.get_or_init(|| {
                let user_def_model = local_model!(UserDefinedEmbeddingModel, "../bge_small_en_v15");
                TextEmbedding::try_new_from_user_defined(user_def_model, Default::default())
                    .expect_or_pg_err("Couldn't load model bge_small_en_v15")
            });

            model
                .embed(input, None)
                .expect_or_pg_err("Unable to generate bge_small_en_v15 embeddings")
        })
    }

    #[pg_extern(immutable, strict)]
    pub fn _embedding_bge_small_en_v15(input: &str) -> Vec<f32> {
        let vectors = embeddings_bge_small_en_v15(vec![input]);
        vectors
            .into_iter()
            .next()
            .unwrap_or_pg_err("Unexpectedly empty result vector")
    }

    extension_sql!(
        "CREATE FUNCTION neon_ai.embedding_for_passage_bge_small_en_v15(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT neon_ai._embedding_bge_small_en_v15(input)::vector(384);
        $$;
        CREATE FUNCTION neon_ai.embedding_for_query_bge_small_en_v15(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT neon_ai._embedding_bge_small_en_v15('Represent this sentence for searching relevant passages: ' || input)::vector(384);
        $$;",
        name = "embedding_bge_small_en_v15",
    );

    // === local reranking ===

    pub fn reranks_jina_v1_tiny_en_base(query: &str, documents: Vec<String>) -> Vec<RerankResult> {
        thread_local! {
            static CELL: OnceCell<TextRerank> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let model = cell.get_or_init(|| {
                let user_def_model = local_model!(UserDefinedRerankingModel, "../jina_reranker_v1_tiny_en");
                TextRerank::try_new_from_user_defined(user_def_model, Default::default())
                    .expect_or_pg_err("Couldn't load model jina_reranker_v1_tiny_en")
            });
            let documents = documents.iter().map(String::as_str).collect();
            model
                .rerank(query, documents, false, None)
                .expect_or_pg_err("Unable to rerank with jina_reranker_v1_tiny_en")
        })
    }

    #[pg_extern(immutable, strict)]
    pub fn rerank_indices_jina_v1_tiny_en(query: &str, documents: Vec<String>) -> Vec<i32> {
        let reranking = reranks_jina_v1_tiny_en_base(query, documents);
        reranking.iter().map(|rr| rr.index as i32).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn rerank_scores_jina_v1_tiny_en(query: &str, documents: Vec<String>) -> Vec<f32> {
        let mut reranking = reranks_jina_v1_tiny_en_base(query, documents);
        reranking.sort_by(|rr1, rr2| rr1.index.cmp(&rr2.index)); // return to input order
        reranking.iter().map(|rr| rr.score as f32).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn rerank_score_jina_v1_tiny_en(query: &str, document: String) -> f32 {
        let scores = rerank_scores_jina_v1_tiny_en(query, vec![document]);
        let score = scores.first().unwrap_or_pg_err("Unexpectedly empty reranking vector");
        -*score
    }
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::neon_ai::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_embedding_bge_small_en_v15_length() {
        assert_eq!(_embedding_bge_small_en_v15("hello world!").len(), 384);
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_immutability() {
        assert_eq!(
            _embedding_bge_small_en_v15("hello world!"),
            _embedding_bge_small_en_v15("hello world!")
        );
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_variability() {
        assert_ne!(
            _embedding_bge_small_en_v15("hello world!"),
            _embedding_bge_small_en_v15("bye moon!")
        );
    }

    #[pg_test]
    fn test_rerank_jina_v1_tiny_en() {
        let candidate_pets = vec![
            "crocodile".to_owned(),
            "hamster".to_owned(),
            "indeterminate".to_owned(),
            "floorboard".to_owned(),
            "cat".to_owned(),
        ];
        let scores = rerank_scores_jina_v1_tiny_en("pet", candidate_pets.clone());
        let mut sorted_pets = candidate_pets.clone();
        sorted_pets.sort_by(|pet1, pet2| {
            let index1 = candidate_pets.iter().position(|pet| pet == pet1).unwrap();
            let index2 = candidate_pets.iter().position(|pet| pet == pet2).unwrap();
            scores[index1].partial_cmp(&scores[index2]).unwrap().reverse()
        });
        log!("{:?}", sorted_pets);
        assert!(sorted_pets == vec!["cat", "hamster", "crocodile", "floorboard", "indeterminate"]);
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

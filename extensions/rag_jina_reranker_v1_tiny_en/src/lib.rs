use pgrx::prelude::*;

mod errors;

pg_module_magic!();

#[pg_schema]
mod rag_jina_reranker_v1_tiny_en {
    use super::errors::*;
    use fastembed::{RerankResult, TextRerank, TokenizerFiles, UserDefinedRerankingModel};
    use pgrx::prelude::*;
    use std::cell::OnceCell;

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

    pub fn reranks(query: &str, documents: Vec<String>) -> Vec<RerankResult> {
        thread_local! {
            static CELL: OnceCell<TextRerank> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let model = cell.get_or_init(|| {
                let user_def_model = local_model!(UserDefinedRerankingModel, "../../../lib/jina_reranker_v1_tiny_en");
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
    pub fn rerank_indices(query: &str, documents: Vec<String>) -> Vec<i32> {
        let reranking = reranks(query, documents);
        reranking.iter().map(|rr| rr.index as i32).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn rerank_scores(query: &str, documents: Vec<String>) -> Vec<f32> {
        let mut reranking = reranks(query, documents);
        reranking.sort_by(|rr1, rr2| rr1.index.cmp(&rr2.index)); // return to input order
        reranking.iter().map(|rr| rr.score as f32).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn rerank_score(query: &str, document: String) -> f32 {
        let scores = rerank_scores(query, vec![document]);
        let score = scores.first().unwrap_or_pg_err("Unexpectedly empty reranking vector");
        -*score
    }
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_jina_reranker_v1_tiny_en::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_rerank() {
        let candidate_pets = vec![
            "crocodile".to_owned(),
            "hamster".to_owned(),
            "indeterminate".to_owned(),
            "floorboard".to_owned(),
            "cat".to_owned(),
        ];
        let scores = rerank_scores("pet", candidate_pets.clone());
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

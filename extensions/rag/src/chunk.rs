use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use pgrx::prelude::*;

    use std::cell::OnceCell;
    use text_splitter::{ChunkConfig, TextSplitter};
    use tokenizers::{AddedToken, Tokenizer};

    #[pg_extern(immutable, strict)]
    pub fn chunks_by_character_count(document: &str, max_characters: i32, max_overlap: i32) -> Vec<&str> {
        if max_characters < 1 || max_overlap < 0 {
            error!("{ERR_PREFIX} max_characters must be >= 1 and max_overlap must be >= 0");
        }

        let config = ChunkConfig::new(max_characters as usize)
            .with_overlap(max_overlap as usize)
            .expect_or_pg_err("Error creating chunk config");

        let splitter = TextSplitter::new(config);
        splitter.chunks(document).collect()
    }

    #[pg_extern(immutable, strict)]
    pub fn chunks_by_token_count_bge_small_en_v15(document: &str, max_tokens: i32, max_overlap: i32) -> Vec<&str> {
        thread_local! {
            static CELL: OnceCell<(Tokenizer, i32)> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let (tokenizer, model_max_length) = cell.get_or_init(|| {
                let mut tokenizer = Tokenizer::from_bytes(include_bytes!("../../../lib/bge_small_en_v15/tokenizer.json"))
                    .expect_or_pg_err("Error loading tokenizer");

                let special_tokens_map: serde_json::Value =
                    serde_json::from_slice(include_bytes!("../../../lib/bge_small_en_v15/special_tokens_map.json"))
                        .expect_or_pg_err("Error loading special tokens");

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

                let tokenizer_config: serde_json::Value =
                    serde_json::from_slice(include_bytes!("../../../lib/bge_small_en_v15/tokenizer_config.json"))
                        .expect_or_pg_err("Error loading tokenizer config");

                let model_max_length = tokenizer_config["model_max_length"]
                    .as_f64()
                    .unwrap_or_pg_err("Missing/invalid max model length in tokenizer config");

                (tokenizer, model_max_length as i32)
            });

            if !(max_tokens > 0
                && max_tokens <= *model_max_length
                && max_overlap >= 0
                && max_overlap < *model_max_length)
            {
                error!(
                    "{ERR_PREFIX} max_tokens must be between 1 and {}, and max_overlap must be between 0 and {}",
                    model_max_length,
                    model_max_length - 1
                );
            }

            let size_config = ChunkConfig::new(max_tokens as usize)
                .with_overlap(max_overlap as usize)
                .expect_or_pg_err("Error creating chunk config");

            let splitter = TextSplitter::new(size_config.with_sizer(tokenizer));
            splitter.chunks(document).collect()
        })
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    use super::rag::*;

    #[pg_test]
    fn test_chunk_by_characters() {
        assert_eq!(
            chunks_by_character_count(
                "The quick brown fox jumps over the lazy dog. In other news, the dish ran away with the spoon.",
                30,
                10
            ),
            vec![
                "The quick brown fox jumps over",
                "jumps over the lazy dog.",
                "In other news, the dish ran",
                "dish ran away with the spoon."
            ]
        );
    }

    #[pg_test]
    fn test_chunk_by_tokens() {
        assert_eq!(
            chunks_by_token_count_bge_small_en_v15(
                "The quick brown fox jumps over the lazy dog. In other news, the dish ran away with the spoon.",
                8,
                2
            ),
            vec![
                "The quick brown fox jumps over the lazy",
                "the lazy dog.",
                "In other news, the dish ran away",
                "ran away with the spoon."
            ]
        );
    }
}

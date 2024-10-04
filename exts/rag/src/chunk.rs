use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use pgrx::prelude::*;
    use text_splitter::{ChunkConfig, TextSplitter};

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
    fn test_chunk_by_characters_2() {
        assert_eq!(
            chunks_by_character_count(
                "The quick brown fox jumps over the lazy dog. In other news, the dish ran away with the spoon.",
                50,
                20
            ),
            vec![
                "The quick brown fox jumps over the lazy dog.",
                "In other news, the dish ran away with the spoon."
            ]
        );
    }

    #[pg_test]
    fn test_chunk_by_characters_empty() {
        assert_eq!(
            chunks_by_character_count(
                "",
                50,
                20
            ),
            vec![] as Vec<&str>
        );
    }
}

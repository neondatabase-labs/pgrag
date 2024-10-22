use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use super::super::json_api::*;
    use pgrx::prelude::*;
    use serde::{Deserialize, Serialize};

    // API key

    extension_sql!(
        "CREATE FUNCTION rag.voyageai_set_api_key(api_key text) RETURNS void
        LANGUAGE SQL VOLATILE STRICT AS $$
            INSERT INTO rag.config VALUES ('VOYAGEAI_KEY', api_key)
            ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value;
        $$;
        CREATE FUNCTION rag.voyageai_get_api_key() RETURNS text
        LANGUAGE SQL VOLATILE STRICT AS $$
            SELECT value FROM rag.config WHERE name = 'VOYAGEAI_KEY';
        $$;",
        name = "voyageai_api_key",
        requires = ["config"],
    );

    // embeddings

    #[derive(Serialize)]
    struct VoyageAIEmbeddingReq {
        model: String,
        input: String,
        input_type: Option<String>,
    }
    #[derive(Deserialize)]
    struct VoyageAIEmbeddingData {
        data: Vec<VoyageAIEmbedding>,
    }
    #[derive(Deserialize)]
    struct VoyageAIEmbedding {
        embedding: Vec<f32>,
    }

    #[pg_extern(immutable)]
    pub fn _voyageai_embedding(model: &str, input_type: Option<&str>, input: &str, key: &str) -> Vec<f32> {
        let body = VoyageAIEmbeddingReq {
            model: model.to_string(),
            input: input.to_string(),
            input_type: input_type.map(|s| s.to_string()),
        };
        let json = json_api("https://api.voyageai.com/v1/embeddings", Some(key), None, body);
        let embed_data: VoyageAIEmbeddingData =
            serde_json::from_value(json).expect_or_pg_err("Unexpected JSON structure in Voyage AI response");

        embed_data
            .data
            .into_iter()
            .next()
            .unwrap_or_pg_err("No embedding object in Voyage AI response")
            .embedding
    }

    extension_sql!(
        "CREATE TYPE rag.voyage_ai_input_type AS ENUM ('document', 'query');
        
        CREATE FUNCTION rag.voyageai_embedding(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE PLPGSQL IMMUTABLE AS $$
            DECLARE
                api_key text := rag.voyageai_get_api_key();
                res vector;
            BEGIN
                IF api_key IS NULL THEN
                    RAISE EXCEPTION '[rag] Voyage AI API key is not set';
                END IF;
                SELECT rag._voyageai_embedding(model, input_type, input, api_key)::vector INTO res;
                RETURN res;
            END;
        $$;

        CREATE FUNCTION rag.voyageai_embedding_3(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-3', input_type, input)::vector(1024);
        $$;
        CREATE FUNCTION rag.voyageai_embedding_3_lite(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-3-lite', input_type, input)::vector(512);
        $$;
        CREATE FUNCTION rag.voyageai_embedding_finance_2(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-finance-2', input_type, input)::vector(1024);
        $$;
        CREATE FUNCTION rag.voyageai_embedding_multilingual_2(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-multilingual-2', input_type, input)::vector(1024);
        $$;
        CREATE FUNCTION rag.voyageai_embedding_law_2(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-law-2', input_type, input)::vector(1024);
        $$;
        CREATE FUNCTION rag.voyageai_embedding_code_2(model text, input_type rag.voyage_ai_input_type, input text) RETURNS vector
        LANGUAGE SQL IMMUTABLE AS $$
          SELECT rag.voyageai_embedding('voyage-code-2', input_type, input)::vector(1536);
        $$;",
        name = "voyageai_embeddings",
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;
    use std::env;

    fn voyageai_api_key() -> String {
        match env::var("VOYAGEAI_API_KEY") {
            Err(err) => error!("Tests require environment variable VOYAGEAI_API_KEY: {}", err),
            Ok(key) => key,
        }
    }

    #[pg_test(error = "[rag] HTTP status code 401 trying to reach API: Provided API key is invalid.")]
    fn test_embedding_voyageai_bad_key() {
        _voyageai_embedding("voyage-3-lite", Some("document"), "hello world!", "invalid-key");
    }

    #[pg_test(error = "[rag] HTTP status code 400 trying to reach API: Model voyage-123-whizzo is not supported. Supported models are ['voyage-large-2-instruct', 'voyage-large-2-instruct-l4', 'voyage-law-2', 'voyage-code-2', 'voyage-02', 'voyage-2', 'voyage-01', 'voyage-lite-01', 'voyage-lite-01-instruct', 'voyage-lite-02-instruct', 'voyage-multilingual-2', 'voyage-large-2'].")]
    fn test_embedding_voyageai_bad_model() {
        _voyageai_embedding("voyage-123-whizzo", Some("query"), "hello world!", &voyageai_api_key());
    }

    #[pg_test]
    fn test_embedding_voyageai_has_data() {
        let embedding = _voyageai_embedding("voyage-3-lite", Some("document"), "hello world!", &voyageai_api_key());
        assert_eq!(embedding.len(), 512);
    }

    #[pg_test]
    fn test_embedding_voyageai_input_types() {
        let embedding_d = _voyageai_embedding("voyage-3-lite", Some("document"), "hello world!", &voyageai_api_key());
        let embedding_q = _voyageai_embedding("voyage-3-lite", Some("query"), "hello world!", &voyageai_api_key());
        let embedding_n = _voyageai_embedding("voyage-3-lite", None, "hello world!", &voyageai_api_key());
        assert_ne!(embedding_d, embedding_q);
        assert_ne!(embedding_d, embedding_n);
        assert_ne!(embedding_q, embedding_n);
    }

}

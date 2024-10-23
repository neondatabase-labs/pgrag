use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use super::super::json_api::*;
    use pgrx::prelude::*;
    use serde::{Deserialize, Serialize};

    extension_sql!(
        "CREATE FUNCTION rag.fireworks_set_api_key(api_key text) RETURNS void
        LANGUAGE SQL VOLATILE STRICT AS $$
            INSERT INTO rag.config VALUES ('FIREWORKS_KEY', api_key)
            ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value;
        $$;
        CREATE FUNCTION rag.fireworks_get_api_key() RETURNS text
        LANGUAGE SQL VOLATILE STRICT AS $$
            SELECT value FROM rag.config WHERE name = 'FIREWORKS_KEY';
        $$;",
        name = "fireworks_api_key",
        requires = ["config"],
    );

    // embeddings

    #[derive(Serialize)]
    struct FireworksAIEmbeddingReq {
        model: String,
        input: String,
    }
    #[derive(Deserialize)]
    struct FireworksAIEmbeddingData {
        data: Vec<FireworksAIEmbedding>,
    }
    #[derive(Deserialize)]
    struct FireworksAIEmbedding {
        embedding: Vec<f32>,
    }

    #[pg_extern(immutable, strict)]
    pub fn _fireworks_text_embedding(model: String, input: String, key: &str) -> Vec<f32> {
        let body = FireworksAIEmbeddingReq { model, input };
        let json = json_api("https://api.fireworks.ai/inference/v1/embeddings", Some(key), None, body);
        let embed_data: FireworksAIEmbeddingData =
            serde_json::from_value(json).expect_or_pg_err("Unexpected JSON structure in Fireworks AI response");

        embed_data
            .data
            .into_iter()
            .next()
            .unwrap_or_pg_err("No embedding object in Fireworks AI response")
            .embedding
    }

    extension_sql!(
        "CREATE FUNCTION rag.fireworks_text_embedding(model text, input text) RETURNS vector
        LANGUAGE PLPGSQL IMMUTABLE STRICT AS $$
            DECLARE
                api_key text := rag.fireworks_get_api_key();
                res vector;
            BEGIN
                IF api_key IS NULL THEN
                    RAISE EXCEPTION '[rag] Fireworks AI API key is not set';
                END IF;
                SELECT rag._fireworks_text_embedding(model, input, api_key)::vector INTO res;
                RETURN res;
            END;
        $$;
        CREATE FUNCTION rag.fireworks_nomic_embed_text_v15(input text) RETURNS vector(768)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
          SELECT rag.fireworks_text_embedding('nomic-ai/nomic-embed-text-v1.5', input)::vector(768);
        $$;
        CREATE FUNCTION rag.fireworks_nomic_embed_text_v1(input text) RETURNS vector(768)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
          SELECT rag.fireworks_text_embedding('nomic-ai/nomic-embed-text-v1', input)::vector(768);
        $$;
        CREATE FUNCTION rag.fireworks_embedding_whereisai_uae_large_v1(input text) RETURNS vector(1024)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
          SELECT rag.fireworks_text_embedding('WhereIsAI/UAE-Large-V1', input)::vector(1024);
        $$;
        CREATE FUNCTION rag.fireworks_embedding_thenlper_gte_large(input text) RETURNS vector(1024)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
          SELECT rag.fireworks_text_embedding('thenlper/gte-large', input)::vector(1024);
        $$;
        CREATE FUNCTION rag.fireworks_embedding_thenlper_gte_base(input text) RETURNS vector(768)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
          SELECT rag.fireworks_text_embedding('thenlper/gte-base', input)::vector(768);
        $$;",
        name = "fireworksai_embeddings",
    );

    // chat

    #[pg_extern(strict)]
    pub fn _fireworks_chat_completion(json_body: pgrx::Json, key: &str) -> pgrx::Json {
        let json = json_api(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            Some(key),
            None,
            json_body,
        );
        pgrx::Json(json)
    }

    extension_sql!(
        "CREATE FUNCTION rag.fireworks_chat_completion(body json) RETURNS json
        LANGUAGE PLPGSQL VOLATILE STRICT AS $$
            DECLARE
                api_key text := rag.fireworks_get_api_key();
                res json;
            BEGIN
                IF api_key IS NULL THEN
                    RAISE EXCEPTION '[rag] Fireworks API key is not set';
                END IF;
                SELECT rag._fireworks_chat_completion(body, api_key) INTO res;
                RETURN res;
            END;
        $$;",
        name = "fireworks_chat_completion",
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;
    use serde_json::json;
    use std::env;

    fn fireworks_api_key() -> String {
        match env::var("FIREWORKS_API_KEY") {
            Err(err) => error!("Tests require environment variable FIREWORKS_API_KEY: {}", err),
            Ok(key) => key,
        }
    }

    #[pg_test]
    fn test_fireworks_embedding_length1() {
        let embedding = _fireworks_text_embedding("nomic-ai/nomic-embed-text-v1.5".to_string(), "hello world!".to_string(), &fireworks_api_key());
        assert_eq!(embedding.len(), 768);
    }

    #[pg_test]
    fn test_fireworks_embedding_length2() {
        let embedding = _fireworks_text_embedding("nomic-ai/nomic-embed-text-v1".to_string(), "hello world!".to_string(), &fireworks_api_key());
        assert_eq!(embedding.len(), 768);
    }

    #[pg_test]
    fn test_fireworks_embedding_length3() {
        let embedding = _fireworks_text_embedding("WhereIsAI/UAE-Large-V1".to_string(), "hello world!".to_string(), &fireworks_api_key());
        assert_eq!(embedding.len(), 1024);
    }

    #[pg_test]
    fn test_fireworks_embedding_length4() {
        let embedding = _fireworks_text_embedding("thenlper/gte-large".to_string(), "hello world!".to_string(), &fireworks_api_key());
        assert_eq!(embedding.len(), 1024);
    }

    #[pg_test]
    fn test_fireworks_embedding_length5() {
        let embedding = _fireworks_text_embedding("thenlper/gte-base".to_string(), "hello world!".to_string(), &fireworks_api_key());
        assert_eq!(embedding.len(), 768);
    }

    #[pg_test(error = "[rag] HTTP status code 403 trying to reach API: unauthorized")]
    fn test_fireworks_bad_key() {
        // interestingly, Fireworks appear to parse the JSON payload before checking the key
        _fireworks_chat_completion(
            pgrx::Json {
                0: json!({
                    "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                    "messages": []
                }),
            },
            "invalid-key",
        );
    }

    #[pg_test(error = "[rag] HTTP status code 400 trying to reach API: Missing model name")]
    fn test_fireworks_bad_json() {
        _fireworks_chat_completion(
            pgrx::Json {
                0: json!({"whoosh": 12}),
            },
            "invalid-key",
        );
    }

    #[pg_test]
    fn test_fireworks_chat_completion() {
        let result = _fireworks_chat_completion(
            pgrx::Json {
                0: json!({
                    "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                    "messages":[
                        {
                            "role":"system",
                            "content":"you are a helpful assistant"
                        }, {
                            "role": "user",
                            "content": "hi!"
                        }
                    ]
                }),
            },
            &fireworks_api_key(),
        );
        assert!(result
            .0
            .as_object()
            .unwrap()
            .get("choices")
            .unwrap()
            .as_array()
            .unwrap()
            .get(0)
            .unwrap()
            .as_object()
            .unwrap()
            .get("message")
            .unwrap()
            .as_object()
            .unwrap()
            .get("content")
            .unwrap()
            .is_string());
    }
}

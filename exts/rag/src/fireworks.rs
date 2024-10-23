use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::json_api::*;
    use pgrx::prelude::*;

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

    #[pg_test(error = "[rag] HTTP status code 403 trying to reach API: unauthorized")]
    fn test_fireworks_bad_key() {
        // interestingly, Fireworks appear to parse the JSON payload before checking the key
        _fireworks_chat_completion(pgrx::Json { 0: json!({
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "messages": []
        }) }, "invalid-key");
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

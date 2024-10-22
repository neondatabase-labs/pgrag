use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::json_api::*;
    use pgrx::prelude::*;

    extension_sql!(
        "CREATE FUNCTION rag.anthropic_set_api_key(api_key text) RETURNS void
        LANGUAGE SQL VOLATILE STRICT AS $$
            INSERT INTO rag.config VALUES ('ANTHROPIC_KEY', api_key)
            ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value;
        $$;
        CREATE FUNCTION rag.anthropic_get_api_key() RETURNS text
        LANGUAGE SQL VOLATILE STRICT AS $$
            SELECT value FROM rag.config WHERE name = 'ANTHROPIC_KEY';
        $$;",
        name = "anthropic_api_key",
        requires = ["config"],
    );

    #[pg_extern(strict)]
    pub fn _anthropic_messages(version: &str, json_body: pgrx::Json, key: &str) -> pgrx::Json {
        let json = json_api(
            "https://api.anthropic.com/v1/messages",
            None,
            Some(vec![("x-api-key", key), ("anthropic-version", version)]),
            json_body,
        );
        pgrx::Json(json)
    }

    extension_sql!(
        "CREATE FUNCTION rag.anthropic_messages(version text, body json) RETURNS json
        LANGUAGE PLPGSQL VOLATILE STRICT AS $$
            DECLARE
                api_key text := rag.anthropic_get_api_key();
                res json;
            BEGIN
                IF api_key IS NULL THEN
                    RAISE EXCEPTION '[rag] Anthropic API key is not set';
                END IF;
                SELECT rag._anthropic_messages(version, body, api_key) INTO res;
                RETURN res;
            END;
        $$;",
        name = "anthropic_messages",
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;
    use serde_json::json;
    use std::env;

    fn anthropic_api_key() -> String {
        match env::var("ANTHROPIC_API_KEY") {
            Err(err) => error!("Tests require environment variable ANTHROPIC_API_KEY: {}", err),
            Ok(key) => key,
        }
    }

    #[pg_test(error = "[rag] HTTP status code 401 trying to reach API: invalid x-api-key")]
    fn test_anthropic_bad_key() {
        _anthropic_messages(
            "2023-06-01",
            pgrx::Json {
                0: json!({}),
            },
            "invalid-key",
        );
    }

    #[pg_test]
    fn test_anthropic_messages() {
        let result = _anthropic_messages(
            "2023-06-01",
            pgrx::Json {
                0: json!({
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 64,
                    "system": "you are a helpful assistant",
                    "messages":[
                        {
                            "role": "user",
                            "content": "hi!"
                        }
                    ]
                }),
            },
            &anthropic_api_key(),
        );
        assert!(result
            .0
            .as_object()
            .unwrap()
            .get("content")
            .unwrap()
            .as_array()
            .unwrap()
            .get(0)
            .unwrap()
            .as_object()
            .unwrap()
            .get("text")
            .unwrap()
            .is_string());
    }
}

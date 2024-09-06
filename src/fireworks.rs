use pgrx::prelude::*;

#[pg_schema]
mod neon_ai {
    use super::super::json_api::*;
    use pgrx::prelude::*;

    extension_sql!(
        "CREATE FUNCTION neon_ai.fireworks_set_api_key(api_key text) RETURNS void
        LANGUAGE SQL VOLATILE STRICT AS $$
            INSERT INTO neon_ai.config VALUES ('FIREWORKS_KEY', api_key)
            ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value;
        $$;
        CREATE FUNCTION neon_ai.fireworks_get_api_key() RETURNS text
        LANGUAGE SQL VOLATILE STRICT AS $$
            SELECT value FROM neon_ai.config WHERE name = 'FIREWORKS_KEY';
        $$;",
        name = "fireworks_api_key",
        requires = ["config"],
    );

    #[pg_extern(strict)]
    pub fn _fireworks_chat_completion(json_body: pgrx::Json, key: &str) -> pgrx::Json {
        let json = json_api("https://api.fireworks.ai/inference/v1/chat/completions", key, json_body);
        pgrx::Json(json)
    }

    extension_sql!(
        "CREATE FUNCTION neon_ai.fireworks_chat_completion(body json) RETURNS json
        LANGUAGE PLPGSQL VOLATILE STRICT AS $$
            DECLARE
                api_key text := neon_ai.fireworks_get_api_key();
                res json;
            BEGIN
                IF api_key IS NULL THEN
                    RAISE EXCEPTION '[neon_ai] Fireworks API key is not set';
                END IF;
                SELECT neon_ai._fireworks_chat_completion(body, api_key) INTO res;
                RETURN res;
            END;
        $$;",
        name = "fireworks_chat_completion",
    );
}

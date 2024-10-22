use pgrx::prelude::*;

mod anthropic;
mod chunk;
mod docx;
mod errors;
mod fireworks;
mod json_api;
mod markdown;
mod openai;
mod pdf;
mod voyageai;

pg_module_magic!();

#[pg_schema]
mod rag {
    use pgrx::prelude::*;

    extension_sql!(
        "CREATE TABLE rag.config(name text PRIMARY KEY, value text);
        REVOKE ALL ON TABLE rag.config FROM PUBLIC;",
        name = "config",
    );
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {}

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

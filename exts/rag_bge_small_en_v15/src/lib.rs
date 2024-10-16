pub mod embeddings {
    tonic::include_proto!("embeddings");
}

mod chunk;
mod errors;

use errors::*;
use pgrx::bgworkers::*;
use pgrx::prelude::*;
use std::cell::OnceCell;
use tonic::{transport::Server, Request, Response, Status};

use embeddings::embedding_generator_server::{EmbeddingGenerator, EmbeddingGeneratorServer};
use embeddings::{EmbeddingReply, EmbeddingRequest};

#[derive(Debug, Default)]
pub struct BgeSmallEnV15EmbeddingGenerator {}

#[tonic::async_trait]
impl EmbeddingGenerator for BgeSmallEnV15EmbeddingGenerator {
    async fn get_embedding(&self, request: Request<EmbeddingRequest>) -> Result<Response<EmbeddingReply>, Status> {
        let text = request.into_inner().text;
        let reply = EmbeddingReply { embedding: vec![1.0] };
        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}

pg_module_magic!();

/*
    In order to use this extension with pgrx, you'll need to edit `postgresql.conf` and add:

    ```
    shared_preload_libraries = 'rag_bge_small_en_v15.so'
    ```

    On Mac, you may need a `.dylib` path instead.

    Background workers **must** be initialized in the extension's `_PG_init()` function, and can **only**
    be started if loaded through the `shared_preload_libraries` configuration setting.
*/

thread_local! {
    static PIDCELL: OnceCell<i64> = OnceCell::new();
}

#[pg_guard]
pub extern "C" fn _PG_init() {
    let pid = std::process::id() as i64;
    PIDCELL.with(|cell| cell.set(pid).unwrap());

    BackgroundWorkerBuilder::new("rag_bge_small_en_v15-embeddings")
        .set_function("rag_bge_small_en_v15_background_main")
        .set_library("rag_bge_small_en_v15")
        .set_argument(pid.into_datum())
        .enable_spi_access()
        .load();
}

#[pg_guard]
#[no_mangle]
pub extern "C" fn rag_bge_small_en_v15_background_main(arg: pg_sys::Datum) {
    let pid = unsafe { i64::from_polymorphic_datum(arg, false, pg_sys::INT8OID).unwrap_or_pg_err("No PID received") };

    // if we don't attach the SIGTERM handler, we'll never be able to exit via an external notification
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);

    log!("{} started with PID argument: {}", BackgroundWorker::get_name(), pid);

    let addr = "[::1]:50051".parse().expect_or_pg_err("Couldn't parse socket address");
    let embedder = BgeSmallEnV15EmbeddingGenerator::default();

    Server::builder()
        .add_service(EmbeddingGeneratorServer::new(embedder))
        .serve(addr)
        .await?;

    // wake up every 10s or if we received a SIGTERM
    // while BackgroundWorker::wait_latch(Some(Duration::from_secs(10))) {
    //     if BackgroundWorker::sighup_received() {
    //         // on SIGHUP, you might want to reload some external configuration or something
    //     }

    //     // within a transaction, execute an SQL statement, and log its results
    //     let result: Result<(), pgrx::spi::Error> = BackgroundWorker::transaction(|| {
    //         Spi::connect(|client| {
    //             let tuple_table = client.select(
    //                 "SELECT 'Hi', id, ''||a FROM (SELECT id, 42 from generate_series(1,10) id) a ",
    //                 None,
    //                 None,
    //             )?;
    //             for tuple in tuple_table {
    //                 let a = tuple.get_datum_by_ordinal(1)?.value::<String>()?;
    //                 let b = tuple.get_datum_by_ordinal(2)?.value::<i32>()?;
    //                 let c = tuple.get_datum_by_ordinal(3)?.value::<String>()?;
    //                 log!("from bgworker: ({:?}, {:?}, {:?})", a, b, c);
    //             }
    //             Ok(())
    //         })
    //     });
    //     result.unwrap_or_else(|e| panic!("got an error: {}", e))
    // }

    log!("{} exited", BackgroundWorker::get_name());
}

#[pg_schema]
mod rag_bge_small_en_v15 {
    use super::errors::*;
    use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
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

    // NOTE. It might be nice to expose this function directly, but as at 2024-07-08 pgrx
    // doesn't support Vec<Vec<_>>: https://github.com/pgcentralfoundation/pgrx/issues/1762.

    // #[pg_extern(immutable, strict, name = "_embedding")]
    pub fn embeddings(input: Vec<&str>) -> Vec<Vec<f32>> {
        thread_local! {
            static CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
        }
        CELL.with(|cell| {
            let model = cell.get_or_init(|| {
                let user_def_model = local_model!(UserDefinedEmbeddingModel, "../../../lib/bge_small_en_v15");
                TextEmbedding::try_new_from_user_defined(user_def_model, Default::default())
                    .expect_or_pg_err("Couldn't load model bge_small_en_v15")
            });

            model
                .embed(input, None)
                .expect_or_pg_err("Unable to generate bge_small_en_v15 embeddings")
        })
    }

    #[pg_extern(immutable, strict)]
    pub fn _embedding(input: &str) -> Vec<f32> {
        let vectors = embeddings(vec![input]);
        vectors
            .into_iter()
            .next()
            .unwrap_or_pg_err("Unexpectedly empty result vector")
    }

    extension_sql!(
        "CREATE FUNCTION rag_bge_small_en_v15.embedding_for_passage(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding(input)::vector(384);
        $$;
        CREATE FUNCTION rag_bge_small_en_v15.embedding_for_query(input text) RETURNS vector(384)
        LANGUAGE SQL IMMUTABLE STRICT AS $$
            SELECT rag_bge_small_en_v15._embedding('Represent this sentence for searching relevant passages: ' || input)::vector(384);
        $$;",
        name = "embeddings",
    );
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_bge_small_en_v15::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_embedding_length() {
        assert_eq!(_embedding("hello world!").len(), 384);
    }

    #[pg_test]
    fn test_embedding_immutability() {
        assert_eq!(_embedding("hello world!"), _embedding("hello world!"));
    }

    #[pg_test]
    fn test_embedding_variability() {
        assert_ne!(_embedding("hello world!"), _embedding("bye moon!"));
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

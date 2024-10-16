use errors::*;
use pgrx::bgworkers::*;
use pgrx::prelude::*;
use std::cell::OnceCell;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use tokio::{
    net::UnixListener,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{transport::Server, Request, Response, Status};

pub mod embeddings {
    tonic::include_proto!("embeddings");
}
use embeddings::embedding_generator_server::{EmbeddingGenerator, EmbeddingGeneratorServer};
use embeddings::{EmbeddingReply, EmbeddingRequest};

mod chunk;
mod errors;

static SOCKET_PATH: &str = "/tmp/.s.pgrag_bge_small_en_v15";

#[derive(Debug, Default)]
pub struct BgeSmallEnV15EmbeddingGenerator {}

#[tonic::async_trait]
impl EmbeddingGenerator for BgeSmallEnV15EmbeddingGenerator {
    async fn get_embedding(&self, request: Request<EmbeddingRequest>) -> Result<Response<EmbeddingReply>, Status> {
        let text = request.into_inner().text;
        let reply = EmbeddingReply {
            embedding: vec![text.len() as f32],
        };
        Ok(Response::new(reply))
    }
}

pg_module_magic!();

thread_local! {
    static PID_CELL: OnceCell<i64> = OnceCell::new();
}

#[pg_guard]
pub extern "C" fn _PG_init() {
    let pid = std::process::id() as i64;
    PID_CELL.with(|cell| cell.set(pid).expect_or_pg_err("Couldn't store PID"));

    BackgroundWorkerBuilder::new("rag_bge_small_en_v15 embeddings")
        .set_function("rag_bge_small_en_v15_background_main")
        .set_library("rag_bge_small_en_v15")
        .set_argument(pid.into_datum())
        .enable_spi_access()
        .load();
}

#[pg_guard]
#[no_mangle]
pub extern "C" fn rag_bge_small_en_v15_background_main(arg: pg_sys::Datum) {
    // if we don't attach the SIGTERM handler, we'll never be able to exit via an external notification
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGTERM);

    let pid = unsafe { i64::from_polymorphic_datum(arg, false, pg_sys::INT8OID).unwrap_or_pg_err("No PID received") };
    let name = BackgroundWorker::get_name();
    log!("{name} started by PID {pid}");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let path = &format!("{SOCKET_PATH}.{pid}");
            fs::remove_file(path).unwrap_or_default(); // file may not exist: this is no problem
            let uds = UnixListener::bind(path).expect_or_pg_err(&format!("Couldn't create socket at {path}"));
            fs::set_permissions(path, fs::Permissions::from_mode(0o777))
                .expect_or_pg_err(&format!("Couldn't set permissions for {path}"));
            log!("{name} created socket {}", path);
            let uds_stream = UnixListenerStream::new(uds);
            let embedder = BgeSmallEnV15EmbeddingGenerator::default();

            Server::builder()
                .add_service(EmbeddingGeneratorServer::new(embedder))
                .serve_with_incoming_shutdown(uds_stream, async {
                    while !BackgroundWorker::sigterm_received() {
                        sleep(Duration::from_millis(1000)).await;
                    }
                    log!("{name} received SIGTERM, exiting");
                })
                .await
                .expect_or_pg_err("Couldn't await server");
        });

}

#[pg_schema]
mod rag_bge_small_en_v15 {
    use super::errors::*;
    use super::PID_CELL;
    use super::SOCKET_PATH;
    use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
    use hyper_util::rt::TokioIo;
    use pgrx::prelude::*;
    use std::cell::OnceCell;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};
    use tower::service_fn;

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

    use embeddings::embedding_generator_client::EmbeddingGeneratorClient;
    use embeddings::EmbeddingRequest;

    pub mod embeddings {
        tonic::include_proto!("embeddings");
    }

    #[pg_extern]
    fn bgwtest(text: &str) -> Vec<f32> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let channel = Endpoint::try_from("http://[::]:80") // URL is ignored but must be valid
                    .expect_or_pg_err("Failed to create endpoint")
                    .connect_with_connector(service_fn(|_: Uri| async {
                        let pid = PID_CELL
                            .with(|cell| cell.get().cloned())
                            .unwrap_or_pg_err("Embedding process PID not found");

                        let path = &format!("{SOCKET_PATH}.{pid}");

                        Ok::<_, std::io::Error>(TokioIo::new(
                            UnixStream::connect(path)
                                .await
                                .expect_or_pg_err("Failed to connect to stream"),
                        ))
                    }))
                    .await
                    .expect_or_pg_err("Failed to connect connector");

                let mut client = EmbeddingGeneratorClient::new(channel);
                let request = tonic::Request::new(EmbeddingRequest { text: text.to_string() });
                let response = client
                    .get_embedding(request)
                    .await
                    .expect_or_pg_err("Couldn't await response from embedding server");

                response.into_inner().embedding
            })
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

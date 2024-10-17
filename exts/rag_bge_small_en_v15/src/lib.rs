mod chunk;
mod errors;
mod embeddings {
    tonic::include_proto!("embeddings");
}

use const_format::formatcp;
use embeddings::embedding_generator_server::{EmbeddingGenerator, EmbeddingGeneratorServer};
use embeddings::{EmbeddingReply, EmbeddingRequest};
use errors::*;
use fastembed::{Pooling, QuantizationMode, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use pgrx::{bgworkers::*, prelude::*};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{cell::OnceCell, fs, os::unix::fs::PermissionsExt, sync::OnceLock};
use tokio::{
    net::UnixListener,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{transport::Server, Request, Response, Status};

const EXT_NAME: &str = "rag_bge_small_en_v15";
const SOCKET_NAME_PREFIX: &str = formatcp!("/tmp/.s.pgrag.{EXT_NAME}");

pg_module_magic!();

thread_local! {
    static PID_CELL: OnceCell<i64> = OnceCell::new();
}
static TEXT_EMBEDDING: OnceLock<TextEmbedding> = OnceLock::new();

macro_rules! local_tokenizer_files {
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
    ($model:ident, $folder:literal) => {
        $model::new(
            include_bytes!(concat!($folder, "/model.onnx")).to_vec(),
            local_tokenizer_files!($folder),
        )
        .with_pooling(Pooling::Cls)
        .with_quantization(QuantizationMode::Static)
    };
}

#[pg_guard]
pub extern "C" fn _PG_init() {
    let pid = std::process::id() as i64;
    PID_CELL.with(|cell| cell.set(pid).expect_or_pg_err("Couldn't store socket path"));

    BackgroundWorkerBuilder::new(formatcp!("{EXT_NAME} embeddings background worker"))
        .set_function(formatcp!("background_main"))
        .set_library(EXT_NAME)
        .set_argument(pid.into_datum())
        .enable_spi_access()
        .load();
}

pub struct EmbeddingGeneratorStruct {
    thread_pool: ThreadPool,
}

#[tonic::async_trait]
impl EmbeddingGenerator for EmbeddingGeneratorStruct {
    async fn get_embedding(&self, request: Request<EmbeddingRequest>) -> Result<Response<EmbeddingReply>, Status> {
        let text = request.into_inner().text;
        let model = TEXT_EMBEDDING.get_or_init(|| {
            let user_def_model = local_model!(UserDefinedEmbeddingModel, "../../../lib/bge_small_en_v15");
            TextEmbedding::try_new_from_user_defined(user_def_model, Default::default())
                .expect_or_pg_err("Couldn't load embedding model")
        });

        let (tx, rx) = tokio::sync::oneshot::channel();

        self.thread_pool.spawn(|| {
            tx.send(model.embed(vec![text], None)).expect("Channel send failed");
        });

        match rx.await {
            // the spawned task didn't complete
            Err(_) => Err(Status::internal("Embedding process crashed")),

            // the spawned embedding task completed with error
            Ok(Err(embed_error)) => Err(Status::internal(embed_error.to_string())),

            // the spawned embedding task completed successfully
            Ok(Ok(embeddings)) => {
                let embedding = embeddings.into_iter().next().unwrap_or_pg_err("Empty result vector");
                let reply = EmbeddingReply { embedding };
                Ok(Response::new(reply))
            }
        }
    }
}

#[pg_guard]
#[no_mangle]
pub extern "C" fn background_main(arg: pg_sys::Datum) {
    let pid = unsafe { i64::from_polymorphic_datum(arg, false, pg_sys::INT8OID).unwrap_or_pg_err("No PID received") };
    let name = BackgroundWorker::get_name();
    log!("{name} started, received PID {pid}");

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGTERM);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect_or_pg_err("Couldn't build runtime for server")
        .block_on(bg_worker_tonic_main(name, pid));
}

async fn bg_worker_tonic_main(name: &str, pid: i64) {
    let path = format!("{SOCKET_NAME_PREFIX}.{pid}");
    fs::remove_file(&path).unwrap_or_default(); // it's not an error if the file isn't there
    let uds = UnixListener::bind(&path).expect_or_pg_err(&format!("Couldn't create socket at {}", &path));
    fs::set_permissions(&path, fs::Permissions::from_mode(0o777))
        .expect_or_pg_err(&format!("Couldn't set permissions for {}", &path));
    log!("{} created socket {}", name, &path);

    let num_threads = match std::thread::available_parallelism() {
        Ok(cpu_count) => match cpu_count.get() {
            1 => 1,
            cpus => cpus - 1,
        },
        Err(_) => 0, // automatic:
    };
    log!("{} passing {} to num_threads()", name, num_threads);

    let uds_stream = UnixListenerStream::new(uds);
    let embedder = EmbeddingGeneratorStruct {
        thread_pool: ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect_or_pg_err("Couldn't build rayon thread pool"),
    };

    Server::builder()
        .add_service(EmbeddingGeneratorServer::new(embedder))
        .serve_with_incoming_shutdown(uds_stream, async {
            while !BackgroundWorker::sigterm_received() {
                sleep(Duration::from_secs(1)).await;
            }
        })
        .await
        .expect_or_pg_err("Couldn't create server");
}

#[pg_schema]
mod rag_bge_small_en_v15 {
    use super::{errors::*, PID_CELL, SOCKET_NAME_PREFIX};
    use hyper_util::rt::TokioIo;
    use pgrx::prelude::*;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};
    use tower::service_fn;

    use embeddings::embedding_generator_client::EmbeddingGeneratorClient;
    use embeddings::EmbeddingRequest;

    pub mod embeddings {
        tonic::include_proto!("embeddings");
    }

    #[pg_extern(immutable, strict)]
    pub fn _embedding(text: &str) -> Vec<f32> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect_or_pg_err("Couldn't build tokio runtime for client")
            .block_on(async {
                let channel = Endpoint::try_from("http://[::]:80") // URL must be valid but is ignored
                    .expect_or_pg_err("Failed to create endpoint")
                    .connect_with_connector(service_fn(|_: Uri| async {
                        let pid = PID_CELL.with(|cell| cell.get().unwrap_or_pg_err("Couldn't get socket name").clone());
                        let path = format!("{SOCKET_NAME_PREFIX}.{pid}").clone();

                        Ok::<_, std::io::Error>(TokioIo::new(
                            UnixStream::connect(&path)
                                .await
                                .expect_or_pg_err(&format!("Couldn't connect embedding worker stream {}", &path)),
                        ))
                    }))
                    .await
                    .expect_or_pg_err("Couldn't connect embedding worker channel");

                let mut client = EmbeddingGeneratorClient::new(channel);
                let request = tonic::Request::new(EmbeddingRequest { text: text.to_string() });
                let response = client
                    .get_embedding(request)
                    .await
                    .expect_or_pg_err("Couldn't get response from embedding worker");

                response.into_inner().embedding
            })
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

mod chunk;
mod errors;
mod embeddings {
    tonic::include_proto!("embeddings");
}
#[macro_use]
mod mconst;

use embeddings::{
    embedding_generator_server::{EmbeddingGenerator, EmbeddingGeneratorServer},
    EmbeddingReply, EmbeddingRequest,
};
use errors::*;
use fastembed::{Pooling, QuantizationMode, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use pgrx::{bgworkers::*, prelude::*};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs, os::unix::fs::PermissionsExt, sync::OnceLock};
use tokio::{
    net::UnixListener,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{transport::Server, Request, Response, Status};

// macros

mconst!(ext_name, "rag_bge_small_en_v15");
mconst!(model_path, "../../../lib/bge_small_en_v15/");

macro_rules! socket_path {
    ($pid:expr) => {
        format!(concat!("/tmp/.s.pgrag.", ext_name!(), ".{}"), $pid)
    };
}

// init

pg_module_magic!();

static PID: OnceLock<i64> = OnceLock::new();
static TEXT_EMBEDDING: OnceLock<TextEmbedding> = OnceLock::new();

#[pg_guard]
pub extern "C" fn _PG_init() {
    let pid = std::process::id() as i64;
    PID.set(pid)
        .expect_or_pg_err("Impossible concurrent access to set PID value");

    BackgroundWorkerBuilder::new(concat!(ext_name!(), " embeddings background worker"))
        .set_function("background_main")
        .set_library(ext_name!())
        .set_argument(pid.into_datum())
        .enable_spi_access()
        .load();
}

// model loading

macro_rules! local_tokenizer_files {
    () => {
        TokenizerFiles {
            tokenizer_file: include_bytes!(concat!(model_path!(), "tokenizer.json")).to_vec(),
            config_file: include_bytes!(concat!(model_path!(), "config.json")).to_vec(),
            special_tokens_map_file: include_bytes!(concat!(model_path!(), "special_tokens_map.json")).to_vec(),
            tokenizer_config_file: include_bytes!(concat!(model_path!(), "tokenizer_config.json")).to_vec(),
        }
    };
}

#[allow(dead_code)]
enum OnnxDownloadError {
    Transport(ureq::Transport),
    Status(u16),
    Io(std::io::Error),
}

#[cfg(not(feature = "remote_onnx"))]
fn get_onnx() -> Result<Vec<u8>, OnnxDownloadError> {
    log!("{ERR_PREFIX} Using embedded ONNX model");
    Ok(include_bytes!(concat!(model_path!(), "model.onnx")).to_vec())
}

#[cfg(feature = "remote_onnx")]
fn get_onnx() -> Result<Vec<u8>, OnnxDownloadError> {
    let url = env!("REMOTE_ONNX_URL");
    log!("{ERR_PREFIX} Downloading ONNX model {url} ...");
    let mut bytes: Vec<u8> = Vec::with_capacity(133_093_490);
    match ureq::get(url).call() {
        Err(ureq::Error::Transport(transport)) => Err(OnnxDownloadError::Transport(transport)),
        Err(ureq::Error::Status(status, _)) => Err(OnnxDownloadError::Status(status)),
        Ok(response) => match response.into_reader().read_to_end(&mut bytes) {
            Err(read_error) => Err(OnnxDownloadError::Io(read_error)),
            Ok(bytes_read) => {
                log!("{ERR_PREFIX} ONNX model downloaded ({bytes_read} bytes)");
                Ok(bytes)
            }
        },
    }
}

// background worker

pub struct EmbeddingGeneratorStruct {
    thread_pool: ThreadPool,
}

fn get_panic_message(panic: &Box<dyn std::any::Any + Send>) -> Option<&str> {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&'static str>().map(std::ops::Deref::deref))
}

#[tonic::async_trait]
impl EmbeddingGenerator for EmbeddingGeneratorStruct {
    async fn get_embedding(&self, request: Request<EmbeddingRequest>) -> Result<Response<EmbeddingReply>, Status> {
        let text = request.into_inner().text;

        // note: using panic!() and catching it is frowned on as a form of ordinary error handling,
        // but we need to ensure that the .get_or_init() only initialize TEXT_EMBEDDING with a TextEmbedding,
        // otherwise transient network errors become permanent

        let model: &TextEmbedding = match std::panic::catch_unwind(|| {
            TEXT_EMBEDDING.get_or_init(|| {
                let onnx = match get_onnx() {
                    Err(OnnxDownloadError::Transport(transport)) => {
                        let msg = transport.to_string();
                        panic!("Transport error downloading ONNX: {}", msg);
                    }
                    Err(OnnxDownloadError::Status(status)) => {
                        panic!("HTTP status {status} downloading ONNX");
                    }
                    Err(OnnxDownloadError::Io(err)) => {
                        let msg = err.to_string();
                        panic!("IO error downloading ONNX: {msg}");
                    }
                    Ok(onnx) => onnx,
                };

                let user_def_model = UserDefinedEmbeddingModel::new(onnx, local_tokenizer_files!())
                    .with_pooling(Pooling::Cls)
                    .with_quantization(QuantizationMode::Static);

                match TextEmbedding::try_new_from_user_defined(user_def_model, Default::default()) {
                    Err(err) => panic!("Couldn't create embedding model from downloaded file: {}", err),
                    Ok(model) => model,
                }
            })
        }) {
            Err(cause) => {
                let msg = get_panic_message(&cause).unwrap_or("Unknown error while loading embedding model");
                return Err(Status::internal(msg));
            }
            Ok(result) => result,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();

        self.thread_pool.spawn(|| {
            let embeddings = model.embed(vec![text], None);
            tx.send(embeddings).expect("Channel send failed");
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
    log!("{ERR_PREFIX} {name} started, received PID {pid}");

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGTERM);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect_or_pg_err("Couldn't build runtime for server")
        .block_on(bg_worker_tonic_main(name, pid));
}

async fn bg_worker_tonic_main(name: &str, pid: i64) {
    let path = socket_path!(pid);
    fs::remove_file(&path).unwrap_or_default(); // it's not an error if the file isn't there
    let uds = UnixListener::bind(&path).expect_or_pg_err(&format!("Couldn't create socket at {}", &path));
    fs::set_permissions(&path, fs::Permissions::from_mode(0o777))
        .expect_or_pg_err(&format!("Couldn't set permissions for {}", &path));
    log!("{ERR_PREFIX} {} created socket {}", name, &path);

    let num_threads = match std::thread::available_parallelism() {
        Ok(cpu_count) => match cpu_count.get() {
            1 => 1,
            cpus => cpus - 1,
        },
        Err(_) => 0, // automatic:
    };
    log!("{ERR_PREFIX} {} setting num_threads({})", name, num_threads);

    let uds_stream = UnixListenerStream::new(uds);
    let embedder = EmbeddingGeneratorStruct {
        thread_pool: ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect_or_pg_err("Couldn't build thread pool"),
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

// extension function(s)

#[pg_schema]
mod rag_bge_small_en_v15 {
    use super::{errors::*, PID};
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
                        let pid = PID.get().unwrap_or_pg_err("Couldn't get PID");
                        let path = socket_path!(pid);

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

mod errors;
mod reranking {
    tonic::include_proto!("reranking");
}
#[macro_use]
mod util;

use errors::*;
use fastembed::{TextRerank, TokenizerFiles, UserDefinedRerankingModel};
use pgrx::{bgworkers::*, prelude::*};
use rayon::{ThreadPool, ThreadPoolBuilder};
use reranking::{
    reranker_server::{Reranker, RerankerServer},
    RerankingReply, RerankingRequest,
};
use std::{fs, os::unix::fs::PermissionsExt, sync::OnceLock};
use tokio::{
    net::UnixListener,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{transport::Server, Request, Response, Status};

// macros

mconst!(ext_name, "rag_jina_reranker_v1_tiny_en");
mconst!(model_path, "../../../lib/jina_reranker_v1_tiny_en/");

macro_rules! socket_path {
    ($pid:expr) => {
        format!(concat!("/tmp/.s.pgrag.", ext_name!(), ".{}"), $pid)
    };
}

// init

pg_module_magic!();

static PID: OnceLock<i64> = OnceLock::new();
static TEXT_RERANK: tokio::sync::OnceCell<TextRerank> = tokio::sync::OnceCell::const_new();

#[pg_guard]
pub extern "C" fn _PG_init() {
    let pid = std::process::id() as i64;
    PID.set(pid)
        .expect_or_pg_err("Impossible concurrent access to set PID value");

    BackgroundWorkerBuilder::new(concat!(ext_name!(), " reranking background worker"))
        .set_function("background_main")
        .set_library(ext_name!())
        .set_argument(pid.into_datum())
        .enable_spi_access()
        .load();
}

// model loading

#[cfg(not(feature = "remote_onnx"))]
async fn get_onnx() -> Result<Vec<u8>, reqwest::Error> {
    Ok(include_bytes!(concat!(model_path!(), "model.onnx")).to_vec())
}

#[cfg(feature = "remote_onnx")]
async fn get_onnx() -> Result<Vec<u8>, reqwest::Error> {
    let url = env!("REMOTE_ONNX_URL");
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

// background worker

pub struct RerankerStruct {
    thread_pool: ThreadPool,
}

#[tonic::async_trait]
impl Reranker for RerankerStruct {
    async fn rerank(&self, request: Request<RerankingRequest>) -> Result<Response<RerankingReply>, Status> {
        let request = request.into_inner();
        let query = request.query;
        let passage = request.passage;

        let model = match TEXT_RERANK
            .get_or_try_init(|| async {
                let onnx_file = get_onnx().await?;
                let tokenizer_files = TokenizerFiles {
                    tokenizer_file: include_bytes!(concat!(model_path!(), "tokenizer.json")).to_vec(),
                    config_file: include_bytes!(concat!(model_path!(), "config.json")).to_vec(),
                    special_tokens_map_file: include_bytes!(concat!(model_path!(), "special_tokens_map.json")).to_vec(),
                    tokenizer_config_file: include_bytes!(concat!(model_path!(), "tokenizer_config.json")).to_vec(),
                };
                let user_def_model = UserDefinedRerankingModel {
                    onnx_file,
                    tokenizer_files,
                };

                TextRerank::try_new_from_user_defined(user_def_model, Default::default())
            })
            .await
        {
            Err(err) => return Err(Status::internal(err.to_string())),
            Ok(model) => model,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread_pool.spawn(|| {
            let reranking = model.rerank(query, vec![passage], false, None);
            tx.send(reranking).expect("Channel send failed");
        });

        match rx.await {
            Err(_) => Err(Status::internal("Reranking process crashed")),
            Ok(Err(rerank_error)) => Err(Status::internal(rerank_error.to_string())),
            Ok(Ok(rerankings)) => {
                let score = rerankings
                    .into_iter()
                    .next()
                    .unwrap_or_pg_err("Empty result vector")
                    .score;

                let reply = RerankingReply { score };
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
        .expect_or_pg_err("Couldn't build tokio runtime for server")
        .block_on(async {
            let path = socket_path!(pid);
            fs::remove_file(&path).unwrap_or_default(); // it's not an error if the file isn't there
            let uds = UnixListener::bind(&path).expect_or_pg_err(&format!("Couldn't create socket at {}", &path));
            fs::set_permissions(&path, fs::Permissions::from_mode(0o777))
                .expect_or_pg_err(&format!("Couldn't set permissions for {}", &path));
            log!("{ERR_PREFIX} {} created socket {}", name, &path);

            let num_threads = match std::thread::available_parallelism() {
                Err(_) => 0, // automatic
                Ok(cpu_count) => match cpu_count.get() {
                    1 => 1,
                    cpus => cpus - 1,
                },
            };
            let reranker = RerankerStruct {
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .expect_or_pg_err("Couldn't build thread pool"),
            };
            log!("{ERR_PREFIX} {} requested num_threads({})", name, num_threads);

            let uds_stream = UnixListenerStream::new(uds);
            Server::builder()
                .add_service(RerankerServer::new(reranker))
                .serve_with_incoming_shutdown(uds_stream, async {
                    while !BackgroundWorker::sigterm_received() {
                        sleep(Duration::from_millis(500)).await;
                    }
                })
                .await
                .expect_or_pg_err("Couldn't create server");
        });
}

// extension function(s)

#[pg_schema]
mod rag_jina_reranker_v1_tiny_en {
    pub mod reranking {
        tonic::include_proto!("reranking");
    }

    use super::{errors::*, PID};
    use hyper_util::rt::TokioIo;
    use pgrx::prelude::*;
    use reranking::reranker_client::RerankerClient;
    use reranking::RerankingRequest;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};
    use tower::service_fn;

    #[pg_extern(immutable, strict)]
    pub fn rerank_distance(query: &str, passage: &str) -> f32 {
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
                                .expect_or_pg_err(&format!("Couldn't connect worker stream {}", &path)),
                        ))
                    }))
                    .await
                    .expect_or_pg_err("Couldn't connect worker channel");

                let mut client = RerankerClient::new(channel);
                let request = tonic::Request::new(RerankingRequest {
                    query: query.to_string(),
                    passage: passage.to_string(),
                });
                let response = client
                    .rerank(request)
                    .await
                    .expect_or_pg_err("Couldn't get response from worker");

                -response.into_inner().score // for distance, lower numbers mean more similarity
            })
    }
}

// === Tests ===

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag_jina_reranker_v1_tiny_en::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_rerank_1() {
        let similar_distance = rerank_distance("cat", "dog");
        let dissimilar_distance = rerank_distance("cat", "pirate");
        assert!(similar_distance < dissimilar_distance);
    }

    #[pg_test]
    fn test_rerank_2() {
        let candidate_pets = vec![
            "crocodile".to_owned(),
            "hamster".to_owned(),
            "indeterminate".to_owned(),
            "floorboard".to_owned(),
            "cat".to_owned(),
        ];

        let mut scored_pets: Vec<(&String, f32)> = candidate_pets
            .iter()
            .map(|pet| (pet, rerank_distance("pet", pet)))
            .collect();

        scored_pets.sort_by(|pet1, pet2| pet1.1.partial_cmp(&(pet2.1)).unwrap());

        let ordered_pets: Vec<&String> = scored_pets.iter().map(|pet| pet.0).collect();
        assert!(ordered_pets == vec!["cat", "hamster", "crocodile", "floorboard", "indeterminate"]);
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

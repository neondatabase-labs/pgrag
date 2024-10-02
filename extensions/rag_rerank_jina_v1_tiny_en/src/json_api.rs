use crate::errors::*;
use pgrx::prelude::*;
use ureq::serde_json::*;

pub fn json_api(endpoint: &str, key: &str, body: impl serde::Serialize) -> Value {
    let auth = format!("Bearer {key}");
    let req = ureq::post(endpoint).set("Authorization", auth.as_str());

    match req.send_json(body) {
        Err(ureq::Error::Transport(err)) => {
            let msg = err.message().unwrap_or("no further details");
            error!("{ERR_PREFIX} Transport error communicating with API: {msg}");
        }
        Err(ureq::Error::Status(code, _)) => {
            error!("{ERR_PREFIX} HTTP status code {code} trying to reach API")
        }
        Ok(response) => response
            .into_json()
            .expect_or_pg_err(&format!("Failed to parse JSON received from {endpoint}")),
    }
}

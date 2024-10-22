use crate::errors::*;
use pgrx::prelude::*;
use serde::Deserialize;
use ureq::serde_json::*;

#[derive(Deserialize)]
struct JSONErrResponse {
    error: JSONErrObject,
}
#[derive(Deserialize)]
struct JSONErrObject {
    message: String,
}
#[derive(Deserialize)]
struct JSONAltErrResponse {
    detail: String,
}

pub fn json_api(
    endpoint: &str,
    key: Option<&str>,
    headers: Option<Vec<(&str, &str)>>,
    body: impl serde::Serialize,
) -> Value {
    let mut req = ureq::post(endpoint);
    if let Some(key) = key {
        let auth = format!("Bearer {key}");
        req = req.set("Authorization", auth.as_str());
    }
    if let Some(headers) = headers {
        for (key, value) in headers {
            req = req.set(key, value);
        }
    }
    match req.send_json(body) {
        Err(ureq::Error::Transport(err)) => {
            let msg = err.message().unwrap_or("no further details");
            log!("{ERR_PREFIX} Error from {endpoint}: {msg}");
            error!("{ERR_PREFIX} Transport error communicating with API");
        }
        Err(ureq::Error::Status(code, response)) => {
            let json: std::io::Result<serde_json::Value> = response.into_json();
            let msg = match json {
                Err(_) => "unparseable response".to_string(),
                Ok(json) => match serde_json::from_value::<JSONErrResponse>(json.clone()) {
                    Err(_) => match serde_json::from_value::<JSONAltErrResponse>(json) {
                        Err(_) => "no further details".to_string(),
                        Ok(json) => json.detail,
                    },
                    Ok(json) => json.error.message,
                },
            };
            error!("{ERR_PREFIX} HTTP status code {code} trying to reach API: {msg}")
        }
        Ok(response) => response
            .into_json()
            .expect_or_pg_err(&format!("Failed to parse response received from API as JSON")),
    }
}

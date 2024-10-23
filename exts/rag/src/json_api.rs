use crate::errors::*;
use pgrx::prelude::*;
use serde::Deserialize;
use ureq::serde_json::*;

#[derive(Deserialize)]
struct Err1 {
    error: JSONErrObject,
}
#[derive(Deserialize)]
struct JSONErrObject {
    message: String,
}
#[derive(Deserialize)]
struct Err2 {
    detail: String,
}
#[derive(Deserialize)]
struct Err3 {
    error: String,
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
            let json: std::io::Result<Value> = response.into_json();
            let msg = match json {
                Err(_) => "no further details".to_string(),
                Ok(json) => from_value::<Err1>(json.clone())
                    .and_then(|obj| Ok(obj.error.message))
                    .or_else(|_| from_value::<Err2>(json.clone()).and_then(|obj| Ok(obj.detail)))
                    .or_else(|_| from_value::<Err3>(json).and_then(|obj: Err3| Ok(obj.error)))
                    .unwrap_or("no further details".to_string()),
            };
            error!("{ERR_PREFIX} HTTP status code {code} trying to reach API: {msg}")
        }
        Ok(response) => response
            .into_json()
            .expect_or_pg_err(&format!("Failed to parse response received from API as JSON")),
    }
}

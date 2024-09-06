use pgrx::prelude::*;

pub const ERR_PREFIX: &'static str = "[neon_ai]";

pub trait UnwrapPgErrExt<T> {
    fn unwrap_or_pg_err(self, msg: &str) -> T;
}

impl<T> UnwrapPgErrExt<T> for Option<T> {
    fn unwrap_or_pg_err(self, msg: &str) -> T {
        match self {
            None => error!("{ERR_PREFIX} {msg}"),
            Some(value) => value,
        }
    }
}

pub trait ExpectPgErrExt<T, E: std::fmt::Display> {
    fn expect_or_pg_err(self, msg: &str) -> T;
}

impl<T, E: std::fmt::Display> ExpectPgErrExt<T, E> for Result<T, E> {
    fn expect_or_pg_err(self, msg: &str) -> T {
        match self {
            Err(err) => error!("{ERR_PREFIX} {msg}: {err}"),
            Ok(value) => value,
        }
    }
}

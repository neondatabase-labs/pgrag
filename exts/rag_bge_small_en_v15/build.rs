use std::{env, path::PathBuf, process::Command};
use bindgen::Builder;
use cc::Build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=signature_check.c");
    println!("cargo:rerun-if-env-changed=PGRX_PG_CONFIG");
    println!("cargo:rerun-if-env-changed=PG_CONFIG");

    let pg_config = env::var("PGRX_PG_CONFIG")
        .or_else(|_| env::var("PG_CONFIG"))
        .unwrap_or_else(|_| "pg_config".into());
    let output = Command::new(&pg_config)
        .arg("--includedir-server")
        .output()
        .expect("failed to run pg_config");
    let pg_inc = String::from_utf8(output.stdout).unwrap().trim().to_string();

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I."))
        .header("bindgen_pmsignal.h")
        .allowlist_function("PostmasterIsAliveInternal")
        .generate()
        .expect("bindgen failed");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("couldn't write bindings");

    cc::Build::new()
        .include(&pg_inc)
        .file("signature_check.c")
        // you can silence warnings:
        .warnings(false)
        .compile("signature_check");

  tonic_build::compile_protos("proto/embeddings.proto")?;
  Ok(())
}

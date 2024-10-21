#!/usr/bin/bash
# example compilation script for Ubuntu 24.04

# allow ~24GB disk and ~1GB RAM per core for compiling

sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential pkg-config libssl-dev libreadline-dev zlib1g-dev libicu-dev libclang-dev protobuf-compiler cmake

# rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# pgrx + pg
cargo install --locked cargo-pgrx@0.12.6
cargo pgrx init --pg16 download
export PATH=/home/ubuntu/.pgrx/16.4/pgrx-install/bin:$PATH

# pgvector
wget https://github.com/pgvector/pgvector/archive/refs/tags/v0.7.4.tar.gz -O pgvector-0.7.4.tar.gz
tar xzf pgvector-0.7.4.tar.gz
cd pgvector-0.7.4
make
make install
cd ..

# ONNX
wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.19.2.tar.gz -O onnxruntime-1.19.2.tar.gz
tar xzf onnxruntime-1.19.2.tar.gz
cd onnxruntime-1.19.2
./build.sh --config Release --parallel --skip_submodule_sync --skip_tests --allow_running_as_root

# pgrag
git clone https://github.com/neondatabase/pgrag.git
cd pgrag
cd lib/bge_small_en_v15 && tar xzf model.onnx.tar.gz && cd ../..
cd lib/jina_reranker_v1_tiny_en && tar xzf model.onnx.tar.gz && cd ../..
cd exts/rag
cargo pgrx install --release
cd ../rag_bge_small_en_v15
ORT_LIB_LOCATION=/home/ubuntu/onnxruntime-1.19.2/build/Linux cargo pgrx install --release
cd ../rag_jina_reranker_v1_tiny_en
ORT_LIB_LOCATION=/home/ubuntu/onnxruntime-1.19.2/build/Linux cargo pgrx install --release
cd ../rag

echo "shared_preload_libraries = 'rag_bge_small_en_v15.so'" >>  ~/.pgrx/data-16/postgresql.conf

cargo pgrx start
echo 'create database rag;' | psql -h ~/.pgrx postgres

echo '
  drop extension if exists rag cascade;
  create extension rag cascade;
  drop extension if exists rag_bge_small_en_v15 cascade;
  create extension rag_bge_small_en_v15;
  drop extension if exists rag_jina_reranker_v1_tiny_en cascade;
  create extension rag_jina_reranker_v1_tiny_en;
' | psql -h ~/.pgrx rag

echo "
  select rag.markdown_from_html('<p>Hello <i>world</i></p>');
  select rag_bge_small_en_v15.embedding_for_passage('hello world');
  select rag_jina_reranker_v1_tiny_en.rerank_distance('hello', 'goodbye');
" | psql -h ~/.pgrx rag

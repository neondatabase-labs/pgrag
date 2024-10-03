# Experimental Postgres extensions for end-to-end Retrieval-Augmented Generation (RAG)

Experimental extensions to support RAG within Postgres. Currently offers:


### Text extraction and conversion

* Simple text extraction from PDF documents using [pdf-extract](https://github.com/jrmuizel/pdf-extract). Currently no OCR, and no support for complex layout or formatting.

* Simple text extraction from .docx documents using [docx-rs](https://github.com/cstkingkey/docx-rs) (docx-rust). Currently no support for complex layout or formatting.

* HTML conversion to Markdown using [htmd](https://github.com/letmutex/htmd).


### Text chunking

* Text chunking by character count using [text-splitter](https://github.com/benbrandt/text-splitter).

* Text chunking by token count (tokenising for [bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5) -- see below), again using [text-splitter](https://github.com/benbrandt/text-splitter).


### Local embedding and reranking models

* Local tokenising + embedding generation with 33M parameter model [bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5) using [fastembed](https://github.com/Anush008/fastembed-rs).

* Local tokenising + reranking with 33M parameter model [jina-reranker-v1-tiny-en](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en) using [fastembed](https://github.com/Anush008/fastembed-rs).


### Remote embedding and chat models

* Querying OpenAI API for embeddings (e.g. `text-embedding-3-small`) and chat completions (e.g. `gpt-4o-mini`).

* Querying Fireworks.ai API for chat completions (e.g. `llama-v3p1-8b-instruct`).


## Installation

First, you'll need to install `pgvector`. For example:

```
wget https://github.com/pgvector/pgvector/archive/refs/tags/v0.7.4.tar.gz
tar xvzf v0.7.4.tar.gz
cd pgvector-0.7.4
export PG_CONFIG=/path/to/pg_config  # should actually end with pg_config
make
make install
```

Next, download the extension source, and uncompress the model files:

* `cd lib/bge_small_en_v15 && tar xzf model.onnx.tar.gz && cd ../..`
* `cd lib/jina_reranker_v1_tiny_en && tar xzf model.onnx.tar.gz && cd ../..`

Then (with Rust installed):

* `cargo install --locked cargo-pgrx@0.12.5`

Finally, inside one of the three folders inside `extensions`:

* `PG_CONFIG=/path/to/pg_config cargo pgrx install --release`


## Installation notes

* The `ort` package supplies precompiled binaries for the ONNX runtime. On some platforms, this may give rise to `undefined symbol` errors. In that case, you'll need to compile an ONNX runtime v18 yourself. On Debian, that looks something like this:

```bash
apt-get update && apt-get install -y build-essential python3 python3-pip
python3 -m pip install cmake
wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.18.1.tar.gz -O onnxruntime.tar.gz
mkdir onnxruntime-src && cd onnxruntime-src && tar xzf ../onnxruntime.tar.gz --strip-components=1 -C .
./build.sh --config Release --parallel --skip_submodule_sync --skip_tests --allow_running_as_root
```

And then when it comes to install the extension:

```bash
ORT_LIB_LOCATION=/home/user/onnxruntime-src/build/Linux cargo pgrx install --release
```

* The `ort` and `ort-sys` packages are drawn from a local folder using `[patch.crates-io]` in `Cargo.toml` because (as at 2024-09-06) otherwise we can end up with `ort@2.0.0-rc.4` and `ort-sys@2.0.0-rc.5`, and this mismatch ends badly.


## Usage

```sql
create extension if not exists rag cascade;  -- `cascade` installs pgvector dependency
create extension if not exists rag_embed_bge_small_en_v15 cascade; 
create extension if not exists rag_rerank_jina_v1_tiny_en cascade; 
```


#### `markdown_from_html(text) -> text`

Locally convert HTML to Markdown:

```sql
select rag.markdown_from_html('<html><body><h1>Title</h1><p>A <i>very</i> short paragraph</p><p>Another paragraph</p></body></html>');
--  '# Title\n\nA _very_ short paragraph\n\nAnother paragraph'
```


#### `text_from_pdf(bytea) -> text`

Locally extract text from a PDF:

```sql
\set contents `base64 < /path/to/your.pdf`
select rag.text_from_pdf(decode(:'contents', 'base64'));
-- 'Text content of PDF'
```


#### `text_from_docx(bytea) -> text`

Locally extract text from a .docx file:

```sql
\set contents `base64 < /path/to/your.docx`
select rag.text_from_docx(decode(:'contents', 'base64'));
-- 'Text content of .docx'
```


#### `chunks_by_character_count(text, max_characters integer, max_overlap_characters integer) -> text[]`

Locally chunk text using character count, with max and overlap:

```sql
select rag.chunks_by_character_count('The quick brown fox jumps over the lazy dog', 20, 4);
-- {"The quick brown fox","fox jumps over the","the lazy dog"}
```


#### `chunks_by_token_count_bge_small_en_v15(text, max_tokens integer, max_overlap_tokens integer) -> text[]`

Locally chunk text using token count for `bge_small_en_v15` embeddings, with max and overlap:

```sql
select rag.chunks_by_token_count_bge_small_en_v15('The quick brown fox jumps over the lazy dog', 4, 1);
-- {"The quick brown fox","fox jumps over the","the lazy dog"}
```


#### `embedding_for_passage_bge_small_en_v15(text) -> vector(384)` and `embedding_for_query_bge_small_en_v15(text) -> vector(384)`

Locally tokenize + generate embeddings using a small (33M param) model:

```sql
select rag_embed_bge_small_en_v15.embedding_for_passage('The quick brown fox jumps over the lazy dog');
-- [-0.1047543,-0.02242211,-0.0126493685, ...]
select rag_embed_bge_small_en_v15.embedding_for_query('What did the quick brown fox jump over?');
-- [-0.09328926,-0.030567117,-0.027558783, ...]
```


#### `rerank_score_jina_v1_tiny_en(text, text) -> real`

Locally tokenize + rerank original texts using a small (33M param) model:

```sql
select rag_rerank_jina_v1_tiny_en.rerank_score('The quick brown fox jumps over the lazy dog', 'What did the quick brown fox jump over?');
-- -1.1093962

select rag_rerank_jina_v1_tiny_en.rerank_score('The quick brown fox jumps over the lazy dog', 'Never Eat Shredded Wheat');
-- 1.4725753
```


#### `openai_set_api_key(text)` and `openai_get_api_key() -> text`

Store and retrieve your OpenAI API key:

```sql
select rag.openai_set_api_key('sk-proj-...');
select rag.openai_get_api_key();
-- 'sk-proj-...'
```


#### `openai_text_embedding_3_small(text) -> vector(1536)`, `openai_text_embedding_3_large(text) -> vector(3072)`, `openai_text_embedding_ada_002(text) -> vector(1536)`, and `openai_text_embedding(model text, text) -> vector`

Call out to OpenAI embeddings API (makes network request):

```sql
select rag.openai_text_embedding_3_small('The quick brown fox jumps over the lazy dog');
-- {-0.020836005,-0.016921125,-0.00450666, ...}
```


#### `openai_chat_completion(json) -> json`

Call out to OpenAI chat/completions API (makes network request):

```sql
select rag.openai_chat_completion('{"model":"gpt-4o-mini","messages":[{"role":"system","content":"you are a helpful assistant"},{"role":"user","content":"hi!"}]}');
-- {"id": "chatcmpl-...", "model": "gpt-4o-mini-2024-07-18", "usage": {"total_tokens": 27, "prompt_tokens": 18, "completion_tokens": 9}, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?", "refusal": null}, "logprobs": null, "finish_reason": "stop"}], "created": 1724765541, "system_fingerprint": "fp_..."}
```


#### `fireworks_set_api_key(text)` and `fireworks_get_api_key() -> text`

Store and retrieve your Fireworks.ai API key:

```sql
select rag.fireworks_set_api_key('fw_...');
select rag.fireworks_get_api_key();
-- 'fw_...'
```


#### `fireworks_chat_completion(json) -> json`

Call out to Fireworks.ai chat/completions API (makes network request):

```sql
select rag.fireworks_chat_completion('{"model":"accounts/fireworks/models/llama-v3p1-8b-instruct","messages":[{"role":"system","content":"you are a helpful assistant"},{"role":"user","content":"hi!"}]}');
--  {"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hi! How can I assist you today?","role":"assistant"}}],"created":1725362940,"id":"...","model":"accounts/fireworks/models/llama-v3p1-8b-instruct","object":"chat.completion","usage":{"completion_tokens":10,"prompt_tokens":23,"total_tokens":33}}
```


## End-to-end RAG example

Setup: create a `docs` table and ingest some PDF documents, then create and index an `embeddings` table. None of this setup uses the extension, only its `pgvector` dependency.

```sql
drop table docs cascade;
create table docs
( id int primary key generated always as identity
, blob bytea not null
);

\set contents `base64 < /path/to/first.pdf`
insert into docs (blob) values (decode(:'contents','base64'));

\set contents `base64 < /path/to/second.pdf`
insert into docs (blob) values (decode(:'contents','base64'));

\set contents `base64 < /path/to/third.pdf`
insert into docs (blob) values (decode(:'contents','base64'));

drop table embeddings;
create table embeddings
( id int primary key generated always as identity
, doc_id int not null references docs(id)
, chunk text not null
, embedding vector(384) not null
);

create index on embeddings using hnsw (embedding vector_cosine_ops);
```

Now we extract text from some PDFs, chunk that text, and generate embeddings for the chunks (this is all done locally).

```sql
with chunks as (
  select id, unnest(rag.chunks_by_token_count_bge_small_en_v15(rag.text_from_pdf(blob), 192, 8)) as chunk
  from docs
)
insert into embeddings (doc_id, chunk, embedding) (
  select id, chunk, rag_embed_bge_small_en_v15.embedding_for_passage(chunk) from chunks
);
```

Let's query the embeddings and rerank the results (still all done locally).

```sql
\set query 'what is [...]? how does it work?'

with naive_ordered as (
  select
    id, doc_id, chunk, embedding <=> rag_embed_bge_small_en_v15.embedding_for_query(:'query') as cosine_distance
  from embeddings
  order by cosine_distance
  limit 10
)
select *, rag_rerank_jina_v1_tiny_en.rerank_score(:'query', chunk) as rerank_distance
from naive_ordered
order by rerank_distance;
```

Building on that, now we can also feed the query and top chunks to remote ChatGPT to complete the RAG pipeline.

```sql
\set query 'what is [...]? how does it work?'

with naive_ordered as (
  select
    id, doc_id, chunk, embedding <=> rag_embed_bge_small_en_v15.embedding_for_query(:'query') as cosine_distance
  from embeddings
  order by cosine_distance
  limit 10
),
reranked as (
  select *, rag_rerank_jina_v1_tiny_en.rerank_score(:'query', chunk) as rerank_distance
  from naive_ordered
  order by rerank_distance limit 5
)
select rag.openai_chat_completion(json_object(
  'model': 'gpt-4o-mini',
  'messages': json_array(
    json_object(
      'role': 'system',
      'content': E'The user is [...].\n\nTry to answer the user''s QUESTION using only the provided CONTEXT.\n\nThe CONTEXT represents extracts from [...] which have been selected as most relevant to this question.\n\nIf the context is not relevant or complete enough to confidently answer the question, your best response is: "I''m afraid I don''t have the information to answer that question".'
    ),
    json_object(
      'role': 'user',
      'content': E'# CONTEXT\n\n```\n' || string_agg(chunk, E'\n\n') || E'\n```\n\n# QUESTION\n\n```\n' || :'query' || E'```'
    )
  )
)) -> 'choices' -> 0 -> 'message' -> 'content' as answer
from reranked;
```


## License

This software is released under the [Apache 2.0 license](LICENSE). Third-party code and data are provided under their respective licenses.

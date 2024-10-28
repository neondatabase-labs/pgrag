drop extension if exists rag_bge_small_en_v15;
create extension rag_bge_small_en_v15 cascade;

-- \df rag_bge_small_en_v15.*

-- rag_bge_small_en_v15 | chunks_by_token_count | text[]           | document text, max_tokens integer, max_overlap integer | func
select rag_bge_small_en_v15.chunks_by_token_count('the cat sat on the mat', 3, 2);

-- rag_bge_small_en_v15 | embedding_for_passage | vector           | input text                                             | func
select rag_bge_small_en_v15.embedding_for_passage('the cat sat on the mat');

-- rag_bge_small_en_v15 | embedding_for_query   | vector           | input text                                             | func
select rag_bge_small_en_v15.embedding_for_query('the cat sat on the mat');

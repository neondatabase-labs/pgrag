drop extension if exists rag_jina_reranker_v1_tiny_en;
create extension rag_jina_reranker_v1_tiny_en cascade;

-- \df rag_jina_reranker_v1_tiny_en.*

-- rag_jina_reranker_v1_tiny_en | rerank_distance | real             | query text, passage text    | func
select rag_jina_reranker_v1_tiny_en.rerank_distance('the cat sat on the mat', 'the baboon played with the balloon');
select rag_jina_reranker_v1_tiny_en.rerank_distance('the cat sat on the mat', 'the tanks fired at the buildings');

-- rag_jina_reranker_v1_tiny_en | rerank_distance | real[]           | query text, passages text[] | func
select rag_jina_reranker_v1_tiny_en.rerank_distance('the cat sat on the mat', ARRAY['the baboon played with the balloon', 'the tanks fired at the buildings']);

-- rag_jina_reranker_v1_tiny_en | rerank_score    | real             | query text, passage text    | func
select rag_jina_reranker_v1_tiny_en.rerank_score('the cat sat on the mat', 'the baboon played with the balloon');
select rag_jina_reranker_v1_tiny_en.rerank_score('the cat sat on the mat', 'the tanks fired at the buildings');

-- rag_jina_reranker_v1_tiny_en | rerank_score    | real[]           | query text, passages text[] | func
select rag_jina_reranker_v1_tiny_en.rerank_score('the cat sat on the mat', ARRAY['the baboon played with the balloon', 'the tanks fired at the buildings']);

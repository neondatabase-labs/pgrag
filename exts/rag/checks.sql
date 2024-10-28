drop extension if exists rag;
create extension rag cascade;

-- \df rag.*

-- rag    | anthropic_set_api_key                           | void             | api_key text                                                | func
select rag.anthropic_set_api_key('xyz');

-- rag    | anthropic_get_api_key                           | text             |                                                             | func
select rag.anthropic_get_api_key();

-- rag    | anthropic_messages                              | json             | version text, body json                                     | func
select rag.anthropic_messages('2023-06-01', '{
    "model": "claude-3-haiku-20240307",
    "max_tokens": 64,
    "system": "you are a helpful assistant",
    "messages":[
        {
            "role": "user",
            "content": "hi!"
        }
    ]
}'::json);

-- rag    | chunks_by_character_count                       | text[]           | document text, max_characters integer, max_overlap integer  | func
select rag.chunks_by_character_count('the cat sat on the mat', 10, 5);

-- rag    | fireworks_set_api_key                           | void             | api_key text                                                | func
select rag.fireworks_set_api_key('abc');

-- rag    | fireworks_get_api_key                           | text             |                                                             | func
select rag.fireworks_get_api_key();

-- rag    | fireworks_chat_completion                       | json             | body json                                                   | func
select rag.fireworks_chat_completion('{
    "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "messages":[
        {
            "role":"system",
            "content":"you are a helpful assistant"
        }, {
            "role": "user",
            "content": "hi!"
        }
    ]
}');

-- rag    | fireworks_nomic_embed_text_v1                   | vector           | input text                                                  | func
select rag.fireworks_nomic_embed_text_v1('the cat sat on the mat');
select vector_dims(rag.fireworks_nomic_embed_text_v1('the cat sat on the mat'));

-- rag    | fireworks_nomic_embed_text_v15                  | vector           | input text                                                  | func
select rag.fireworks_nomic_embed_text_v15('the cat sat on the mat');
select vector_dims(rag.fireworks_nomic_embed_text_v15('the cat sat on the mat'));

-- rag    | fireworks_text_embedding                        | vector           | model text, input text                                      | func
select rag.fireworks_text_embedding('thenlper/gte-base', 'the cat sat on the mat');

-- rag    | fireworks_text_embedding_thenlper_gte_base      | vector           | input text                                                  | func
select rag.fireworks_text_embedding_thenlper_gte_base('the cat sat on the mat');
select vector_dims(rag.fireworks_text_embedding_thenlper_gte_base('the cat sat on the mat'));

-- rag    | fireworks_text_embedding_thenlper_gte_large     | vector           | input text                                                  | func
select rag.fireworks_text_embedding_thenlper_gte_large('the cat sat on the mat');
select vector_dims(rag.fireworks_text_embedding_thenlper_gte_large('the cat sat on the mat'));

-- rag    | fireworks_text_embedding_whereisai_uae_large_v1 | vector           | input text                                                  | func
select rag.fireworks_text_embedding_whereisai_uae_large_v1('the cat sat on the mat');
select vector_dims(rag.fireworks_text_embedding_whereisai_uae_large_v1('the cat sat on the mat'));

-- rag    | markdown_from_html                              | text             | document text                                               | func
select rag.markdown_from_html('<p>Hello</p>');

-- rag    | openai_set_api_key                              | void             | api_key text                                                | func
select rag.openai_set_api_key('qwe');

-- rag    | openai_get_api_key                              | text             |                                                             | func
select rag.openai_get_api_key();

-- rag    | openai_chat_completion                          | json             | body json                                                   | func
select rag.openai_chat_completion('{
    "model": "gpt-4o-mini",
    "messages":[
        {
            "role": "system",
            "content": "you are a helpful assistant"
        }, {
            "role": "user",
            "content": "hi!"
        }
    ]
}');

-- rag    | openai_text_embedding                           | vector           | model text, input text                                      | func
select rag.openai_text_embedding('text-embedding-3-small', 'the cat sat on the mat');

-- rag    | openai_text_embedding_3_large                   | vector           | input text                                                  | func
select rag.openai_text_embedding_3_large('the cat sat on the mat');
select vector_dims(rag.openai_text_embedding_3_large('the cat sat on the mat'));

-- rag    | openai_text_embedding_3_small                   | vector           | input text                                                  | func
select rag.openai_text_embedding_3_small('the cat sat on the mat');
select vector_dims(rag.openai_text_embedding_3_small('the cat sat on the mat'));

-- rag    | openai_text_embedding_ada_002                   | vector           | input text                                                  | func
select rag.openai_text_embedding_ada_002('the cat sat on the mat');
select vector_dims(rag.openai_text_embedding_ada_002('the cat sat on the mat'));

-- rag    | text_from_docx                                  | text             | document bytea                                              | func
-- rag    | text_from_pdf                                   | text             | document bytea                                              | func

-- rag    | voyageai_set_api_key                            | void             | api_key text                                                | func
select rag.voyageai_set_api_key('uio');

-- rag    | voyageai_get_api_key                            | text             |                                                             | func
select rag.voyageai_get_api_key();

-- rag    | voyageai_embedding                              | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding('voyage-3-lite', 'query', 'the cat sat on the mat');
select rag.voyageai_embedding('voyage-3-lite', 'document', 'the cat sat on the mat');
select rag.voyageai_embedding('voyage-3-lite', NULL, 'the cat sat on the mat');

-- rag    | voyageai_embedding_3                            | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_3('query', 'the cat sat on the mat');
select vector_dims(rag.voyageai_embedding_3('query', 'the cat sat on the mat'));

-- rag    | voyageai_embedding_3_lite                       | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_3_lite('document', 'the cat sat on the mat');
select vector_dims(rag.voyageai_embedding_3_lite('document', 'the cat sat on the mat'));

-- rag    | voyageai_embedding_code_2                       | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_code_2('document', 'the_cat.sat_on("the mat")');
select vector_dims(rag.voyageai_embedding_code_2('document', 'the_cat.sat_on("the mat")'));

-- rag    | voyageai_embedding_finance_2                    | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_finance_2('document', 'the cat sat on the mat');
select vector_dims(rag.voyageai_embedding_finance_2('document', 'the cat sat on the mat'));

-- rag    | voyageai_embedding_law_2                        | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_law_2('document', 'the cat sat on the mat');
select vector_dims(rag.voyageai_embedding_law_2('document', 'the cat sat on the mat'));

-- rag    | voyageai_embedding_multilingual_2               | vector           | model text, input_type rag.voyage_ai_input_type, input text | func
select rag.voyageai_embedding_multilingual_2(NULL, 'the cat sat on the mat');
select vector_dims(rag.voyageai_embedding_multilingual_2(NULL, 'the cat sat on the mat'));

-- rag    | voyageai_rerank_distance                        | real             | model text, query text, document text                       | func
select rag.voyageai_rerank_distance('rerank-2-lite', 'the cat sat on the mat', 'the baboon played with the balloon');

-- rag    | voyageai_rerank_distance                        | real[]           | model text, query text, documents text[]                    | func
select rag.voyageai_rerank_distance('rerank-2-lite', 'the cat sat on the mat', ARRAY['the baboon played with the balloon', 'how much wood would a woodchuck chuck?']);

-- rag    | voyageai_rerank_score                           | real             | model text, query text, document text                       | func
select rag.voyageai_rerank_score('rerank-2-lite', 'the cat sat on the mat', 'the baboon played with the balloon');

-- rag    | voyageai_rerank_score                           | real[]           | model text, query text, documents text[]                    | func
select rag.voyageai_rerank_score('rerank-2-lite', 'the cat sat on the mat', ARRAY['the baboon played with the balloon', 'how much wood would a woodchuck chuck?']);

corpus_file=../index_musique_db/corpus.jsonl # jsonl
save_dir=../index_musique_db
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

echo "Starting FlashRAG server..."
python flashrag_server.py \
    --index_path $save_dir/${retriever_name}_Flat.index \
    --corpus_path $corpus_file \
    --retrieval_topk 15 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_model \
    --reranking_topk 10 \
    --reranker_model "cross-encoder/ms-marco-MiniLM-L12-v2" \
    --reranker_batch_size 64 \
    --host "0.0.0.0" \
    --port 2223
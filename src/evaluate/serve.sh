#jan-hq/Qwen3-4B-v0.4-deepresearch-no-think, Qwen/Qwen3-4B, jan-hq/Qwen3-4B-v0.3-deepresearch-100-step
export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve jan-hq/Qwen3-4B-v0.4-deepresearch-no-think \
    --host 127.0.0.1 --port 8080 \
    --data-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template ./qwen3_nonthinking.jinja
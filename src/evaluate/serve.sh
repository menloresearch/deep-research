#jan-hq/Qwen3-4B-v0.4-deepresearch-no-think
vllm serve Qwen/Qwen3-4B \
    --host 127.0.0.1 --port 8008 \
    --data-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template ./qwen3_nonthinking.jinja
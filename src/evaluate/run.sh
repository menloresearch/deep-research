# python vllm_mcp_agent.py --test --model_id openai:jan-hq/Qwen3-4B-v0.4-deepresearch-no-think --base_url http://localhost:8008/v1 --mcp_server_url "http://127.0.0.1:2323/mcp" --query "Does Putin have a secret wife and children. Do deep research and uncover information that is not public and write a thorough Journalism level report." --temperature 0.7 --model_kwargs '{"top_p": 0.8, "top_k": 20, "min_p": 0}'
MODEL_ID="Qwen/Qwen3-4B" #jan-hq/Qwen3-4B-v0.4-deepresearch-no-think
MODEL_ID_CLEAN=$(echo "$MODEL_ID" | sed 's/[\/:]/_/g')
export OPENROUTER_API_KEY="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"
# Run the main command
python vllm_mcp_agent.py \
  --csv input_simpleqa_432.csv \
  --model_id openai:$MODEL_ID \
  --base_url http://localhost:8008/v1 \
  --mcp_server_url "http://127.0.0.1:2323/mcp" \
  --temperature 0.7 \
  --model_kwargs '{"top_p": 0.8}' \
  --max_concurrent 8 \
  --output_csv "${MODEL_ID_CLEAN}_simple_qa_subset_output.csv"

python grade_answers_simpleqa.py "${MODEL_ID_CLEAN}_simple_qa_subset_output.csv" "graded_${MODEL_ID_CLEAN}_simple_qa_subset_output.csv" --workers 16

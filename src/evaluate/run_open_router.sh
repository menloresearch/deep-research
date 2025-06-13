MODEL_ID="deepseek/deepseek-chat-v3-0324" #jan-hq/Qwen3-4B-v0.4-deepresearch-no-think, Qwen/Qwen3-4B
MODEL_ID_CLEAN=$(echo "$MODEL_ID" | sed 's/[\/:]/_/g')
export OPENROUTER_API_KEY=""
# Run the main command
python vllm_mcp_agent.py \
  --csv input_simpleqa_432.csv \
  --model_id openai:$MODEL_ID \
  --base_url https://openrouter.ai/api/v1 \
  --mcp_server_url "http://127.0.0.1:2323/mcp" \
  --max_concurrent 32 \
  --output_csv "${MODEL_ID_CLEAN}_simple_qa_subset_output.csv"

python grade_answers_simpleqa.py "${MODEL_ID_CLEAN}_simple_qa_subset_output.csv" "graded_${MODEL_ID_CLEAN}_simple_qa_subset_output.csv" --workers 16
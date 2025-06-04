# Deep Research Eval

### 1. VLLM Model Server

- Edit the model name in `serve_deepresearch_model.sh`

```bash
# Start VLLM model server
tmux new-session -s dr-vllm
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
./serve_deepresearch_model.sh
```

### 2. Gradio Evaluation Interface

- Edit the model name in `01_qwen_eval_gradio_parallel_vllm....py`

```bash
# In another terminal
tmux new-session -s dr-eval
uv venv .venv_eval
source .venv_eval/bin/activate
uv pip install -r requirements_eval.txt

# Fix PDF processing
uv pip uninstall pdfminer pdfminer-six
uv pip install pdfminer-six

# Run evaluation interface
python 01_qwen_eval_gradio_parallel_vllm_local.py
```

## Included Benchmarks

- SimpleQA (10% - 432 questions)
- WebWalkerQA (680 questions)
    - export OPEN_AI_API_KEY=YOUR_API_KEY
    - export OPEN_AI_API_BASE_URL=YOUR_API_BASE_URL

## Run SimpleQA Bench

1. Upload the simpleqa input csv to the gradio interface
2. Click process and wait (about 1 hours). The output csv is updated in realtime in `batch_outputs` dir.
3. Download the output csv, put the csv inside of this repo, modify the path in `03_grade_answers_simpleqa.py`
4. Run `python 03_grade_answers_simpleqa.py`. The result will be printed to the console.
   1. The result file for each row will be saved to `graded_simpleqa_....`
   2. The log will be saved to `logs`

## Run WebWalkerQA Bench

1. Upload the webwalkerqa input csv to the gradio interface
2. Click process and wait (about 1.5 hours). The output csv is updated in realtime in `batch_outputs` dir.
3. Download the output csv, put the csv inside of this repo, modify the path in `03_grade_answers_webwalkerqa.py`
4. Run `python 03_grade_answers_webwalkerqa.py`. The result will be printed to the console and saved to `...report.json`

## Acknowledgements

- [Open Deep Research](https://huggingface.co/spaces/m-ric/open_Deep-Research) for the original codebase
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format and quantization
- [VLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [WebAgent](https://github.com/Alibaba-NLP/WebAgent) for the WebWalkerQA benchmark

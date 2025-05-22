# Deep Research Model Deployment

## Prerequisites

- Python 3.11+
- uv (universal package manager)
- tmux

## Running the Project

### 1. Start VLLM Model Server (MUST BE FIRST)

```bash
# Start VLLM model server tmux session
tmux new-session -t dr-vllm
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
./serve_deepresearch_model.sh
```

### 2. Run Gradio Demo (In Another Terminal)

```bash
# Start Gradio demo tmux session
tmux new-session -t dr-eval
uv venv .venv_eval
source .venv_eval/bin/activate
uv pip install -r requirements_eval.txt

uv pip uninstall pdfminer
uv pip uninstall pdfminer-six
uv pip install pdfminer-six

python 01_qwen_eval_gradio_parallel_vllm_local.py
```

## Notes

- Start the VLLM model server FIRST before running the Gradio demo
- Run each tmux session in a separate terminal
- The model server and Gradio demo run in separate tmux sessions
- Make sure all dependencies are correctly installed before running scripts

## Unzipping Results

To unzip results, you can use the following command format. Replace the example filename and target directory with your actual file and desired directory.

```bash
mkdir -p target_directory_name
unzip path/to/your_zip_file.zip -d target_directory_name
```

For example:

```bash
mkdir -p qwen3_14b_openrouter
unzip output/results_simpleqa_432_qwen3_14b_openrouter_250522.zip -d qwen3_14b_openrouter
```

## Grading Answers (`03_grade_answers.py`)

This script automates the grading of model-generated answers against gold standard answers provided in a CSV file.

### Logic

1. **Input**: Reads a CSV file (e.g., `output/combined_simpleqa_output_432.csv`) containing questions, gold answers, and predicted answers.
2. **Environment Variable**: Requires the `OPENROUTER_API_KEY` environment variable to be set for accessing the grading LLM.
3. **Answer Cleaning**: Preprocesses the predicted answers by removing common prefixes like "**Final answer:**".
4. **LLM-based Grading**: For each row:
    - Constructs a prompt using a predefined template (`GRADER_TEMPLATE`). This template includes the question, gold target, and the cleaned predicted answer. It also provides examples of "CORRECT", "INCORRECT", and "NOT_ATTEMPTED" grades.
    - Sends the prompt to an LLM (e.g., `openai/gpt-4o` via OpenRouter API).
    - The LLM is expected to return a single letter: "A" (CORRECT), "B" (INCORRECT), or "C" (NOT_ATTEMPTED).
5. **Error Handling**:
    - If the `OPENROUTER_API_KEY` is not set, it defaults to a random grade.
    - If the LLM returns an empty or unexpected response, or if there's an API error, it defaults to "C" (NOT_ATTEMPTED) or marks the row as an "ERROR".
    - If essential information (question or gold answer) is missing in a row, or if the predicted answer is empty, it's graded as "C" (NOT_ATTEMPTED).
6. **Parallel Processing**: Uses `concurrent.futures.ProcessPoolExecutor` to process multiple rows in parallel, speeding up the grading process.
7. **Output**:
    - Appends two new columns to each row: `grade_letter` (A, B, or C) and `grade_description` (CORRECT, INCORRECT, or NOT_ATTEMPTED).
    - Prints the graded results to the console.
    - Saves the graded results to a new CSV file (e.g., `output/graded_combined_simpleqa_output_432.csv`).

### Configuration

- The input folder, input filename, and output filename are configurable within the `main()` function of the script.
- The LLM model used for grading and the number of parallel workers can also be modified in the script.

## Acknowledgements

We would like to thank the authors of the open deep research code for the awesome work! <https://huggingface.co/spaces/m-ric/open_Deep-Research>.

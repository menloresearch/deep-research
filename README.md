# Deep Research Model Deployment

A comprehensive toolkit for deploying and converting deep research models with support for VLLM serving, GGUF conversion, and evaluation workflows.

## Project Structure

```
├── utils/                          # Utility scripts
│   ├── 00_original_app.py          # Original application
│   ├── 97_chat_gguf.py             # Chat with GGUF models
│   ├── 98_convert_gguf.py          # GGUF conversion script
│   ├── 99_upload.py                # HuggingFace upload utility
│   ├── setup_telemetry.py          # Telemetry setup
│   └── test_vllm_connection.py     # VLLM connection tester
├── 01_qwen_eval_gradio_*.py        # Gradio evaluation interfaces
├── 02_concat_csv.py                # CSV concatenation utility
├── 03_grade_answers.py             # Answer grading script
├── Makefile                        # Main workflow automation
└── serve_deepresearch_model.sh     # Model serving script
```

## Prerequisites

- Python 3.11+
- uv (universal package manager)
- tmux
- cmake (for building llama.cpp)
- git-lfs (for large model files)

## Quick Start

### 1. GGUF Model Conversion (Recommended)

Convert your model to efficient GGUF format with Q4_K_M quantization:

```bash
# Complete automated workflow
make gguf-all

# Individual steps (if needed)
make gguf-setup      # Install dependencies and clone llama.cpp
make gguf-build      # Build llama.cpp binaries
make gguf-download   # Download the source model
make gguf-convert    # Convert to GGUF with Q4_K_M quantization
make gguf-upload     # Upload to HuggingFace
make gguf-clean      # Clean up temporary files

# Get help
make gguf-help
```

**Model Configuration** (edit in Makefile):

- Source: `jan-hq/Qwen3-14B-v0.1-deepresearch-100-step`
- Target: `jan-hq/Qwen3-14B-v0.1-deepresearch-100-step-gguf`
- Output: `temp_model-q4_k_m.gguf` (~8.4GB for 14B model)

### 2. Chat with GGUF Model

```bash
# Install dependencies
uv add torch transformers

# Chat with your converted model
python utils/97_chat_gguf.py
```

### 3. VLLM Model Server (Alternative)

```bash
# Start VLLM model server
tmux new-session -s dr-vllm
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
./serve_deepresearch_model.sh
```

### 4. Gradio Evaluation Interface

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

## Utility Scripts

### GGUF Conversion (`utils/98_convert_gguf.py`)

- **Purpose**: Convert HuggingFace models to GGUF format with quantization
- **Default**: Q4_K_M quantization (71% size reduction)
- **Auto-naming**: `<model_name>-<quantization>.gguf`
- **Error handling**: Size validation and fallback mechanisms

```bash
# Use defaults (temp_model → gguf_output)
python utils/98_convert_gguf.py

# Custom paths
python utils/98_convert_gguf.py model_dir output_dir Q4_K_M
```

### Upload to HuggingFace (`utils/99_upload.py`)

- **Purpose**: Upload directories to HuggingFace Hub
- **Features**: Automatic repo creation, ignore patterns

```bash
python utils/99_upload.py --local-dir gguf_output --repo-id username/model-name [--public]
```

### Chat Interface (`utils/97_chat_gguf.py`)

- **Purpose**: Interactive chat with GGUF models
- **Features**: Auto model loading, conversation loop
- **Config**: Edit model ID and filename in script

## Environment Variables

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key  # For grading
```

## Workflows

### GGUF Conversion Workflow

1. **Setup**: Install dependencies, clone llama.cpp
2. **Build**: Compile llama-quantize binary using CMake
3. **Download**: Download source model from HuggingFace
4. **Convert**: Two-step conversion (HF → f16 GGUF → Q4_K_M)
5. **Upload**: Push to HuggingFace with auto-generated README
6. **Clean**: Remove temporary files

### Quantization Options

- `Q2_K`: Smallest (lowest quality)
- `Q3_K_S/M/L`: Small to medium
- `Q4_0`, `Q4_K_S`, **`Q4_K_M`**: Recommended balance
- `Q5_0`, `Q5_K_S/M`: Higher quality
- `Q6_K`, `Q8_0`: Highest quality (larger size)

## Data Processing

### CSV Operations (`02_concat_csv.py`)

Concatenate multiple CSV files from evaluation runs.

### Answer Grading (`03_grade_answers.py`)

- **Purpose**: LLM-based grading of model answers
- **Features**: Parallel processing, error handling
- **Grades**: A (CORRECT), B (INCORRECT), C (NOT_ATTEMPTED)
- **Requirements**: OpenRouter API key

## Troubleshooting

### GGUF Conversion Issues

1. **Build errors**: Ensure cmake is installed
2. **Large file sizes**: Check quantization worked (should be ~8GB for 14B Q4_K_M)
3. **Permission errors**: Check HF_TOKEN is set
4. **Memory issues**: Ensure sufficient disk space (60GB+ for 14B models)

### VLLM Server Issues

1. **Connection errors**: Use `python utils/test_vllm_connection.py`
2. **GPU memory**: Check CUDA availability and memory
3. **Port conflicts**: Change port in serving script

## File Size Guide

**14B Model Sizes**:

- Original (BF16): ~28GB
- F16 GGUF: ~27GB
- Q4_K_M GGUF: ~8.4GB (recommended)
- Q2_K GGUF: ~4.2GB (lowest quality)

## Makefile Targets

```bash
make gguf-all      # Complete workflow
make gguf-setup    # Dependencies only
make gguf-build    # Build binaries only
make gguf-download # Download only
make gguf-convert  # Convert only
make gguf-upload   # Upload only
make gguf-clean    # Clean up
make gguf-help     # Show help
```

## Results Processing

### Unzipping Results

```bash
mkdir -p target_directory
unzip output/results_file.zip -d target_directory
```

Example:

```bash
mkdir -p qwen3_14b_results
unzip output/results_simpleqa_432_qwen3_14b_openrouter_250522.zip -d qwen3_14b_results
```

## Performance Notes

- **GGUF models**: 4x faster loading, 70% smaller files
- **Q4_K_M**: Best quality/size balance for most use cases
- **Parallel processing**: Evaluation and grading use multiprocessing
- **Memory efficiency**: VLLM optimized for inference

## Acknowledgements

- [Open Deep Research](https://huggingface.co/spaces/m-ric/open_Deep-Research) for the original codebase
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format and quantization
- [VLLM](https://github.com/vllm-project/vllm) for efficient model serving

# 🔬 DeepResearcher

<div align="center">

### *Next-Generation Research Intelligence*

**Unleashing the power of advanced AI for comprehensive analysis, multi-dimensional reasoning, and breakthrough insights**

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Preview-orange?style=for-the-badge&logo=flask" alt="Status"/>
  <img src="https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge&logo=robot" alt="AI Powered"/>
  <img src="https://img.shields.io/badge/Research-Agent-purple?style=for-the-badge&logo=search" alt="Research Agent"/>
</p>

---

<p align="center">
  <a href='#'><img src='https://img.shields.io/badge/📊-Project%20Overview-2E86AB?style=flat-square&logoColor=white'/></a>
  <a href='#'><img src='https://img.shields.io/badge/🚀-Live%20Demo-FF6B6B?style=flat-square&logoColor=white'/></a>
  <a href='#'><img src='https://img.shields.io/badge/📄-Research%20Paper-4ECDC4?style=flat-square&logoColor=white'/></a>
  <a href='#'><img src='https://img.shields.io/badge/🤗-HuggingFace%20Models-FFE66D?style=flat-square&logoColor=black'/></a>
  <a href='https://huggingface.co/datasets/jan-hq/Musique-subset'><img src='https://img.shields.io/badge/📚-Dataset-95E1D3?style=flat-square&logoColor=black'/></a>
  <a href='#'><img src='https://colab.research.google.com/assets/colab-badge.svg'/></a>
</p>

</div>

---

## 🌟 **What is DeepResearcher?**

DeepResearcher represents a paradigm shift in AI-driven research methodology. Developed by **Menlo Research**, this cutting-edge research agent transcends traditional information retrieval by employing sophisticated multi-hop reasoning, contextual synthesis, and adaptive learning mechanisms.

Our model doesn't just find answers—it *understands*, *connects*, and *discovers*. 

---

*Empowering researchers, analysts, and innovators to unlock deeper understanding through intelligent automation.*


## ⚙️ Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/menloresearch/deep-research.git
    cd deep-research
    ```

2.  **Set up Python Environment (using `uv`):**
    If you use `pyenv`, it will automatically pick up the version from `.python-version`.
    ```bash
    # Create a virtual environment
    uv venv
    # Activate the virtual environment
    source .venv/bin/activate
    # Install dependencies
    uv pip install -r requirements.txt # Or `uv pip install` if pyproject.toml is primary
    ```
    *Note: This installs dependencies specified in `pyproject.toml` and locked in `uv.lock`.*

3.  **Configure Environment Variables:**
    Copy the example environment file and fill in your API keys:
    ```bash
    cp .env.example .env
    nano .env  # Or your preferred editor
    ```
    You'll need to provide keys for services like `TAVILY_API_KEY`, `HF_TOKEN`, etc., as used by the tools.


## 💡 Running the Demo

You can experience the Deep Research model in action in two primary ways:

### 1. Local Demo with `demo_app.py` (Recommended for Development)

This sets up a Gradio interface to interact with the agent.

*   **Step A: Start the VLLM Model Server (Critical!)**
    The demo application (`src/demo_app.py`) expects a VLLM-compatible API server running locally. We use a fine-tuned Qwen3-14B model.
    Open a new terminal and run:
    ```bash
    # (Activate your virtual environment if not already done: source .venv/bin/activate)
    vllm serve jan-hq/Qwen3-14B-v0.2-deepresearch-no-think-100-step \
        --port 8000 \
        --tensor-parallel-size 1 \
        --max-model-len 8192 # Adjust as per your model's needs
        # Add --enable-prefix-caching or other VLLM optimizations if desired
    ```
    *Note: Ensure you have VLLM installed (`uv pip install vllm`) and sufficient GPU memory.*
    *The model `jan-hq/Qwen3-14B-v0.2-deepresearch-no-think-100-step` is specifically configured in `demo_app.py`. If you change it, update the script.*

*   **Step B: Run the Gradio Demo Application**
    In another terminal (with the virtual environment activated):
    ```bash
    python src/demo_app.py
    ```
    This will launch a Gradio web interface, typically at `http://127.0.0.1:7860`. Open this URL in your browser to interact with the agent.

### 2. Via Jan App (Production Model)

Our production-ready Deep Research model is integrated into the **Jan App**.
*   [Download and install Jan App](https://jan.ai/).
*   

## 🏋️ Training
### 📚 Data Preparation & RAG System

The Deep Research model relies on a robust Retrieval Augmented Generation (RAG) system. We primarily use the Musique dataset.

*   **Prepare Data & Build Index:**
    The `Makefile` provides a convenient way to download the Musique dataset, preprocess it, and build a FAISS index:
    ```bash
    make data
    ```
    This command will:
    1.  Download the Musique dataset (raw files into `./data/raw/`).
    2.  Process it into a `corpus.jsonl` file (`./data/processed/corpus.jsonl`).
    3.  Build a FAISS index using `intfloat/e5-base-v2` embeddings (index stored in `./index_musique_db/`).

*   **Run the RAG Server:**
    Once the data is prepared and the index is built, you can start the RAG server (which `demo_app.py` and evaluation scripts might query if configured to do so, or if they implement their own RAG client):
    ```bash
    bash src/rag_setup/rag_server.sh
    ```
    This server uses `FlashRAG` components and serves retrieved documents based on semantic similarity.


## 📊 Evaluation

To evaluate the model's performance on benchmarks like SimpleQA and WebWalkerQA, please refer to the detailed instructions in:
➡️ **[`src/evaluate/README.md`](src/evaluate/README.md)**

This guide covers:
*   Setting up a VLLM server specifically for evaluation.
*   Running the Gradio-based evaluation interface (`src/evaluate/eval_app.py`).
*   Executing benchmark-specific grading scripts.


## 🙏 Acknowledgements

This project builds upon the great work of the open-source community. We'd like to thank:
*  [Verifier](https://github.com/willccbb/verifiers) for their foundational training code, which significantly inspired our work in `verifiers-deepresearch/`.
* [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for their insightful RAG (Retrieval Augmented Generation) methodologies
*   The [Hugging Face](https://huggingface.co/) team for `smol-agents`.
*   The [Autogen team (Microsoft)](https://github.com/microsoft/autogen) for utility scripts like `mdconvert`.
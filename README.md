# Deep Research

## Prerequisites

- Python >=3.11
- `uv` (Python package installer)

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/menloresearch/deep-research.git
    cd deep-research
    ```

2. **Create a virtual environment and install dependencies using uv:**

    ```bash
    uv venv
    uv pip install
    ```

    *Note: This command installs dependencies specified in `pyproject.toml`.*

3. **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

## Project Structure

```
📦 .
├── 📂 assets                    # Project assets (e.g. images, logos)
├── 📂 data                      # Data files
├── 📂 docs                      # Project documentation
│   └── 📜 git-workflow.md       # Guide for Git collaboration
├── 📂 logs                      # Log files
├── 📂 notebooks                 # Jupyter notebooks for experimentation
├── 📂 scripts                   # Utility scripts
├── 📂 src                       # Source code
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── embeddings.py
│   ├── knowledge_base.py
│   ├── prompts.py
│   ├── rewards.py
│   ├── search_module.py
│   ├── tools.py
│   └── utils.py
├── 📂 tests                     # Test suites
├── 📂 third_party               # Third-party libraries or code
├── 📜 .gitignore                # Files and directories to be ignored by Git
├── 📜 .python-version           # Specifies Python version for pyenv
├── 📜 Makefile                  # Makefile for common development tasks
├── 📜 README.md                 # This file
├── 📜 pyproject.toml            # Project metadata and dependencies
└── 📜 uv.lock                   # Lock file for uv package manager
```

## Development

- Follow the Git workflow outlined in [docs/git-workflow.md](docs/git-workflow.md).
- Use the `Makefile` for common tasks. For example:
    - To prepare data: `make data`
    - To run tests: `make test`
    - To lint your code: `make lint`

## Simple Retrieval Server

A lightweight FastAPI-based server for performing semantic search using the pre-built FAISS index from this project. This server is located in `src/search/`.

### Prerequisites

Ensure you have the necessary dependencies installed. If you are using `uv`, you might need to add FastAPI, Uvicorn, and Pydantic:

```bash
uv pip install fastapi "pydantic[email]" uvicorn
# Ensure other dependencies from knowledge_base.py (langchain, faiss, etc.) are also installed.
```

### Running the Server

You can run the server using the Makefile target:

```bash
make run-simple-retrieval-server
```

For development with auto-reload:

```bash
make run-simple-retrieval-server-dev
```

The server will start on `http://localhost:8001` by default (or the port specified by the `SIMPLE_RETRIEVAL_PORT` environment variable if set, though `Makefile` commands explicitly set the port). API documentation (Swagger UI) will be available at `http://localhost:<port>/docs`.

### Testing the Server

A client script is provided at `src/search/retrieval_request.py` to test the `/retrieve` endpoint.

**Usage:**

```bash
python src/search/retrieval_request.py "your search query here" -k 5 --host localhost --port 8001
```

**Parameters:**

- `query`: (Required) The search query string.
- `-k`: (Optional) Number of results to retrieve. Defaults to the server's configured `RAG_SEARCH_RESULTS_COUNT`.
- `--host`: (Optional) Server host. Defaults to `localhost`.
- `--port`: (Optional) Server port. Defaults to `8001`.

Example:

```bash
python src/search/retrieval_request.py "what are transformers in NLP?" -k 3
```

You can also use `curl`:

```bash
curl -X POST "http://localhost:8001/retrieve" \
-H "Content-Type: application/json" \
-d '{
  "query": "your search query here",
  "k": 5
}'
```

---

*This is a simplified README. Please adapt it to your project's specifics, especially regarding the `Makefile` commands.*

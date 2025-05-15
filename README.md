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

- Install verl

    ```bash
    git clone https://github.com/volcengine/verl
    cd verl
    uv pip install --no-deps -e ."[sglang]"
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
├── 📂 deploy                    # Deployment scripts and configurations
│   └── 📂 docker                # Dockerfiles
│   └── 📂 serving               # Serving scripts
├── 📂 docs                      # Project documentation
│   └── 📜 git-workflow.md       # Guide for Git collaboration
├── 📂 logs                      # Log files
├── 📂 notebooks                 # Jupyter notebooks for experimentation
├── 📂 scripts                   # Utility scripts (excluding serving)
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
- After deploying, ensure that all necessary services (e.g., retrieval server, sandbox environment) are running. You can typically manage these using `docker compose` commands found in the Makefile (e.g., `make docker-up`).

## Running Services with Docker Compose

The project uses Docker Compose to manage and run its services, including the retrieval server and any other backend components (e.g., sandbox environment).

To start all services defined in the `docker-compose.yml` file (implicitly used by the `Makefile` targets), use:

```bash
make docker-up
```

This command will build the Docker images if necessary and start all services in detached mode.

Once the services are running:

- The **Simple Retrieval Server** API documentation (Swagger UI) can typically be accessed at `http://localhost:8002/docs`. The exact port mapping can be confirmed in your `docker-compose.yml` or by checking `make docker-ps`.
- Other services (like `mock_retriever`, `flashrag_retriever`, `sandbox_env`) will be available as per their configurations in the `docker-compose.yml` and their respective Dockerfiles in `deploy/docker/`.

To view logs for all services:

```bash
make docker-logs
```

To stop all services:

```bash
make docker-down
```

### Service Components

The server components (like FastAPI, Uvicorn, etc., and their Python dependencies) are defined and installed within their respective Docker images (see `deploy/docker/`). You do not need to install these Python packages manually on your host if you are running the services via Docker Compose.

## Simple Retrieval Server

A lightweight FastAPI-based server for performing semantic search. This server is located in `deploy/serving/`.

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

The server will start on `http://localhost:8002` by default. API documentation (Swagger UI) will be available at `http://localhost:8002/docs`.

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

## Acknowledgements

This project is under active development. During this phase, we may reference or adapt code from various excellent open-source repositories. We are committed to giving proper credit and will consolidate all attributions as the project matures.

We would like to acknowledge the [ReCall project](https://github.com/Agent-RL/ReCall) and its authors, as their work in learning to reason with tool calls for LLMs via reinforcement learning has been an inspiration and a valuable resource.

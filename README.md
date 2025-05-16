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
    source .venv/bin/activate
    uv pip install -e .
    ```

- Install flash-attn

    ```bash
    uv pip install flash-attn --no-build-isolation
    ```

3. **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

## Running Services with Makefile

- Open new terminal or tmux session
- Run `make run-all-services-nodocker` to start all services
- Run `make kill-all-services-nodocker` to stop all services

## Run Training

```bash
bash train.sh
```

## Running Services with Docker Compose (Deprecated)

The project uses Docker Compose to manage and run its services, including the retrieval server and any other backend components (e.g., sandbox environment).

To start all services defined in the `docker-compose.yml` file (implicitly used by the `Makefile` targets), use:

```bash
make docker-up
```

## Project Structure

```
📦 .
├── 📂 assets                    # Project assets (e.g. images, logos)
├── 📂 data                      # Data files
├── 📂 deploy                    # Deployment scripts and configurations
│   ├── 📂 docker                # Dockerfiles
│   ├── 📂 serving               # Serving scripts
│   └── 📜 requirements.txt      # Deployment specific requirements
├── 📂 docs                      # Project documentation
│   └── 📜 git-workflow.md       # Guide for Git collaboration
├── 📂 logs                      # Log files
├── 📂 notebooks                 # Jupyter notebooks for experimentation
├── 📂 scripts                   # Utility scripts (excluding serving)
│   ├── 📂 demo
│   ├── 📂 download
│   ├── 📂 prepare_train_data
│   ├── 📂 train
│   ├── 📂 upload
│   ├── 📜 .gitkeep
│   ├── 📜 __init__.py
│   ├── 📜 README.md
│   └── 📜 train_agent.py
├── 📂 src                       # Source code
│   ├── 📂 deep_research.egg-info # Packaging info
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
├── 📜 train.sh                  # Main training script 
└── 📜 uv.lock                   # Lock file for uv package manager
```

## Development

- Follow the Git workflow outlined in [docs/git-workflow.md](docs/git-workflow.md).
- Use the `Makefile` for common tasks. For example:
    - To prepare data: `make data`
    - To run tests: `make test`
    - To lint your code: `make lint`

## Acknowledgements

This project is under active development. During this phase, we may reference or adapt code from various excellent open-source repositories. We are committed to giving proper credit and will consolidate all attributions as the project matures.

We would like to acknowledge the [ReCall project](https://github.com/Agent-RL/ReCall) and its authors, as their work in learning to reason with tool calls for LLMs via reinforcement learning has been an inspiration and a valuable resource.

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

## Project Resources

| Resource | URL |
|----------|-----|
| Data Repository | [janhq/demo-deep-research-data](https://huggingface.co/datasets/janhq/demo-deep-research-data) |
| Model Repository | [janhq/demo-deep-research-model](https://huggingface.co/janhq/demo-deep-research-model) |
| Wandb | [menlo-research/deep-research](https://wandb.ai/menlo-research/deep-research) |

## Data Management

### Download Data from HuggingFace

To download data from HuggingFace, use the following command:

```bash
make download-data
```

This will download the dataset from `janhq/demo-deep-research-data`.

To download the model:

```bash
make download-model
```

This will download the model from `janhq/demo-deep-research-model` (including wandb and tensorboard_logs if available).

### Prepare All Datasets (One time run)

To download and prepare all datasets (Musique and ReCall), run:

```bash
make data
```

This will:

1. Download the Musique dataset
2. Process and prepare the Musique dataset
3. Convert Musique data to ReCall format
4. Download the ReCall syntool dataset
5. Create sampled subsets (1000 train and 50 test examples) of syntool_re_call data

### Prepare Specific Datasets

For more granular control, you can use these commands:

```bash
# Download and prepare only Musique data
make prepare-data

# Download and sample only ReCall syntool data
make download-recall-syntool
```

### Upload Data to HuggingFace

To upload data to HuggingFace, use the following command:

```bash
make upload-data
```

This will upload the `data` directory to `janhq/demo-deep-research-data`.

To upload a model:

```bash
make upload-model
```

This will upload the `data` directory and model-specific directories (wandb, tensorboard_logs) to `janhq/demo-deep-research-model`.

To upload model checkpoints:

```bash
make upload-checkpoint
```

This will upload the `checkpoints` directory to `janhq/demo-deep-research-model`.

To download model checkpoints:

```bash
make download-checkpoint
```

This will download checkpoint files from `janhq/demo-deep-research-model` to the `checkpoints` directory.

Both upload commands require setting the HuggingFace token:

```bash
export HF_TOKEN="your-token"
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

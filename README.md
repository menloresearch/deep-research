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

---

*This is a simplified README. Please adapt it to your project's specifics, especially regarding the `Makefile` commands.*

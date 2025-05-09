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
├── 📂 docs                      # Project documentation
│   └── 📜 git-workflow.md       # Guide for Git collaboration
├── 📂 notebooks                 # Jupyter notebooks for experimentation
├── 📂 scripts                   # Utility scripts
├── 📂 src                       # Source code
│   └── 📂 deep_research         # Main application package (example)
│       ├── __init__.py
│       └── ...                  # Other modules and sub-packages
├── 📂 tests                     # Test suites
├── 📂 third_party               # Third-party libraries or code
├── 📜 .gitignore                # Files and directories to be ignored by Git
├── 📜 .python-version           # Specifies Python version for pyenv
├── 📜 Makefile                  # Makefile for common development tasks
├── 📜 README.md                 # This file
├── 📜 main.py                   # Main entry point for the application (example)
├── 📜 pyproject.toml            # Project metadata and dependencies for Poetry/PEP 621 build systems
└── 📜 uv.lock                   # Lock file for uv package manager
```

## Development

- Follow the Git workflow outlined in [docs/git-workflow.md](docs/git-workflow.md).
- Use the `Makefile` for common tasks (e.g., `make lint`, `make test`). *Please update `Makefile` usage instructions as per its actual commands.*

---

*This is a simplified README. Please adapt it to your project's specifics, especially regarding the `Makefile` commands.*

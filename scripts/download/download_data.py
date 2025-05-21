import os

import dotenv
from huggingface_hub import hf_hub_download, snapshot_download

# Load environment variables from .env file with override=True
dotenv.load_dotenv(override=True)

# Hard-coded repositories
MODEL_REPO_ID = "janhq/demo-deep-research-model"
DATA_REPO_ID = "janhq/demo-deep-research-data"


def download_model(output_dir: str = "data", token: str | None = None) -> None:
    """Download model from HuggingFace."""
    _download_from_hf(
        MODEL_REPO_ID, output_dir, "model", token, include_model_dirs=True
    )


def download_data(output_dir: str = "data", token: str | None = None) -> None:
    """Download dataset from HuggingFace."""
    _download_from_hf(DATA_REPO_ID, output_dir, "dataset", token)


def _download_from_hf(
    repo_id: str,
    output_dir: str = "data",
    repo_type: str = "dataset",
    token: str | None = None,
    single_file: str | None = None,
    include_model_dirs: bool = False,
) -> None:
    """Download data from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Directory to save downloaded data
        repo_type: Type of repository (dataset or model)
        token: HuggingFace token (or uses HF_TOKEN env var)
        single_file: Optional single file to download
        include_model_dirs: Whether to include wandb and tensorboard_logs
    """
    # Input validation
    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable not set")

    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process
    if single_file:
        # Download single file
        print(f"Downloading {single_file} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=single_file,
            repo_type=repo_type,
            token=token,
            local_dir=output_dir,
        )
        print(f"File downloaded to {output_dir}")
    else:
        # Download entire repo
        print(f"Downloading repository {repo_id}...")
        snapshot_download(
            repo_id=repo_id, repo_type=repo_type, token=token, local_dir=output_dir
        )
        print(f"Repository downloaded to {output_dir}")

        # Download model-specific directories if requested
        if include_model_dirs:
            model_dirs = ["wandb", "tensorboard_logs"]
            for dir_name in model_dirs:
                try:
                    print(f"Downloading {dir_name} from {repo_id}...")
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        local_dir=dir_name,
                        allow_patterns=f"{dir_name}/**",
                    )
                    print(f"{dir_name} downloaded successfully")
                except Exception as e:
                    print(f"Warning: Could not download {dir_name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["model", "data"],
        default="data",
        help="Type of data to download (model or data)",
    )
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for downloaded data"
    )
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument(
        "--single-file", help="Download single file instead of entire repo"
    )

    args = parser.parse_args()

    if args.type == "model":
        download_model(args.output_dir, args.token)
        print(f"Model downloaded successfully from {MODEL_REPO_ID}")
    else:
        download_data(args.output_dir, args.token)
        print(f"Data downloaded successfully from {DATA_REPO_ID}")

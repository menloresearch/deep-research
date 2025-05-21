import os

import dotenv
from huggingface_hub import HfApi

# Load environment variables from .env file with override=True
dotenv.load_dotenv(override=True)

# Hard-coded repositories
MODEL_REPO_ID = "janhq/demo-deep-research-model"
DATA_REPO_ID = "janhq/demo-deep-research-data"


def upload_model(model_dir: str = "data", token: str | None = None) -> None:
    """Upload model data to HuggingFace."""
    _upload_to_hf(MODEL_REPO_ID, model_dir, "model", token, include_model_dirs=True)


def upload_data(data_dir: str = "data", token: str | None = None) -> None:
    """Upload dataset to HuggingFace."""
    _upload_to_hf(DATA_REPO_ID, data_dir, "dataset", token)


def _upload_to_hf(
    repo_id: str,
    data_dir: str = "data",
    repo_type: str = "dataset",
    token: str | None = None,
    include_model_dirs: bool = False,
) -> None:
    """Upload data to HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        data_dir: Directory to upload
        repo_type: Type of repository (dataset or model)
        token: HuggingFace token (or uses HF_TOKEN env var)
        include_model_dirs: Whether to include wandb and tensorboard_logs
    """
    # Input validation
    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable not set")

    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")

    # Process
    api = HfApi(token=token)

    # Upload main data directory
    print(f"Uploading {data_dir} to {repo_id}...")
    api.upload_folder(folder_path=data_dir, repo_id=repo_id, repo_type=repo_type)

    # Upload model-specific directories if requested
    if include_model_dirs:
        model_dirs = ["wandb", "tensorboard_logs"]
        for dir_name in model_dirs:
            if os.path.exists(dir_name):
                print(f"Uploading {dir_name} to {repo_id}...")
                api.upload_folder(
                    folder_path=dir_name, repo_id=repo_id, repo_type=repo_type
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["model", "data"],
        default="data",
        help="Type of data to upload (model or data)",
    )
    parser.add_argument("--dir", default="data", help="Directory to upload")
    parser.add_argument("--token", help="HuggingFace token")

    args = parser.parse_args()

    if args.type == "model":
        upload_model(args.dir, args.token)
        print(f"Model uploaded successfully to {MODEL_REPO_ID}")
    else:
        upload_data(args.dir, args.token)
        print(f"Data uploaded successfully to {DATA_REPO_ID}")

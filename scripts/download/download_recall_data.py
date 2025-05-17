from huggingface_hub import snapshot_download
import os

def download_dataset(repo_id: str, local_dir: str, repo_type: str = "dataset") -> None:
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Created directory: {local_dir}")

    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Dataset {repo_id} downloaded successfully to {local_dir}")

if __name__ == "__main__":
    dataset_repo_id = "agentrl/ReCall-data"
    output_directory = "data/ReCall-data"
    
    base_output_dir = os.path.dirname(output_directory)
    if base_output_dir and not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created base directory: {base_output_dir}")

    download_dataset(repo_id=dataset_repo_id, local_dir=output_directory)

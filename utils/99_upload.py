import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory to upload")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--public", action="store_true", help="Make repository public (default: private)")
    args = parser.parse_args()
    
    load_dotenv(override=True)
    hf_token = os.getenv("HF_TOKEN")
    
    ignore_patterns = [
        "*.log", "*.pyc", ".git*", "*.bin", "*.pt", 
        "*.ckpt", "events.*", "wandb/*", "runs/*"
    ]
    
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.repo_id, private=not args.public, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model"
    )
    print(f"✅ Done: {args.local_dir} -> {args.repo_id}")

if __name__ == "__main__":
    main()

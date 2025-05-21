import argparse
import json
import os
import random

from datasets import load_dataset  # type: ignore


def sample_data(
    input_dir: str,
    output_dir: str,
    train_samples: int = 1000,
    test_samples: int = 50,
    seed: int = 42,
) -> None:
    """
    Randomly sample from the syntool_re_call dataset.

    Args:
        input_dir: Directory containing the original parquet files
        output_dir: Directory to save the sampled data
        train_samples: Number of samples for train set
        test_samples: Number of samples for test set
        seed: Random seed
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Sample train data
    train_path = os.path.join(input_dir, "train.parquet")
    if os.path.exists(train_path):
        try:
            # Load the dataset using datasets library
            train_dataset = load_dataset("parquet", data_files={"train": train_path})  # type: ignore
            train_data = train_dataset["train"]  # type: ignore

            # Get total number of samples
            total_train = len(train_data)
            print(f"Found {total_train} training samples")

            # Sample rows
            if train_samples < total_train:
                indices = random.sample(range(total_train), train_samples)
                sampled_train = train_data.select(indices)  # type: ignore
                print(f"Sampled {train_samples} training examples")
            else:
                sampled_train = train_data
                print(
                    f"Using all {total_train} training examples (requested {train_samples})"
                )

            # Save to output
            output_train_path = os.path.join(output_dir, "train.parquet")
            sampled_train.to_parquet(output_train_path)  # type: ignore

            # Save as JSON for inspection
            json_path = os.path.join(output_dir, "train.parquet.as.json")
            with open(json_path, "w") as f:
                # Convert each example to a JSON string and write line by line
                for example in sampled_train:  # type: ignore
                    f.write(json.dumps(example) + "\n")

            print(f"Saved sampled training data to {output_dir}")
        except Exception as e:
            print(f"Error processing train data: {e}")
    else:
        print(f"Warning: {train_path} not found")

    # Sample test data
    test_path = os.path.join(input_dir, "test.parquet")
    if os.path.exists(test_path):
        try:
            # Load the dataset using datasets library
            test_dataset = load_dataset("parquet", data_files={"train": test_path})  # type: ignore
            test_data = test_dataset["train"]  # type: ignore

            # Get total number of samples
            total_test = len(test_data)
            print(f"Found {total_test} test samples")

            # Sample rows
            if test_samples < total_test:
                indices = random.sample(range(total_test), test_samples)
                sampled_test = test_data.select(indices)  # type: ignore
                print(f"Sampled {test_samples} test examples")
            else:
                sampled_test = test_data
                print(
                    f"Using all {total_test} test examples (requested {test_samples})"
                )

            # Save to output
            output_test_path = os.path.join(output_dir, "test.parquet")
            sampled_test.to_parquet(output_test_path)  # type: ignore

            # Save as JSON for inspection
            json_path = os.path.join(output_dir, "test.parquet.as.json")
            with open(json_path, "w") as f:
                # Convert each example to a JSON string and write line by line
                for example in sampled_test:  # type: ignore
                    f.write(json.dumps(example) + "\n")

            print(f"Saved sampled test data to {output_dir}")
        except Exception as e:
            print(f"Error processing test data: {e}")
    else:
        print(f"Warning: {test_path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory with original data"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save sampled data"
    )
    parser.add_argument(
        "--train-samples", type=int, default=1000, help="Number of train samples"
    )
    parser.add_argument(
        "--test-samples", type=int, default=50, help="Number of test samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    sample_data(
        args.input_dir,
        args.output_dir,
        args.train_samples,
        args.test_samples,
        args.seed,
    )

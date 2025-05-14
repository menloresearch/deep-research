import argparse
import json

import requests

# Default host and port match the server's defaults
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8001


def main():
    parser = argparse.ArgumentParser(description="Send a search request to the retrieval server.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument("-k", type=int, help="Number of results to return.")
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help=f"Host of the retrieval server (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port of the retrieval server (default: {DEFAULT_PORT})"
    )

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/retrieve"
    payload = {"query": args.query}
    if args.k is not None:
        payload["k"] = args.k

    print(f"Sending request to {url} with payload: {json.dumps(payload)}")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to the server or during request: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response. Raw response text:")
        print(response.text)


if __name__ == "__main__":
    main()

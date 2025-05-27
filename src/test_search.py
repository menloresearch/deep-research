import json
from typing import Any, Dict

import requests


def search_with_urls(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Search for documents and return results with their corpus IDs"""
    server_url = "http://localhost:2223"
    payload = {
        "queries": [query],
        "topk_retrieval": max(num_results * 3, 10),
        "topk_rerank": num_results,
        "return_scores": True,  # Get scores to see ranking
    }

    response = requests.post(
        f"{server_url}/retrieve", json=payload, headers={"Content-Type": "application/json"}, timeout=600
    )
    return response.json()


def visit_site(doc_id: str, doc_title: str) -> Dict[str, Any]:
    """Visit a specific document by its corpus ID and title"""
    server_url = "http://localhost:2223"
    # Search by exact title since it's unique in the corpus
    payload = {
        "queries": [f'"{doc_title}"'],  # Exact title match in quotes
        "topk_retrieval": 1,
        "topk_rerank": 1,
        "return_scores": False,
    }

    response = requests.post(
        f"{server_url}/retrieve", json=payload, headers={"Content-Type": "application/json"}, timeout=600
    )
    result = response.json()

    # Verify we got the right document
    if result.get("result", []) and result["result"][0] and result["result"][0][0].get("doc_id") == doc_id:
        return result

    # If not found or wrong document, try searching by content
    payload["queries"] = [doc_title]  # Try without exact match
    response = requests.post(
        f"{server_url}/retrieve", json=payload, headers={"Content-Type": "application/json"}, timeout=600
    )
    return response.json()


if __name__ == "__main__":
    # Test search
    print("\n=== Testing Search ===")
    print("Searching for: 'David Bowie'...")
    results = search_with_urls("David Bowie", num_results=2)

    print("\nSearch Results:")
    for i, doc in enumerate(results["result"][0], 1):
        print(f"\nDocument {i}:")
        print(f"ID: {doc['doc_id']}")  # Original corpus ID
        print(f"Title: {doc['title']}")
        print(f"Score: {doc.get('score', 'N/A')}")  # Show ranking score
        print(f"Preview: {doc['text'][:200]}...")  # Show preview of content
        print("-" * 80)

    # Test visit using first document's ID and title
    if results["result"][0]:
        first_doc = results["result"][0][0]
        doc_id = first_doc["doc_id"]
        doc_title = first_doc["title"]

        print(f"\n=== Testing Visit ===")
        print(f"Visiting document with ID: {doc_id}")
        print(f"Title: {doc_title}")
        doc_content = visit_site(doc_id, doc_title)

        print("\nDocument Content:")
        if doc_content["result"] and doc_content["result"][0]:
            doc = doc_content["result"][0][0]
            print(f"ID: {doc['doc_id']}")  # Verify we got same doc_id back
            print(f"Title: {doc['title']}")
            print(f"Full Content:\n{doc['text']}")

            if doc["doc_id"] != doc_id:
                print("\nWARNING: Retrieved document ID doesn't match requested ID!")
        else:
            print("No content found")
    else:
        print("\nNo search results to test visit functionality")

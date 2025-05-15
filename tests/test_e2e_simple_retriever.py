import pytest
import requests

# Assuming the simple_retriever service is running on localhost:8002
BASE_URL = "http://localhost:8002"


def test_health_check() -> None:
    """
    Tests the /health endpoint to ensure the service is up and reports being healthy.
    Assumes the service is started (e.g., via 'make docker-up') before running these tests.
    """
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        data = response.json()
        assert data["status"] == "healthy"
        assert "retrievers" in data
        assert (
            data["retrievers"]["available"] >= 0
        )  # Can be 0 if mock store fails, but should exist
        assert data["retrievers"]["total"] >= 0

    except requests.exceptions.ConnectionError:
        pytest.fail(
            f"ConnectionError: Could not connect to the service at {BASE_URL}. Ensure the service is running."
        )
    except requests.exceptions.HTTPError as e:
        pytest.fail(f"HTTPError: {e}. Response content: {response.text}")
    except KeyError as e:
        pytest.fail(
            f"KeyError: Missing expected key {e} in health check response. Response: {data}"
        )


def test_search_simple_query() -> None:
    """
    Tests the /search endpoint with a basic query.
    """
    try:
        payload = {"query": "test query", "top_n": 2}
        response = requests.post(f"{BASE_URL}/search", json=payload)
        response.raise_for_status()

        results = response.json()
        # In this mock setup, results is a list of Document objects
        assert isinstance(results, list)
        assert len(results) <= payload["top_n"]  # Should return up to top_n results
        if results:  # If there are results, check their structure
            assert "id" in results[0]
            assert "contents" in results[0]

    except requests.exceptions.ConnectionError:
        pytest.fail(
            f"ConnectionError: Could not connect to the service at {BASE_URL}. Ensure the service is running."
        )
    except requests.exceptions.HTTPError as e:
        pytest.fail(f"HTTPError: {e}. Response content: {response.text}")
    except (
        Exception
    ) as e:  # Catch other potential errors like JSONDecodeError or KeyErrors
        pytest.fail(
            f"An unexpected error occurred: {e}. Response content: {response.text if 'response' in locals() else 'N/A'}"
        )


# You can add more tests here for /batch_search or different scenarios for /search
# For example, testing with return_score=True:
# def test_search_with_scores() -> None:
#     ...


# Example for testing an empty query (expecting a 400 error)
def test_search_empty_query() -> None:
    """
    Tests the /search endpoint with an empty query, expecting a 400 error.
    """
    try:
        payload = {"query": "", "top_n": 1}
        response = requests.post(f"{BASE_URL}/search", json=payload)
        assert response.status_code == 400  # Bad Request
        # Optionally, check the detail message if your API provides consistent error details
        # error_data = response.json()
        # assert "Query content cannot be empty" in error_data.get("detail", "")

    except requests.exceptions.ConnectionError:
        pytest.fail(
            f"ConnectionError: Could not connect to the service at {BASE_URL}. Ensure the service is running."
        )
    # No pytest.fail on HTTPError here, as we expect a 400
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")

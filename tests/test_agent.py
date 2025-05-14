# tests/test_agent.py
import pytest

from src.agent import DeepResearchAgent, MockLLMClient
from src.config import (
    BEGIN_ANSWER,
    BEGIN_SEARCH,
    BEGIN_THINK,
    DEFAULT_LLM_MODEL,
    END_ANSWER,
    END_SEARCH,
    END_THINK,
    INFORMATION_END,
    INFORMATION_START,
    OPENROUTER_API_KEY,
)
from src.tools import Tool

# Store the original API key to restore it later if needed for other tests
ORIGINAL_OPENROUTER_API_KEY = OPENROUTER_API_KEY


@pytest.fixture(autouse=True)
def ensure_mock_llm_if_no_key(monkeypatch):
    """Fixture to ensure MockLLMClient is used if OPENROUTER_API_KEY is not set or is placeholder."""
    if not ORIGINAL_OPENROUTER_API_KEY or "YOUR_KEY_HERE" in ORIGINAL_OPENROUTER_API_KEY:
        monkeypatch.setattr("src.agent.OPENROUTER_API_KEY", None)
        monkeypatch.setattr("src.agent.OPENAI_API_KEY", None)
    yield
    # monkeypatch handles cleanup


def test_deep_research_agent_instantiation_default_mock() -> None:
    """Test agent instantiation when no API key forces MockLLMClient."""
    agent = DeepResearchAgent()
    assert isinstance(agent.llm_client, MockLLMClient)
    assert agent.llm_model_name == DEFAULT_LLM_MODEL
    assert agent.llm_model_name == "qwen/qwen3-32b"
    assert len(agent.tools) > 0


def test_deep_research_agent_instantiation_specific_tools() -> None:
    agent = DeepResearchAgent(tool_names=["search"])
    assert isinstance(agent.llm_client, MockLLMClient)
    assert "search" in agent.tools
    assert len(agent.tools) == 1
    assert isinstance(agent.tools["search"], Tool)


def test_deep_research_agent_run_mock_direct_answer() -> None:
    agent = DeepResearchAgent(llm_model_name=DEFAULT_LLM_MODEL)
    user_query = "What is the capital of France?"
    expected_answer = "The capital of France is Paris."
    final_response = agent.run(user_query)
    assert final_response == expected_answer


# MockLLMClient tests (directly)
def test_mock_llm_client_generates_search_for_france() -> None:
    client = MockLLMClient(model_name=DEFAULT_LLM_MODEL)
    messages = [{"role": "user", "content": "Please answer the following question: what is the capital of france?"}]
    response = client.generate(messages)
    assert (
        f"{BEGIN_THINK}The user wants to know the capital of France. I should search for this.{END_THINK}{BEGIN_SEARCH}capital of France{END_SEARCH}"
        == response
    )


def test_mock_llm_client_generates_answer_for_france_after_info() -> None:
    client = MockLLMClient(model_name=DEFAULT_LLM_MODEL)
    messages = [
        {"role": "user", "content": "Please answer the following question: what is the capital of france?"},
        {
            "role": "assistant",
            "content": f"{BEGIN_THINK}I should search.{END_THINK}{BEGIN_SEARCH}capital of France{END_SEARCH}",
        },
        {
            "role": "user",
            "content": f"{INFORMATION_START}Paris is the capital...{INFORMATION_END}",
        },
    ]
    response = client.generate(messages)
    assert f"{BEGIN_ANSWER}The capital of France is Paris.{END_ANSWER}" in response
    assert f"{BEGIN_THINK}I have found information about the capital of France.{END_THINK}" in response


def test_mock_llm_client_generic_query_search() -> None:
    client = MockLLMClient(model_name=DEFAULT_LLM_MODEL)
    messages = [{"role": "user", "content": "Please answer the following question: what is the capital of france?"}]
    response = client.generate(messages)
    assert f"{BEGIN_SEARCH}capital of France{END_SEARCH}" in response

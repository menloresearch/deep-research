# tests/test_prompts.py
from src.config import (
    BEGIN_ANSWER as FINAL_ANSWER_TOKEN,  # FINAL_ANSWER_TOKEN in tests refers to the agent's tag for starting an answer
)
from src.config import INFORMATION_END, INFORMATION_START
from src.prompts import (
    build_initial_user_prompt,
    format_search_results_for_llm,
    get_agent_system_prompt,
)
from src.tools import Tool


# Mock Tool for testing get_agent_system_prompt
class MockTool(Tool):
    def __init__(self, name: str, description: str, schema: str):
        super().__init__(name, description)
        self._schema = schema

    def execute(self, args: str | dict, full_context: dict | None = None) -> str:
        return "Mocked execution"

    def get_schema(self) -> str:
        return self._schema


def test_get_agent_system_prompt_no_tools() -> None:
    prompt = get_agent_system_prompt([])
    assert isinstance(prompt, str)
    assert "Cutting Knowledge Date:" in prompt
    assert "You are a helpful and meticulous AI research assistant" in prompt
    assert "Available Tools:" not in prompt
    assert "You have no tools available for use." in prompt


def test_get_agent_system_prompt_with_tools() -> None:
    mock_tool1 = MockTool("tool1", "Description 1", "<tool1>query1</tool1>")
    mock_tool2 = MockTool("tool2", "Description 2", "<tool2>query2</tool2>")
    tools = [mock_tool1, mock_tool2]
    prompt = get_agent_system_prompt(tools)

    assert isinstance(prompt, str)
    assert "Cutting Knowledge Date:" in prompt
    assert "You are a helpful and meticulous AI research assistant" in prompt
    assert "You have access to the following tools:" in prompt
    assert "- Tool Name: tool1" in prompt
    assert "Description: Description 1" in prompt
    assert "How to use: <tool1>query1</tool1>" in prompt
    assert "- Tool Name: tool2" in prompt
    assert "Description: Description 2" in prompt
    assert "How to use: <tool2>query2</tool2>" in prompt
    assert FINAL_ANSWER_TOKEN in prompt


def test_build_initial_user_prompt() -> None:
    user_query = "Tell me about llamas."
    prompt = build_initial_user_prompt(user_query)
    assert isinstance(prompt, str)
    assert user_query in prompt
    assert "Please answer the following question:" in prompt


def test_format_search_results_for_llm_empty() -> None:
    results: list[dict] = []
    formatted_string = format_search_results_for_llm(results)
    assert formatted_string == "No search results found."


def test_format_search_results_for_llm_single_result() -> None:
    results = [
        {
            "title": "Llama Facts",
            "url": "http://example.com/llama",
            "snippet": "Llamas are cool.",
        }
    ]
    formatted_string = format_search_results_for_llm(results)
    assert "Doc 1:" in formatted_string
    assert "(Title: Llama Facts)" in formatted_string
    assert "(URL: http://example.com/llama)" in formatted_string
    assert "\nLlamas are cool." in formatted_string
    assert INFORMATION_START not in formatted_string
    assert INFORMATION_END not in formatted_string


def test_format_search_results_for_llm_multiple_results() -> None:
    results = [
        {
            "title": "Alpaca Info",
            "url": "http://example.com/alpaca",
            "snippet": "Alpacas are fluffy.",
        },
        {
            "title": "Guanaco Details",
            "url": "http://example.com/guanaco",
            "snippet": "Guanacos are wild.",
        },
    ]
    formatted_string = format_search_results_for_llm(results)
    assert (
        "Doc 1:(Title: Alpaca Info)(URL: http://example.com/alpaca)\nAlpacas are fluffy."
        in formatted_string
    )
    assert (
        "Doc 2:(Title: Guanaco Details)(URL: http://example.com/guanaco)\nGuanacos are wild."
        in formatted_string
    )


def test_format_search_results_for_llm_missing_keys() -> None:
    results = [{"title": "Only Title", "snippet": "Just a snippet"}]
    formatted_string = format_search_results_for_llm(results)
    assert "Doc 1:(Title: Only Title)\nJust a snippet" in formatted_string


def test_format_search_results_for_llm_only_content_key() -> None:
    results = [{"content": "This is pure content."}]
    formatted_string = format_search_results_for_llm(results)
    assert "Doc 1:\nThis is pure content." in formatted_string


def test_format_search_results_for_llm_various_missing_keys() -> None:
    results = [
        {"title": "Title1", "url": "url1", "snippet": "snippet1"},
        {"url": "url2", "content": "content2"},
        {"title": "Title3", "snippet": "snippet3"},
        {"title": "Title4", "url": "url4"},
        {},
        {"page_info": "direct page info"},
    ]
    formatted_string = format_search_results_for_llm(results)
    assert "Doc 1:(Title: Title1)(URL: url1)\nsnippet1" in formatted_string
    assert "Doc 2:(URL: url2)\ncontent2" in formatted_string
    assert "Doc 3:(Title: Title3)\nsnippet3" in formatted_string
    assert "Doc 4:(Title: Title4)(URL: url4)\nN/A" in formatted_string
    assert "Doc 5:\nN/A" in formatted_string
    assert "Doc 6:\ndirect page info" in formatted_string

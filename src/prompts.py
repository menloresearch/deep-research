# src/prompts.py
from datetime import datetime
from typing import Any

from .config import (
    BEGIN_ANSWER,
    BEGIN_SEARCH,
    BEGIN_THINK,
    END_ANSWER,
    END_SEARCH,
    END_THINK,
)


def get_tool_descriptions(
    tools: list[Any],
) -> str:  # Use list[Tool] once Tool is defined and importable
    """Generates a string describing available tools for the LLM."""
    if not tools:
        return "You have no tools available for use."

    descriptions = "You have access to the following tools:\n"
    for tool in tools:
        descriptions += f"- Tool Name: {tool.name}\n"
        descriptions += f"  Description: {tool.description}\n"
        # The schema tells the LLM how to format the call
        descriptions += f"  How to use: {tool.get_schema()}\n"
    descriptions += "\n"
    descriptions += "When you decide to use a tool, you MUST use the exact format specified for that tool. Only one primary tool call per turn."
    return descriptions


def get_agent_system_prompt(available_tools: list[Any]) -> str:  # Use list[Tool]
    """Get the system prompt for the DeepResearchAgent."""
    current_date = datetime.now().strftime("%d %b %Y")
    # Base prompt for the agent
    prompt = f"""Cutting Knowledge Date: December 2023
Today Date: {current_date}

You are a helpful and meticulous AI research assistant. Your goal is to answer the user's question thoroughly and accurately.
Always reason about your plan and next steps within {BEGIN_THINK} and {END_THINK} tags before taking any action or providing an answer.
Based on your reasoning, you can choose one of the following actions:
1. If you need more information or need to perform an operation, use one of the available tools.
2. If you have sufficient information and are confident in your answer, provide the final answer directly within {BEGIN_ANSWER} and {END_ANSWER} tags.

{get_tool_descriptions(available_tools if available_tools else [])}

Follow these strict rules:
- You MUST start every response with your reasoning enclosed in {BEGIN_THINK} and {END_THINK} tags.
- After thinking, you MUST choose to either call exactly one tool OR provide the final answer. Do not do both.
- If calling a tool, use the precise format specified for that tool.
- Tool results will be provided to you in a subsequent message, typically enclosed in <information> tags.
- You can use tools multiple times if needed to gather sufficient information.
- Only provide the final answer once you are confident. If unsure, continue researching with tools.
- Your final response MUST end with either a tool call (e.g., {BEGIN_SEARCH}query{END_SEARCH}) OR a final answer ({BEGIN_ANSWER}your answer{END_ANSWER}).
"""
    return prompt


def build_initial_user_prompt(question: str) -> str:
    """
    Builds the initial user prompt that kicks off the agent's process.
    The system prompt already contains detailed instructions, so this can be simpler.
    """
    # The agent's system prompt already details how to think, search, and answer.
    # This user prompt just provides the question.
    return f"Please answer the following question: {question}"


def format_search_results_for_llm(results: list[dict[str, str]] | str) -> str:
    """
    Formats search results (from RAG or Web) into a string for the LLM.
    Each result should be clearly demarcated.
    Input `results` can be a list of dicts (e.g., [{'title': 'T1', 'snippet': 'S1', 'url': 'U1', 'content': 'C1'}])
    or a pre-formatted string.
    """
    if isinstance(results, str):  # Already formatted
        return results

    if not results:
        return "No search results found."

    content_parts = []
    for i, res_item in enumerate(results):
        doc_str = f"Doc {i + 1}:"
        if isinstance(res_item, dict):
            if res_item.get("title"):
                doc_str += f"(Title: {res_item['title']})"
            if res_item.get("url"):  # Useful for web search results
                doc_str += f"(URL: {res_item['url']})"
            doc_str += "\n"
            # Prefer 'content' if available (e.g., from RAG or fetched page), then 'snippet'
            page_info = (
                res_item.get("page_info")
                or res_item.get("content")
                or res_item.get("snippet", "N/A")
            )
            doc_str += page_info
        elif isinstance(res_item, str):  # Raw string content
            doc_str += f"\n{res_item}"
        else:
            doc_str += "\n[Unsupported result format]"
        content_parts.append(doc_str)

    return "\n\n".join(content_parts)


# Your original format_search_results, keep if used by rewards or other parts directly
def format_search_results(results: str | list[str]) -> str:
    """
    Format search results for display, matching the format from infer.py.
    Each result should be in the format: "Doc X(Title: Y) content"
    This seems more for human-readable display or specific reward expectations.
    The `format_search_results_for_llm` is for feeding back to the LLM.
    """
    if isinstance(results, list):
        if any("Doc" in r and "Title:" in r for r in results):
            content = "\n".join(results)
        else:
            # If results are raw content, format them with default titles
            content = "\n".join(
                [
                    f"Doc {i + 1}(Title: Document {i + 1})\n{r}"
                    for i, r in enumerate(results)
                ]
            )
    else:  # Single string
        if "Doc" in results and "Title:" in results:
            content = results
        else:
            content = f"Doc 1(Title: Document 1)\n{results}"
    return content

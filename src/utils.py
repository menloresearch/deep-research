# src/utils.py
import re
from typing import Optional, Tuple


def extract_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Extracts text between the first occurrence of start_tag and end_tag.
    Returns None if tags are not found or in incorrect order.
    """
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None

    # Adjust start_idx to be after the start_tag
    start_idx += len(start_tag)

    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None

    return text[start_idx:end_idx].strip()


def parse_agent_action(response_text: str) -> Optional[Tuple[str, str]]:
    """
    Parses the LLM response for a single primary action (tool call or answer).
    This is a simplified parser based on the current prompt structure.
    It expects one main action tag like <search> or <answer>.
    """
    from .config import (  # Local import to avoid circular dependencies at module load time
        BEGIN_ANSWER,
        BEGIN_CLICK_LINK,
        BEGIN_SEARCH,
        BEGIN_WRITE_SECTION,
        END_ANSWER,
        END_CLICK_LINK,
        END_SEARCH,
        END_WRITE_SECTION,
        BEGIN_GENERATE_OUTLINE,
        END_GENERATE_OUTLINE,
        BEGIN_GENERATE_TABLE,
        END_GENERATE_TABLE,
        BEGIN_EVALUATE_CONTENT,
        END_EVALUATE_CONTENT,
        # Add other primary action tags here
    )

    action_parsers = {
        "search": (BEGIN_SEARCH, END_SEARCH),
        "answer": (BEGIN_ANSWER, END_ANSWER),
        "click_link": (BEGIN_CLICK_LINK, END_CLICK_LINK),
        "write_section": (BEGIN_WRITE_SECTION, END_WRITE_SECTION),
        "generate_outline": (BEGIN_GENERATE_OUTLINE, END_GENERATE_OUTLINE), 
        "generate_table": (BEGIN_GENERATE_TABLE, END_GENERATE_TABLE),
        "evaluate_content": (BEGIN_EVALUATE_CONTENT, END_EVALUATE_CONTENT),
        # Add more tools that are invoked as primary actions
    }

    for action_name, (start_tag, end_tag) in action_parsers.items():
        content = extract_between_tags(response_text, start_tag, end_tag)
        if content is not None:  # Check for not None, as empty string can be valid content
            return action_name, content

    return None  # No recognizable primary action tag found


def clean_llm_output(text: str) -> str:
    """
    Basic cleaning of LLM output, removing extraneous common artifacts.
    """
    if text is None:
        return ""
    # Example: remove "Assistant:" prefix if present
    text = re.sub(r"^\s*Assistant:\s*", "", text, flags=re.IGNORECASE).strip()
    # Add other cleaning rules as needed
    return text

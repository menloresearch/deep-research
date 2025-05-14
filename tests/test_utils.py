# tests/test_utils.py
from src.config import BEGIN_ANSWER, BEGIN_SEARCH, BEGIN_THINK, END_ANSWER, END_SEARCH, END_THINK
from src.utils import (
    clean_llm_output,
    extract_between_tags,
    parse_agent_action,
)


# Tests for clean_llm_output
def test_clean_llm_output_basic() -> None:
    text = "  Hello World!  \n\n "
    assert clean_llm_output(text) == "Hello World!"


def test_clean_llm_output_no_change() -> None:
    text = "Hello World!"
    assert clean_llm_output(text) == "Hello World!"


def test_clean_llm_output_empty() -> None:
    text = ""
    assert clean_llm_output(text) == ""


def test_clean_llm_output_only_whitespace() -> None:
    text = "   \n\t  "
    assert clean_llm_output(text) == ""


# Tests for extract_between_tags
def test_extract_between_tags_single_match() -> None:
    text = f"Some preamble {BEGIN_THINK}This is the thought.{END_THINK} Some postamble"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) == "This is the thought."


def test_extract_between_tags_no_match_start_tag_missing() -> None:
    text = f"This is the thought.{END_THINK} Some postamble"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) is None


def test_extract_between_tags_no_match_end_tag_missing() -> None:
    text = f"Some preamble {BEGIN_THINK}This is the thought. Some postamble"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) is None


def test_extract_between_tags_no_tags_present() -> None:
    text = "Some preamble This is the thought. Some postamble"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) is None


def test_extract_between_tags_empty_content() -> None:
    text = f"Some preamble {BEGIN_THINK}{END_THINK} Some postamble"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) == ""


def test_extract_between_tags_multiple_matches_first_returned() -> None:
    text = f"{BEGIN_THINK}First thought{END_THINK} then {BEGIN_THINK}Second thought{END_THINK}"
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) == "First thought"


def test_extract_between_tags_with_whitespace_around_content() -> None:
    text = f"{BEGIN_THINK}  \nThought content  \n{END_THINK}"
    # extract_between_tags now strips content by default due to .strip() in its implementation
    assert extract_between_tags(text, BEGIN_THINK, END_THINK) == "Thought content"


def test_extract_between_tags_custom_tags() -> None:
    text = "<custom>data</custom>"
    assert extract_between_tags(text, "<custom>", "</custom>") == "data"


# Tests for parse_agent_action
def test_parse_agent_action_search() -> None:
    text = f"{BEGIN_THINK}I need to search.{END_THINK}{BEGIN_SEARCH}llamas{END_SEARCH}"
    action = parse_agent_action(text)
    assert action is not None
    tool_name, tool_args = action
    assert tool_name == "search"
    assert tool_args == "llamas"


def test_parse_agent_action_answer() -> None:
    text = f"{BEGIN_THINK}I will answer.{END_THINK}{BEGIN_ANSWER}Llamas are fluffy.{END_ANSWER}"
    action = parse_agent_action(text)
    assert action is not None
    tool_name, tool_args = action
    assert tool_name == "answer"
    assert tool_args == "Llamas are fluffy."


def test_parse_agent_action_no_action_tag() -> None:
    text = f"{BEGIN_THINK}Just thinking, no action.{END_THINK}"
    assert parse_agent_action(text) is None


def test_parse_agent_action_only_search_tag_no_think() -> None:
    # parse_agent_action looks for the first valid action tag pair after any think block
    text = f"{BEGIN_SEARCH}query without thought block first{END_SEARCH}"
    action = parse_agent_action(text)
    assert action is not None
    tool_name, tool_args = action
    assert tool_name == "search"
    assert tool_args == "query without thought block first"


def test_parse_agent_action_empty_args_in_search() -> None:
    text = f"{BEGIN_THINK}Thinking...{END_THINK}{BEGIN_SEARCH}{END_SEARCH}"
    action = parse_agent_action(text)
    assert action is not None
    tool_name, tool_args = action
    assert tool_name == "search"
    assert tool_args == ""


def test_parse_agent_action_multiple_actions_first_is_taken() -> None:
    # The current implementation of parse_agent_action iterates through known action tags
    # and returns the first one found.
    # The order of checking is usually based on a predefined list or dict iteration order.
    # For this test, let's assume 'search' might be checked before 'answer' or vice-versa.
    # We need to know the order in parse_agent_action or test for a specific known order.
    # src.utils.ACTION_TAGS defines this order. Let's assume it's [("search", (BEGIN_SEARCH, END_SEARCH)), ...]
    text_search_first = f"{BEGIN_SEARCH}search query{END_SEARCH}{BEGIN_ANSWER}answer content{END_ANSWER}"
    action_search_first = parse_agent_action(text_search_first)
    assert action_search_first is not None
    # This depends on the internal order of checks in parse_agent_action
    # If search is checked first:
    assert action_search_first[0] == "search"
    assert action_search_first[1] == "search query"

    # text_answer_first = f"{BEGIN_ANSWER}answer content{END_ANSWER}{BEGIN_SEARCH}search query{END_SEARCH}"
    # action_answer_first = parse_agent_action(text_answer_first)
    # assert action_answer_first is not None
    # If answer is checked first (or search is not present before it):
    # assert action_answer_first[0] == "answer"
    # assert action_answer_first[1] == "answer content"
    # For simplicity, this specific test relies on the known current order in ACTION_TAGS in utils.py


def test_parse_agent_action_unknown_action_tags() -> None:
    text = f"{BEGIN_THINK}Thinking...{END_THINK}<unknown_tool>args</unknown_tool>"
    assert parse_agent_action(text) is None


def test_parse_agent_action_malformed_action_tag() -> None:
    text = f"{BEGIN_THINK}Thinking...{END_THINK}{BEGIN_SEARCH}missing end tag"
    assert parse_agent_action(text) is None

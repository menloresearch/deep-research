# tests/test_config.py
from pathlib import Path

from src.config import (
    # Agent & App Configs
    AGENT_MAX_TURNS,
    BEGIN_ANSWER,
    BEGIN_SEARCH,
    # Special Tokens (check they are strings)
    BEGIN_THINK,
    DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    END_ANSWER,
    END_SEARCH,
    END_THINK,
    MAX_GENERATION_TOKENS,
    PROJ_ROOT,
    # Numeric Search Configs
    RAG_SEARCH_RESULTS_COUNT,
    # Core Paths (check existence or type)
    SRC_DIR,
    WEB_SEARCH_RESULTS_COUNT,
)


def test_path_constants_are_paths() -> None:
    assert isinstance(SRC_DIR, Path)
    assert isinstance(PROJ_ROOT, Path)
    assert isinstance(DATA_DIR, Path)


def test_string_constants_are_strings() -> None:
    assert isinstance(DEFAULT_LLM_MODEL, str)
    assert len(DEFAULT_LLM_MODEL) > 0
    assert isinstance(DEFAULT_EMBEDDING_MODEL, str)
    assert len(DEFAULT_EMBEDDING_MODEL) > 0
    assert isinstance(BEGIN_THINK, str)
    assert isinstance(END_THINK, str)
    assert isinstance(BEGIN_SEARCH, str)
    assert isinstance(END_SEARCH, str)
    assert isinstance(BEGIN_ANSWER, str)
    assert isinstance(END_ANSWER, str)


def test_integer_constants_are_integers() -> None:
    assert isinstance(AGENT_MAX_TURNS, int)
    assert AGENT_MAX_TURNS > 0
    assert isinstance(MAX_GENERATION_TOKENS, int)
    assert MAX_GENERATION_TOKENS > 0
    assert isinstance(RAG_SEARCH_RESULTS_COUNT, int)
    assert RAG_SEARCH_RESULTS_COUNT >= 0
    assert isinstance(WEB_SEARCH_RESULTS_COUNT, int)
    assert WEB_SEARCH_RESULTS_COUNT >= 0


# Example for a specific value if it's critical and fixed
# def test_specific_critical_constant_value() -> None:
#     assert BEGIN_THINK == "<think>"
# This is already implicitly tested by other tests using these constants,
# so direct value checks for all might be redundant for "simple" tests.

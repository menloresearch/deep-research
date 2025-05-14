# src/__init__.py
"""
Main package exports for the Deep Research Model.
Includes agent, tools, data handling, prompts, and RL training helpers.
"""

# Configurations and Utilities
# Agent Logic
from .agent import DeepResearchAgent
from .config import (
    BEGIN_ANSWER,
    BEGIN_CLICK_LINK,
    BEGIN_SEARCH,
    BEGIN_THINK,
    BEGIN_WRITE_SECTION,
    END_ANSWER,
    END_CLICK_LINK,
    END_SEARCH,
    END_THINK,
    END_WRITE_SECTION,
    logger,
)
from .embeddings import CustomHuggingFaceEmbeddings

# Knowledge Base (formerly search_module)
from .knowledge_base import (
    get_qa_dataset,
    load_qa_data,
    load_vectorstore,
)
from .knowledge_base import (
    simple_rag_search as rag_search,
)

# Prompts and Embeddings (Core LLM and Data Prep)
from .prompts import (
    format_search_results,
    get_agent_system_prompt,
    get_tool_descriptions,
)

# RL Training Rewards
from .rewards import (
    reward_format,
)

# Tools
from .tools import (
    AVAILABLE_TOOLS_CLASSES,
    ClickAndFetchTool,
    RAGSearchTool,
    Tool,
    WebSearchTool,
    WriteSectionTool,
    get_all_tool_instances,
    get_tool_instance,
)
from .utils import extract_between_tags, parse_agent_action

__all__ = [
    # Config & Utils
    "logger",
    "extract_between_tags",
    "parse_agent_action",
    "BEGIN_THINK",
    "END_THINK",
    "BEGIN_SEARCH",
    "END_SEARCH",
    "BEGIN_ANSWER",
    "END_ANSWER",
    "BEGIN_CLICK_LINK",
    "END_CLICK_LINK",
    "BEGIN_WRITE_SECTION",
    "END_WRITE_SECTION",
    # Prompts & Embeddings
    "format_search_results",
    "get_agent_system_prompt",
    "get_tool_descriptions",
    "CustomHuggingFaceEmbeddings",
    # Knowledge Base
    "load_vectorstore",
    "rag_search",
    "load_qa_data",
    "get_qa_dataset",
    # Agent
    "DeepResearchAgent",
    # Tools
    "Tool",
    "RAGSearchTool",
    "WebSearchTool",
    "ClickAndFetchTool",
    "WriteSectionTool",
    "get_tool_instance",
    "get_all_tool_instances",
    "AVAILABLE_TOOLS_CLASSES",
    # Rewards
    "reward_format",
]

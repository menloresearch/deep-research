# src/tools.py
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import (
    BEGIN_CLICK_LINK,
    BEGIN_SEARCH,
    BEGIN_WEB_SEARCH,
    BEGIN_WRITE_SECTION,
    END_CLICK_LINK,
    END_SEARCH,
    END_WEB_SEARCH,
    END_WRITE_SECTION,
    logger,
)
from .knowledge_base import simple_rag_search
from .prompts import format_search_results_for_llm


# --- Base Tool Class ---
class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description  # For LLM to understand what the tool does

    @abstractmethod
    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        """
        Executes the tool with given arguments.
        full_context can provide additional information like conversation history.
        Returns a string representation of the tool's output.
        """
        pass

    @abstractmethod
    def get_schema(self) -> str:
        """
        Returns a string describing how to use the tool, for the LLM prompt.
        e.g., "<search>query_string</search>"
        """
        pass


# --- RAG Search Tool ---
class RAGSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="search",  # Corresponds to <search> tag in basic prompts
            description="Searches a local knowledge base for information relevant to a query. Use this for foundational knowledge, definitions, or specific context from provided documents.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        query = args if isinstance(args, str) else args.get("query", "")
        if not query:
            logger.warning("RAGSearchTool called with no query.")
            return "Error: No search query provided to RAGSearchTool."

        logger.info(f"RAGSearchTool executing with query: '{query}'")
        try:
            search_results_data = simple_rag_search(query)
            if not search_results_data:
                return "No relevant information found in the local knowledge base for your query."
            return format_search_results_for_llm(search_results_data)
        except Exception as e:
            logger.error(f"RAGSearchTool error for query '{query}': {e}", exc_info=True)
            return f"Error during local knowledge base search: {str(e)}"

    def get_schema(self) -> str:
        return f"{BEGIN_SEARCH}your_search_query{END_SEARCH}"


# --- Web Search Tool (Placeholder) ---
class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Searches the live internet using a search engine for up-to-date information, current events, or topics not covered by the local knowledge base.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        query = args if isinstance(args, str) else args.get("query", "")
        if not query:
            logger.warning("WebSearchTool called with no query.")
            return "Error: No search query provided to WebSearchTool."

        logger.info(f"WebSearchTool executing with query: '{query}'")
        # For synchronous execution in a synchronous agent loop, you might need to run async code
        # This is a common challenge. Solutions:
        # 1. Make the agent loop async (preferred for I/O bound tasks).
        # 2. Use asyncio.run() here (can cause issues if already in an event loop).
        # 3. Use a library like `nest_asyncio` if you must mix sync/async in tricky ways.
        try:
            # This is a simplified synchronous call for the example.
            # In a real scenario, this would involve an HTTP client.
            # Placeholder for actual Bing/Google search call
            # from search.bing_search import bing_web_search_async (from your demo)
            # search_api_results = await bing_web_search_async(query, BING_API_KEY, ...)
            # relevant_info = extract_relevant_info(search_api_results) # from demo
            # return format_search_results_for_llm(relevant_info) # format as list of dicts
            return format_search_results_for_llm(  # Simulate returning formatted results
                [
                    {
                        "title": f"Web Result 1 for {query}",
                        "snippet": "This is a snippet...",
                        "url": "http://example.com/1",
                    },
                    {
                        "title": f"Web Result 2 for {query}",
                        "snippet": "Another relevant snippet.",
                        "url": "http://example.com/2",
                    },
                ]
            )
        except Exception as e:
            logger.error(f"WebSearchTool error for query '{query}': {e}", exc_info=True)
            return f"Error during web search: {str(e)}"

    def get_schema(self) -> str:
        # Consider a more descriptive tag for web search if it's distinct from local search for the LLM
        return f"{BEGIN_WEB_SEARCH}your_web_query{END_WEB_SEARCH}"


# --- Click Link / Fetch Page Tool (Placeholder) ---
# This might be two tools: one to identify a URL to click, another to fetch/summarize.
# Or one tool that takes a URL.
class ClickAndFetchTool(Tool):
    def __init__(self):
        super().__init__(
            name="click_link",  # Corresponds to <|begin_click_link|>
            description="Fetches the content of a given web page URL and provides a summary or key information. Use this to get details from a specific web page found in search results.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        url = args if isinstance(args, str) else args.get("url", "")
        if not url:
            return "Error: No URL provided to ClickAndFetchTool."

        original_query = (
            full_context.get("original_user_query", "the current topic") if full_context else "the current topic"
        )
        logger.info(f"ClickAndFetchTool executing for URL: '{url}' with context: '{original_query}'")
        # Similar async challenge as WebSearchTool
        try:
            # 1. Fetch content (e.g., using aiohttp, Jina from your demo)
            #    raw_content = await fetch_page_content_async([url], use_jina=..., jina_api_key=...)
            #    page_text = raw_content.get(url, "Error: Could not fetch content.")
            # 2. If successful, summarize (e.g., using another LLM call or extraction logic)
            #    reader_prompt = get_click_web_page_reader_instruction(query_context, page_text[:20000]) # From demo
            #    summary = await llm_client.generate(reader_prompt, ...) # Needs access to an LLM client
            return f"Summary of content from {url} (stubbed). Details about the page would be here."
        except Exception as e:
            logger.error(f"ClickAndFetchTool error for URL '{url}': {e}", exc_info=True)
            return f"Error fetching or processing URL '{url}': {str(e)}"

    def get_schema(self) -> str:
        return f"{BEGIN_CLICK_LINK}URL_to_fetch_and_summarize{END_CLICK_LINK}"


# --- Write Section Tool (Placeholder - complex interaction) ---
class WriteSectionTool(Tool):
    def __init__(self):
        super().__init__(
            name="write_section",
            description="Signals the intent to write a specific section of a report. Provide the section name and the task/content to be written for that section.",
        )
        # This tool is more of a meta-action. The "execution" is handled by the agent
        # by re-prompting the LLM with specific instructions to generate the section.

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        # Args might be a string like "Section Name: Introduction\nTask: Write a brief intro about X."
        # Or a dict: {"section_name": "Introduction", "task": "Write a brief intro about X."}
        section_details = args if isinstance(args, str) else str(args)
        logger.info(f"WriteSectionTool called with details: {section_details}")

        # This tool's "output" to the agent's scratchpad is essentially a confirmation
        # that the agent should now proceed with generating the section.
        # The actual section content will come from a subsequent LLM call orchestrated by the agent.
        return f"Instruction to write section received. Details: '{section_details}'. Agent will now proceed to generate this section content."

    def get_schema(self) -> str:
        return f"{BEGIN_WRITE_SECTION}Section Name: [Your Section Title]\nTask: [Detailed description of what to write in this section, including key points or questions to address. Refer to previously gathered information if necessary.]{END_WRITE_SECTION}"


# --- Tool Registry & Getter ---
# Using a dictionary that maps tool names (as used in LLM prompts) to their classes
AVAILABLE_TOOLS_CLASSES: dict[str, type[Tool]] = {
    "search": RAGSearchTool,
    "web_search": WebSearchTool,
    "click_link": ClickAndFetchTool,
    "write_section": WriteSectionTool,
}


def get_tool_instance(name: str, **kwargs) -> Optional[Tool]:
    """Gets an instance of a tool by its name."""
    tool_name_cleaned = name.lower().strip()  # Clean up name
    tool_class = AVAILABLE_TOOLS_CLASSES.get(tool_name_cleaned)
    if tool_class:
        try:
            return tool_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate tool '{tool_name_cleaned}': {e}", exc_info=True)
            return None
    logger.warning(f"Tool '{tool_name_cleaned}' not found in AVAILABLE_TOOLS_CLASSES.")
    return None


def get_all_tool_instances(tool_names: Optional[list[str]] = None) -> dict[str, Tool]:
    """
    Gets instances for a list of tool names, or all available tools if tool_names is None.
    Returns a dictionary mapping the tool's official name to its instance.
    """
    instances: dict[str, Tool] = {}

    if tool_names is None:  # Instantiate all available tools
        names_to_instantiate = list(AVAILABLE_TOOLS_CLASSES.keys())
    else:
        names_to_instantiate = [name.lower().strip() for name in tool_names]

    for name_key in names_to_instantiate:
        if name_key in AVAILABLE_TOOLS_CLASSES:
            instance = get_tool_instance(name_key)  # Uses the cleaned name_key
            if instance:
                instances[instance.name] = instance  # Use tool.name (which should match name_key)
        else:
            logger.warning(f"Requested tool '{name_key}' not found in available tool classes.")

    return instances

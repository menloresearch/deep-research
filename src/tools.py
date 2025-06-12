# src/tools.py
from abc import ABC, abstractmethod
from typing import Any, Optional
import json

from .config import (
    BEGIN_CLICK_LINK,
    BEGIN_SEARCH,
    BEGIN_WEB_SEARCH,
    BEGIN_WRITE_SECTION,
    END_CLICK_LINK,
    END_SEARCH,
    END_WEB_SEARCH,
    END_WRITE_SECTION,
    BEGIN_GENERATE_OUTLINE,
    END_GENERATE_OUTLINE,
    BEGIN_GENERATE_TABLE,
    END_GENERATE_TABLE,
    BEGIN_EVALUATE_CONTENT,
    END_EVALUATE_CONTENT,
    logger
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


# --- Generate Outline Tool ---
class GenerateOutlineTool(Tool):
    def __init__(self):
        super().__init__(
            name="generate_outline",
            description="Generates a structured outline based on research results and context. Use this to organize information into a logical structure.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        # Args can be a string or dict containing research results and context
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for outline generation arguments."

        research_results = args.get("research_results", {})
        context = args.get("context", {})
        
        logger.info(f"GenerateOutlineTool executing with context: {context.get('main_topic', '')}")
        
        try:
            # Create a prompt for generating the outline
            prompt = f"""
            Based on the following research results and context, generate an appropriate outline structure.
            
            Main Topic: {context.get('main_topic', '')}
            Key Aspects: {', '.join(context.get('aspects', [])[:3])}
            
            Research Results Summary:
            {json.dumps(research_results, indent=2)}
            
            Generate a JSON array of strings, where each string is a main topic or subtopic.
            The outline should:
            1. Focus on answering the main query
            2. Be comprehensive yet concise
            3. Flow logically
            4. Include only relevant sections
            
            Return ONLY the JSON array of strings.
            """
            
            # Use the LLM client from the context if available
            llm_client = full_context.get("llm_client") if full_context else None
            if not llm_client:
                return "Error: LLM client not available for outline generation."
            
            response = llm_client.completion(
                [{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Extract and parse the JSON response
            json_str = response['text'].strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            outline = json.loads(json_str)
            return json.dumps(outline, indent=2)
            
        except Exception as e:
            logger.error(f"GenerateOutlineTool error: {e}", exc_info=True)
            return f"Error generating outline: {str(e)}"

    def get_schema(self) -> str:
        return f"{BEGIN_GENERATE_OUTLINE}Research Results: [JSON object with research results]\nContext: [JSON object with main topic and aspects]{END_GENERATE_OUTLINE}"


# --- Generate Table Tool ---
class GenerateTableTool(Tool):
    def __init__(self):
        super().__init__(
            name="generate_table",
            description="Generates an HTML table that best represents the given content. Use this to present information in a structured, tabular format.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        # Args can be a string or dict containing content and outline
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for table generation arguments."

        outline = args.get("outline", [])
        content = args.get("content", {})
        
        logger.info(f"GenerateTableTool executing for outline with {len(outline)} items")
        
        try:
            # Create a prompt for generating the table
            prompt = f"""
            Based on the following outline and content, generate an appropriate HTML table.
            
            Outline:
            {json.dumps(outline, indent=2)}
            
            Content:
            {json.dumps(content, indent=2)}
            
            Generate an HTML table that:
            1. Best represents the information structure
            2. Is well-formatted and readable
            3. Includes all relevant information
            4. Uses appropriate HTML styling
            
            Return ONLY the HTML table code.
            """
            
            # Use the LLM client from the context if available
            llm_client = full_context.get("llm_client") if full_context else None
            if not llm_client:
                return "Error: LLM client not available for table generation."
            
            response = llm_client.completion(
                [{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Extract the HTML table
            html_str = response['text'].strip()
            if "```html" in html_str:
                html_str = html_str.split("```html")[1].split("```")[0].strip()
            elif "```" in html_str:
                html_str = html_str.split("```")[1].split("```")[0].strip()
            
            return html_str
            
        except Exception as e:
            logger.error(f"GenerateTableTool error: {e}", exc_info=True)
            return f"Error generating table: {str(e)}"

    def get_schema(self) -> str:
        return f"{BEGIN_GENERATE_TABLE}Outline: [JSON array of outline items]\nContent: [JSON object with content for each outline item]{END_GENERATE_TABLE}"


# --- Evaluate Content Tool ---
class EvaluateContentTool(Tool):
    def __init__(self):
        super().__init__(
            name="evaluate_content",
            description="Evaluates content quality and identifies issues or areas for improvement. Use this to ensure high-quality, well-structured content.",
        )

    def execute(self, args: str | dict[str, Any], full_context: Optional[dict] = None) -> str:
        # Args can be a string or dict containing content and context
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for content evaluation arguments."

        content = args.get("content", "")
        context = args.get("context", {})
        section = args.get("section", "")
        
        logger.info(f"EvaluateContentTool executing for section: {section}")
        
        try:
            # Create a prompt for evaluating the content
            prompt = f"""
            Evaluate the following content for the section "{section}" and identify issues or areas for improvement.
            
            Content:
            {content}
            
            Context:
            - Main Topic: {context.get('main_topic', '')}
            - Aspects: {context.get('aspects', '')}
            
            Evaluate if the content has any of the following issues:
            1. Misalignment with the section topic
            2. Misalignment with the main topic and aspects
            3. Lack of depth or comprehensiveness
            4. Inconsistencies or contradictions
            5. Unsupported claims
            6. Poor organization or structure
            7. Lack of clarity or readability
            8. Missing important information
            9. Inappropriate tone or style
            10. Missing citations
            
            Return a JSON array of strings, where each string describes a specific issue.
            Maximum of 5 issues only.
            """
            
            # Use the LLM client from the context if available
            llm_client = full_context.get("llm_client") if full_context else None
            if not llm_client:
                return "Error: LLM client not available for content evaluation."
            
            response = llm_client.completion(
                [{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Extract and parse the JSON response
            json_str = response['text'].strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            issues = json.loads(json_str)
            return json.dumps(issues, indent=2)
            
        except Exception as e:
            logger.error(f"EvaluateContentTool error: {e}", exc_info=True)
            return f"Error evaluating content: {str(e)}"

    def get_schema(self) -> str:
        return f"{BEGIN_EVALUATE_CONTENT}Section: [Section name]\nContent: [Content to evaluate]\nContext: [JSON object with main topic and aspects]{END_EVALUATE_CONTENT}"


# --- Tool Registry & Getter ---
# Using a dictionary that maps tool names (as used in LLM prompts) to their classes
AVAILABLE_TOOLS_CLASSES: dict[str, type[Tool]] = {
    "search": RAGSearchTool,
    "web_search": WebSearchTool,
    "click_link": ClickAndFetchTool,
    "write_section": WriteSectionTool,
    "generate_outline": GenerateOutlineTool,
    "generate_table": GenerateTableTool,
    "evaluate_content": EvaluateContentTool,
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

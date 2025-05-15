# tests/test_tools.py
from src.config import (
    BEGIN_CLICK_LINK,
    BEGIN_SEARCH,
    BEGIN_WEB_SEARCH,
    BEGIN_WRITE_SECTION,
    END_CLICK_LINK,
    END_SEARCH,
    END_WEB_SEARCH,
    END_WRITE_SECTION,
)
from src.tools import (
    AVAILABLE_TOOLS_CLASSES,
    ClickAndFetchTool,
    RAGSearchTool,
    Tool,
    WebSearchTool,
    WriteSectionTool,
    get_all_tool_instances,
    get_tool_instance,
)


def test_get_all_tool_instances_default() -> None:
    tools = get_all_tool_instances()
    assert isinstance(tools, dict)
    assert len(tools) == len(AVAILABLE_TOOLS_CLASSES)
    for name, tool_instance in tools.items():
        assert name in AVAILABLE_TOOLS_CLASSES
        assert isinstance(tool_instance, Tool)
        assert tool_instance.name == name


def test_get_all_tool_instances_specific() -> None:
    tool_names = ["search", "web_search"]
    tools = get_all_tool_instances(tool_names)
    assert isinstance(tools, dict)
    assert len(tools) == 2
    assert "search" in tools
    assert "web_search" in tools
    assert isinstance(tools["search"], RAGSearchTool)
    assert isinstance(tools["web_search"], WebSearchTool)


def test_get_all_tool_instances_empty_list() -> None:
    tools = get_all_tool_instances([])
    assert isinstance(tools, dict)
    assert len(tools) == 0


def test_get_all_tool_instances_invalid_name() -> None:
    tools = get_all_tool_instances(["invalid_tool_name", "search"])
    assert isinstance(tools, dict)
    assert len(tools) == 1  # Only valid tool should be returned
    assert "search" in tools
    assert "invalid_tool_name" not in tools


def test_get_tool_instance_valid() -> None:
    tool = get_tool_instance("search")
    assert isinstance(tool, RAGSearchTool)
    assert tool.name == "search"


def test_get_tool_instance_invalid() -> None:
    tool = get_tool_instance("invalid_tool_name_qwerty")
    assert tool is None


# RAGSearchTool Tests
def test_rag_search_tool_instantiation() -> None:
    tool = RAGSearchTool()
    assert tool.name == "search"
    assert "Searches a local knowledge base" in tool.description


def test_rag_search_tool_get_schema() -> None:
    tool = RAGSearchTool()
    assert tool.get_schema() == f"{BEGIN_SEARCH}your_search_query{END_SEARCH}"


def test_rag_search_tool_execute_simple() -> None:
    tool = RAGSearchTool()
    # This test assumes simple_rag_search can handle queries that might not find results
    # or that it has a mockable/testable state for empty KBs.
    # For a truly simple test, we check if it returns a string without error.
    output = tool.execute("test query")
    assert isinstance(output, str)
    # A more robust test would mock simple_rag_search if it hits a real DB


def test_rag_search_tool_execute_no_query() -> None:
    tool = RAGSearchTool()
    output = tool.execute("")
    assert "Error: No search query provided" in output


# WebSearchTool Tests
def test_web_search_tool_instantiation() -> None:
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "Searches the live internet" in tool.description


def test_web_search_tool_get_schema() -> None:
    tool = WebSearchTool()
    assert tool.get_schema() == f"{BEGIN_WEB_SEARCH}your_web_query{END_WEB_SEARCH}"


def test_web_search_tool_execute_simple() -> None:
    tool = WebSearchTool()
    output = tool.execute("latest news")
    assert isinstance(output, str)
    assert "Web Result 1 for latest news" in output  # Checks stubbed response


def test_web_search_tool_execute_no_query() -> None:
    tool = WebSearchTool()
    output = tool.execute("")
    assert "Error: No search query provided" in output


# ClickAndFetchTool Tests
def test_click_and_fetch_tool_instantiation() -> None:
    tool = ClickAndFetchTool()
    assert tool.name == "click_link"
    assert "Fetches the content of a given web page URL" in tool.description


def test_click_and_fetch_tool_get_schema() -> None:
    tool = ClickAndFetchTool()
    assert (
        tool.get_schema()
        == f"{BEGIN_CLICK_LINK}URL_to_fetch_and_summarize{END_CLICK_LINK}"
    )


def test_click_and_fetch_tool_execute_simple() -> None:
    tool = ClickAndFetchTool()
    output = tool.execute("http://example.com")
    assert isinstance(output, str)
    assert (
        "Summary of content from http://example.com (stubbed)" in output
    )  # Checks stubbed response


def test_click_and_fetch_tool_execute_no_url() -> None:
    tool = ClickAndFetchTool()
    output = tool.execute("")
    assert "Error: No URL provided" in output


# WriteSectionTool Tests
def test_write_section_tool_instantiation() -> None:
    tool = WriteSectionTool()
    assert tool.name == "write_section"
    assert "Signals the intent to write a specific section" in tool.description


def test_write_section_tool_get_schema() -> None:
    tool = WriteSectionTool()
    schema = tool.get_schema()
    assert BEGIN_WRITE_SECTION in schema
    assert END_WRITE_SECTION in schema
    assert "Section Name: [Your Section Title]" in schema


def test_write_section_tool_execute_simple_str_args() -> None:
    tool = WriteSectionTool()
    details = "Section Name: Intro\nTask: Write intro"
    output = tool.execute(details)
    assert isinstance(output, str)
    assert f"Instruction to write section received. Details: '{details}'" in output


def test_write_section_tool_execute_simple_dict_args() -> None:
    tool = WriteSectionTool()
    details = {"section_name": "Conclusion", "task": "Summarize findings"}
    output = tool.execute(details)  # type: ignore
    assert isinstance(output, str)
    assert f"Instruction to write section received. Details: '{str(details)}'" in output

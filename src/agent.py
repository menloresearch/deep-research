# src/agent.py
from typing import Optional

from .config import (
    AGENT_MAX_TURNS,
    BEGIN_ANSWER,
    BEGIN_SEARCH,
    BEGIN_THINK,
    DEFAULT_LLM_MODEL,
    END_ANSWER,
    END_SEARCH,
    END_THINK,
    INFORMATION_END,
    INFORMATION_START,
    MAX_GENERATION_TOKENS,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    logger,
)
from .prompts import build_initial_user_prompt, get_agent_system_prompt
from .tools import Tool, get_all_tool_instances
from .utils import clean_llm_output, extract_between_tags, parse_agent_action

try:
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI library not found. LLM interactions will be mocked.")
    OpenAI = None


class DeepResearchAgent:
    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        tool_names: Optional[list[str]] = None,
        openrouter_api_key_override: Optional[str] = None,
        openai_api_key_override: Optional[str] = None,
    ):
        self.llm_model_name = llm_model_name
        self.tools: dict[str, Tool] = get_all_tool_instances(tool_names)

        if not OpenAI:
            logger.warning("OpenAI library not installed. Using MockLLMClient.")
            self.llm_client = MockLLMClient(model_name=self.llm_model_name)
        else:
            actual_openrouter_key = openrouter_api_key_override or OPENROUTER_API_KEY
            actual_openai_key = openai_api_key_override or OPENAI_API_KEY

            if actual_openrouter_key and "YOUR_KEY_HERE" not in actual_openrouter_key:
                logger.info(
                    f"Initializing LLM client for OpenRouter. Model: {self.llm_model_name}, Base URL: {OPENROUTER_BASE_URL}"
                )
                self.llm_client = OpenAI(
                    api_key=actual_openrouter_key, base_url=OPENROUTER_BASE_URL
                )
            elif actual_openai_key and "YOUR_KEY_HERE" not in actual_openai_key:
                logger.info(
                    f"Initializing LLM client for direct OpenAI. Model: {self.llm_model_name}"
                )
                self.llm_client = OpenAI(api_key=actual_openai_key)
            else:
                logger.warning(
                    "Neither OpenRouter nor OpenAI API key is valid or configured. Using MockLLMClient."
                )
                self.llm_client = MockLLMClient(model_name=self.llm_model_name)

        self.system_prompt = get_agent_system_prompt(list(self.tools.values()))
        self.conversation_history = []
        self.original_user_query: Optional[str] = None

    def _add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def _get_llm_response_sync(self) -> str:
        """Synchronous LLM call."""
        if isinstance(self.llm_client, MockLLMClient):
            return self.llm_client.generate(self.conversation_history)

        messages_for_llm = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        try:
            logger.debug(
                f"Sending to LLM ({self.llm_model_name}): Last message - {self.conversation_history[-1]['content'][:200]}..."
            )
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages_for_llm,
                temperature=0.5,
                max_tokens=MAX_GENERATION_TOKENS,
            )
            raw_content = response.choices[0].message.content
            response_text = raw_content if raw_content is not None else ""

            logger.debug(f"LLM Raw Response: {response_text[:300]}...")
            return clean_llm_output(response_text)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return f"{BEGIN_THINK}An error occurred while contacting the language model. I should inform the user or try a different approach.{END_THINK}{BEGIN_ANSWER}I'm sorry, I encountered an internal error and cannot proceed with your request at this moment.{END_ANSWER}"

    def run(self, user_query: str) -> str:
        self.conversation_history = []
        self.original_user_query = user_query

        initial_prompt_content = build_initial_user_prompt(user_query)
        self._add_to_history("user", initial_prompt_content)

        final_answer_content: Optional[str] = None

        for turn_count in range(AGENT_MAX_TURNS):
            logger.info(f"\n--- Agent Turn {turn_count + 1}/{AGENT_MAX_TURNS} ---")

            llm_response_text = self._get_llm_response_sync()
            self._add_to_history("assistant", llm_response_text)

            thought = extract_between_tags(llm_response_text, BEGIN_THINK, END_THINK)
            if thought:
                logger.info(f"Agent Thought: {thought}")
            else:
                logger.warning("LLM response did not contain valid <think> tags.")

            action_details = parse_agent_action(llm_response_text)

            if not action_details:
                logger.warning(
                    f"LLM response did not contain a recognizable action tag. Response: {llm_response_text}"
                )
                if BEGIN_ANSWER in llm_response_text:
                    final_answer_content = extract_between_tags(
                        llm_response_text, BEGIN_ANSWER, END_ANSWER
                    )
                    if final_answer_content:
                        logger.info(
                            f"Agent decided to Answer (implicitly parsed): {final_answer_content}"
                        )
                        break
                self._add_to_history(
                    "user",
                    f"{INFORMATION_START}Your response was not in the expected format. Please use a tool or provide an answer with correct tags.{INFORMATION_END}",
                )
                continue

            action_name, action_args_str = action_details

            if action_name == "answer":
                final_answer_content = action_args_str
                logger.info(f"Agent decided to Answer: {final_answer_content}")
                break

            elif action_name in self.tools:
                tool_to_use = self.tools[action_name]
                logger.info(
                    f"Agent wants to use Tool: '{tool_to_use.name}' with Args: '{action_args_str}'"
                )

                tool_context = {
                    "conversation_history": self.conversation_history,
                    "original_user_query": self.original_user_query,
                }
                try:
                    tool_output = tool_to_use.execute(
                        action_args_str, full_context=tool_context
                    )
                except Exception as e:
                    logger.error(
                        f"Error executing tool {tool_to_use.name}: {e}", exc_info=True
                    )
                    tool_output = f"Error: Tool '{tool_to_use.name}' failed with message: {str(e)}"

                logger.info(
                    f"Tool Output ({tool_to_use.name}) [shortened]:\n{tool_output[:300]}..."
                )

                tool_feedback_message = f"{INFORMATION_START}\nTool: {tool_to_use.name}\nArguments: {action_args_str}\nOutput:\n{tool_output}\n{INFORMATION_END}"
                self._add_to_history("user", tool_feedback_message)

            else:
                logger.warning(
                    f"LLM tried to call unknown/unavailable tool: '{action_name}'"
                )
                error_message = f"{INFORMATION_START}\nError: You tried to use a tool named '{action_name}', which is not available. Please choose from the list of available tools.\n{INFORMATION_END}"
                self._add_to_history("user", error_message)

            if final_answer_content:
                break

        if not final_answer_content:
            logger.warning("Agent reached max turns without a final answer.")
            last_thought = (
                extract_between_tags(
                    self.conversation_history[-1]["content"], BEGIN_THINK, END_THINK
                )
                if self.conversation_history
                else None
            )
            if last_thought:
                return f"Agent reached max turns. Last thought: {last_thought}"
            return "Agent could not determine an answer within the allowed turns."

        return final_answer_content


class MockLLMClient:
    def __init__(self, model_name="mock_model"):
        self.model_name = model_name
        logger.info(f"Initialized MockLLMClient with model: {self.model_name}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        # Determine the initial user query (stripping the standard prefix)
        initial_user_query_content = ""
        for msg in messages:
            if msg["role"] == "user":
                potential_query = msg["content"]
                prefix_to_strip = "please answer the following question: "
                if potential_query.lower().startswith(prefix_to_strip):
                    initial_user_query_content = potential_query[
                        len(prefix_to_strip) :
                    ].lower()
                else:
                    initial_user_query_content = potential_query.lower()
                break

        last_message_content = messages[-1]["content"].lower() if messages else ""
        # Simplified check: if the last message is an information block, assume results are present.
        has_search_results = INFORMATION_START.lower() in last_message_content

        response_content = ""

        # Scenario 1: "what is the capital of france?" (Llama scenario removed)
        if "what is the capital of france?" in initial_user_query_content:
            if has_search_results:
                response_content = f"{BEGIN_THINK}I have found information about the capital of France.{END_THINK}{BEGIN_ANSWER}The capital of France is Paris.{END_ANSWER}"
            else:  # Initial query, suggest search
                response_content = f"{BEGIN_THINK}The user wants to know the capital of France. I should search for this.{END_THINK}{BEGIN_SEARCH}capital of France{END_SEARCH}"

        # Default fallback for other queries
        else:
            if has_search_results:
                # If there are search results for an unrecognized query, attempt a generic answer
                response_content = f"{BEGIN_THINK}I have processed search results for '{initial_user_query_content}'.{END_THINK}{BEGIN_ANSWER}This is a mock answer for '{initial_user_query_content}' based on search results.{END_ANSWER}"
            else:
                # If it's an initial, unrecognized query, suggest search
                search_term = (
                    initial_user_query_content
                    if initial_user_query_content
                    else "the user's request"
                )
                response_content = f"{BEGIN_THINK}I need to find information about {search_term}.{END_THINK}{BEGIN_SEARCH}information about {search_term}{END_SEARCH}"

        return response_content

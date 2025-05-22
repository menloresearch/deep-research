import json
import mimetypes
import os
import re
import shutil
import threading
from typing import Any, Callable, Dict, List, Optional

import gradio as gr
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent, GoogleSearchTool, HfApiModel, LiteLLMModel, Tool
from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.gradio_ui import stream_to_gradio
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer


# Create a wrapper for our local model that mimics the expected interface
class LocalQwenModelWrapper:
    def __init__(
        self,
        model_id: Optional[str] = "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
        custom_role_conversions: Optional[Dict[str, str]] = None,
    ):
        self.model_id = model_id
        self.custom_role_conversions = custom_role_conversions or {}
        self._load_model()

    def _load_model(self):
        print(f"Loading local model {self.model_id}...")

        # Use simple auto device mapping - let transformers handle it
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        print("Model loaded successfully")

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        """Call method that processes chat messages and returns a response"""
        try:
            # Convert all message roles according to custom_role_conversions if needed
            processed_messages = []
            for message in messages:
                role = message.get("role", "user")
                if role in self.custom_role_conversions:
                    role = self.custom_role_conversions[role]
                content = message.get("content", "")
                processed_messages.append({"role": role, "content": content})

            # Format messages into a prompt
            prompt = self._format_messages_as_prompt(processed_messages)

            # Get model parameters from kwargs or use defaults
            max_new_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.7)

            print(f"Generating with max_new_tokens={max_new_tokens}, temperature={temperature}")

            # Generate the response - use the simple approach
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )

            # Decode and format the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the model's response (removing the original prompt)
            if prompt in generated_text:
                response_text = generated_text[len(prompt) :].strip()
            else:
                response_text = generated_text.strip()

            # Create response in the format expected by smolagents
            return {"role": "assistant", "content": response_text}

        except Exception as e:
            print(f"Error in model generation: {str(e)}")
            return {
                "role": "assistant",
                "content": f"I apologize, but I encountered an error processing your request: {str(e)}",
            }

    def _format_messages_as_prompt(self, messages):
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n\n"

        prompt += "Assistant: "
        return prompt


web_search = GoogleSearchTool(provider="serper")

print(web_search(query="Donald Trump news"))

# quit()
AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

# Define Model Choices
MODEL_CHOICES = [
    "LOCAL_MODEL:jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",  # Added local model option
    "openrouter/qwen/qwen3-30b-a3b",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/qwen/qwen3-32b",
    "openrouter/qwen/qwen3-235b-a22b",
]
DEFAULT_MODEL = MODEL_CHOICES[0]  # Set local model as default

text_limit = 20000
browser = SimpleTextBrowser(**BROWSER_CONFIG)

# Store local model instance to avoid reloading
local_model_cache = {}


# Agent creation in a factory function
def create_agent(selected_model_id: str):
    """Creates a fresh agent instance for each session with the selected model"""
    print(f"Creating agent with model: {selected_model_id}")

    # Check if this is a local model request
    if selected_model_id.startswith("LOCAL_MODEL:"):
        model_path = selected_model_id.split(":", 1)[1]
        print(f"Using local model at {model_path}")

        # Use cached model if available
        if model_path not in local_model_cache:
            local_model_cache[model_path] = LocalQwenModelWrapper(
                model_id=model_path,
                custom_role_conversions=custom_role_conversions,
            )

        dynamic_model = local_model_cache[model_path]
    else:
        # Use API model
        dynamic_model = LiteLLMModel(
            model_id=selected_model_id,
            custom_role_conversions=custom_role_conversions,
        )

    dynamic_ti_tool = TextInspectorTool(dynamic_model, text_limit)

    dynamic_web_tools = [
        web_search,
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        dynamic_ti_tool,
    ]

    return CodeAgent(
        model=dynamic_model,
        tools=[visualizer] + dynamic_web_tools,
        max_steps=10,
        verbosity_level=1,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
    )


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: Optional[str] = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(self.file_upload_folder):
                os.makedirs(self.file_upload_folder, exist_ok=True)

    def interact_with_agent(self, prompt, messages, session_state, selected_model_id: str):
        # Get or create session-specific agent
        if "agent" not in session_state or session_state.get("current_model_id") != selected_model_id:
            print(
                f"Model selection changed or no agent. Old: {session_state.get('current_model_id')}, New: {selected_model_id}"
            )
            session_state["agent"] = create_agent(selected_model_id)
            session_state["current_model_id"] = selected_model_id

        # Adding monitoring
        try:
            # log the existence of agent memory
            has_memory = hasattr(session_state["agent"], "memory")
            print(f"Agent has memory: {has_memory}")
            if has_memory:
                print(f"Memory type: {type(session_state['agent'].memory)}")

            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                messages.append(msg)
                yield messages
            yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            raise

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        if self.file_upload_folder is None:
            return gr.Textbox("No file upload folder configured", visible=True), file_uploads_log

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        assert self.file_upload_folder is not None  # Assure type checker this is not None
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            gr.Textbox(
                value="",
                interactive=False,
                placeholder="Please wait while Steps are getting populated",
            ),
            gr.Button(interactive=False),
        )

    def detect_device(self, request: gr.Request):
        # Check whether the user device is a mobile or a computer

        if not request:
            return "Unknown device"
        # Method 1: Check sec-ch-ua-mobile header
        is_mobile_header = request.headers.get("sec-ch-ua-mobile")
        if is_mobile_header:
            return "Mobile" if "?1" in is_mobile_header else "Desktop"

        # Method 2: Check user-agent string
        user_agent = request.headers.get("user-agent", "").lower()
        mobile_keywords = ["android", "iphone", "ipad", "mobile", "phone"]

        if any(keyword in user_agent for keyword in mobile_keywords):
            return "Mobile"

        # Method 3: Check platform
        platform = request.headers.get("sec-ch-ua-platform", "").lower()
        if platform:
            if platform in ['"android"', '"ios"']:
                return "Mobile"
            elif platform in ['"windows"', '"macos"', '"linux"']:
                return "Desktop"

        # Default case if no clear indicators
        return "Desktop"

    def launch(self, **kwargs):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Different layouts for mobile and computer devices
            @gr.render()
            def layout(request: gr.Request):
                device = self.detect_device(request)
                print(f"device - {device}")
                # Render layout with sidebar
                if device == "Desktop":
                    with gr.Blocks(
                        fill_height=True,
                    ):
                        file_uploads_log = gr.State([])
                        with gr.Sidebar():
                            gr.Markdown("""# open Deep Research - free the AI agents!
                            
                OpenAI just published [Deep Research](https://openai.com/index/introducing-deep-research/), an amazing assistant that can perform deep searches on the web to answer user questions.
                
                However, their agent has a huge downside: it's not open. So we've started a 24-hour rush to replicate and open-source it. Our resulting [open-Deep-Research agent](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) took the #1 rank of any open submission on the GAIA leaderboard! ✨
                
                You can try a simplified version here that uses `Qwen-Coder-32B` instead of `o1`.<br><br>""")

                            model_selector = gr.Dropdown(
                                choices=MODEL_CHOICES,
                                value=DEFAULT_MODEL,
                                label="Select Model",
                                info="Choose the AI model for the agent.",
                            )

                            with gr.Group():
                                gr.Markdown("**Your request**", container=True)
                                text_input = gr.Textbox(
                                    lines=3,
                                    label="Your request",
                                    container=False,
                                    placeholder="Enter your prompt here and press Shift+Enter or press the button",
                                )
                                launch_research_btn = gr.Button("Run", variant="primary")

                            # If an upload folder is provided, enable the upload feature
                            if self.file_upload_folder is not None:
                                upload_file = gr.File(label="Upload a file")
                                upload_status = gr.Textbox(
                                    label="Upload Status",
                                    interactive=False,
                                    visible=False,
                                )
                                upload_file.change(
                                    self.upload_file,
                                    [upload_file, file_uploads_log],
                                    [upload_status, file_uploads_log],
                                )

                            gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                            with gr.Row():
                                gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
                        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                        <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                        </div>""")

                        # Add session state to store session-specific data
                        session_state = gr.State({})  # Initialize empty state for each session
                        stored_messages = gr.State([])
                        chatbot = gr.Chatbot(
                            label="open-Deep-Research",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=False,
                            scale=1,
                            elem_id="my-chatbot",
                        )

                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            # Include session_state AND model_selector in function calls
                            [stored_messages, chatbot, session_state, model_selector],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            # Include session_state AND model_selector in function calls
                            [stored_messages, chatbot, session_state, model_selector],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )

                # Render simple layout
                else:
                    with gr.Blocks(
                        fill_height=True,
                    ):
                        gr.Markdown("""# open Deep Research - free the AI agents!
            _Built with [smolagents](https://github.com/huggingface/smolagents)_
            
            OpenAI just published [Deep Research](https://openai.com/index/introducing-deep-research/), a very nice assistant that can perform deep searches on the web to answer user questions.
            
            However, their agent has a huge downside: it's not open. So we've started a 24-hour rush to replicate and open-source it. Our resulting [open-Deep-Research agent](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) took the #1 rank of any open submission on the GAIA leaderboard! ✨
            
            You can try a simplified version below (uses `Qwen-Coder-32B` instead of `o1`, so much less powerful than the original open-Deep-Research)👇""")
                        # Add session state to store session-specific data
                        session_state = gr.State({})  # Initialize empty state for each session
                        stored_messages = gr.State([])
                        file_uploads_log = gr.State([])
                        chatbot = gr.Chatbot(
                            label="open-Deep-Research",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=True,
                            scale=1,
                        )
                        # If an upload folder is provided, enable the upload feature
                        if self.file_upload_folder is not None:
                            upload_file = gr.File(label="Upload a file")
                            upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                            upload_file.change(
                                self.upload_file,
                                [upload_file, file_uploads_log],
                                [upload_status, file_uploads_log],
                            )

                        model_selector_mobile = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=DEFAULT_MODEL,
                            label="Select Model",
                            info="Choose the AI model for the agent.",
                        )

                        text_input = gr.Textbox(
                            lines=1,
                            label="Your request",
                            placeholder="Enter your prompt here and press the button",
                        )
                        launch_research_btn = gr.Button(
                            "Run",
                            variant="primary",
                        )

                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            # Include session_state AND model_selector_mobile in function calls
                            [stored_messages, chatbot, session_state, model_selector_mobile],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            # Include session_state AND model_selector_mobile in function calls
                            [stored_messages, chatbot, session_state, model_selector_mobile],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )

        demo.launch(debug=True, share=True, **kwargs)


GradioUI().launch()

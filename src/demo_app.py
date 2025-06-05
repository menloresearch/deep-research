import json
import mimetypes
import os
import re
import shutil
import threading
from datetime import datetime
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent, GoogleSearchTool, HfApiModel, LiteLLMModel, OpenAIServerModel, Tool
from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.gradio_ui import stream_to_gradio

from evaluate.scripts.text_inspector_tool import TextInspectorTool
from evaluate.scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchToolTavily,
    SimpleTextBrowser,
    VisitToolTavily,
)
from evaluate.scripts.visual_qa import visualizer

web_search = SearchToolTavily()

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

# Replace previous model initialization with local vLLM server model
model = OpenAIServerModel(
    model_id="jan-hq/Qwen3-14B-v0.2-deepresearch-no-think-100-step",
    api_base="http://localhost:8000/v1/",  # Local vLLM server URL
    api_key="EMPTY",  # vLLM uses "EMPTY" as the default API key
    custom_role_conversions=custom_role_conversions,
    temperature=0.7,
    top_p=0.8,
)

text_limit = 20000
browser = SimpleTextBrowser(**BROWSER_CONFIG)


# Agent creation in a factory function
def create_agent():
    """Creates a fresh agent instance for each session"""
    print("Creating new agent instance")

    # Re-initialize browser and tools for each agent to ensure thread-safety
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    web_tools_for_agent = [
        web_search,  # Using Tavily search tool
        VisitToolTavily(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    return CodeAgent(
        model=model,
        tools=[visualizer] + web_tools_for_agent,
        max_steps=10,
        verbosity_level=1,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
    )


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):
        # Get or create session-specific agent
        if "agent" not in session_state:
            print("No existing agent in session, creating new one...")
            session_state["agent"] = create_agent()

        # Adding monitoring
        try:
            # log the existence of agent memory
            has_memory = hasattr(session_state["agent"], "memory")
            print(f"Agent has memory: {has_memory}")
            if has_memory:
                print(f"Memory type: {type(session_state['agent'].memory)}")

            now = datetime.now()
            current_date = f"Today is {now.strftime('%A, %B %d, %Y')}"
            # Show only the user's original prompt in the chat UI
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            # Send the date-prefixed prompt only to the agent
            for msg in stream_to_gradio(
                session_state["agent"], task=f"{current_date}. {prompt}", reset_agent_memory=False
            ):
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
        with gr.Blocks(
            theme="ocean",
            fill_height=True,
            css="""
            .primary-button {
                background-color: #ff5c00 !important;
                border-color: #ff5c00 !important;
            }
            .primary-button:hover {
                background-color: #e65300 !important;
                border-color: #e65300 !important;
            }
        """,
        ) as demo:
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
                            gr.Markdown("""# 🧠 Menlo Deep Research Demo

A powerful AI research assistant that can perform deep searches on the web to answer your questions.

This demo is built on top of [huggingface/smolagents](https://github.com/huggingface/smolagents)'s open-Deep-Research implementation.

Powered by Menlo's Qwen3-14b Deep-Research model and Tavily search.<br><br>""")

                            with gr.Group():
                                gr.Markdown("**Your request**", container=True)
                                text_input = gr.Textbox(
                                    lines=3,
                                    label="Your request",
                                    container=False,
                                    placeholder="Enter your prompt here and press Shift+Enter or press the button",
                                )
                                launch_research_btn = gr.Button(
                                    "Run", variant="primary", elem_classes=["primary-button"]
                                )

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
                            [stored_messages, chatbot, session_state],
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
                            [stored_messages, chatbot, session_state],
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
                        gr.Markdown("""# 🧠 Menlo Deep Research Demo

A powerful AI research assistant that can perform deep searches on the web to answer your questions.

This demo is built on top of [huggingface/smolagents](https://github.com/huggingface/smolagents)'s open-Deep-Research implementation, which took the #1 rank of any open submission on the GAIA leaderboard! ✨

Powered by a local Qwen model and Tavily search 👇""")
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

                        text_input = gr.Textbox(
                            lines=1,
                            label="Your request",
                            placeholder="Enter your prompt here and press the button",
                        )
                        launch_research_btn = gr.Button(
                            "Run",
                            variant="primary",
                            elem_classes=["primary-button"],
                        )

                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot, session_state],
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
                            [stored_messages, chatbot, session_state],
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

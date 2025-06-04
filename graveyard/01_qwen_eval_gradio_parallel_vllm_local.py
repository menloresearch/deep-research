import concurrent.futures
import csv
import datetime
import json
import mimetypes
import os
import re
import shutil
import threading
import zipfile
from typing import Optional

import gradio as gr
import pandas as pd
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

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    # VisitTool,
    VisitToolSerperAPI,
)
from scripts.visual_qa import visualizer


# Helper function to extract text from various agent response content types
def extract_text_from_agent_response(response_content) -> str:
    if isinstance(response_content, AgentText):
        # Rely on str() for AgentText if .text attribute is problematic
        return str(response_content)
    elif isinstance(response_content, str):
        return response_content
    elif isinstance(response_content, dict) and "text" in response_content:  # For generic dicts
        return response_content["text"]
    # Add more specific handlers if other AgentOutput types are common and have text
    return str(response_content)  # Fallback


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

# Replace previous model initialization with local vLLM server model
model = OpenAIServerModel(
    model_id="jan-hq/Qwen3-14B-v0.2-deepresearch-200-step",
    api_base="http://localhost:8000/v1/",  # Local vLLM server URL
    api_key="EMPTY",  # vLLM uses "EMPTY" as the default API key
)

# model = LiteLLMModel(
#     model_id="openrouter/qwen/qwen3-14b",
#     custom_role_conversions=custom_role_conversions,
# )


text_limit = 20000
ti_tool = TextInspectorTool(model, text_limit)

browser = SimpleTextBrowser(**BROWSER_CONFIG)

WEB_TOOLS = [
    web_search,
    # VisitTool(browser),
    VisitToolSerperAPI(browser),
    PageUpTool(browser),
    PageDownTool(browser),
    FinderTool(browser),
    FindNextTool(browser),
    ArchiveSearchTool(browser),
    TextInspectorTool(model, text_limit),
]


# Agent creation in a factory function
def create_agent():
    """Creates a fresh agent instance for each session"""
    # Re-initialize browser and tools for each agent to ensure thread-safety if needed,
    # or to reset state for interactive mode if desired.
    # This is crucial if browser/tools have internal state that shouldn't be shared across sessions/threads.
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    web_tools_for_agent = [
        web_search,  # Assuming web_search is stateless or its state is managed elsewhere/globally
        VisitToolSerperAPI(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),  # Assuming TextInspectorTool is stateless or its state is managed
    ]

    return CodeAgent(
        model=model,
        tools=[visualizer] + web_tools_for_agent,
        max_steps=10,
        verbosity_level=1,  # Default verbosity
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
    )


document_inspection_tool = TextInspectorTool(model, 20000)


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(self.file_upload_folder):
                os.makedirs(self.file_upload_folder, exist_ok=True)

        self.batch_output_folder = "batch_outputs"
        if not os.path.exists(self.batch_output_folder):
            os.makedirs(self.batch_output_folder, exist_ok=True)

    def interact_with_agent(self, prompt, messages, session_state):
        # Get or create session-specific agent
        if "agent" not in session_state:
            session_state["agent"] = create_agent()

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
            return gr.Textbox(
                "File upload feature is not configured (no upload folder specified).", visible=True
            ), file_uploads_log

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
        sanitized_name_candidate = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Use os.path.splitext for robust base and extension splitting
        base_name, original_sanitized_extension = os.path.splitext(sanitized_name_candidate)

        final_sanitized_name = base_name  # Start with the sanitized base name

        if mime_type:  # If MIME type is known
            guessed_ext_from_mime = mimetypes.guess_extension(mime_type)
            if guessed_ext_from_mime:
                final_sanitized_name += guessed_ext_from_mime
            elif original_sanitized_extension:  # MIME known, but no standard ext, use original sanitized ext
                final_sanitized_name += original_sanitized_extension
            # If MIME known, no standard ext, and no original ext, name remains base_name
        elif original_sanitized_extension:  # MIME not known, use original sanitized ext if it exists
            final_sanitized_name += original_sanitized_extension
        # If MIME not known and no original ext, name remains base_name

        # Ensure final_sanitized_name is not empty if original was e.g. ".bashrc" -> base="", ext=".bashrc"
        if not final_sanitized_name and original_sanitized_extension:  # Handles cases like '.filename'
            final_sanitized_name = original_sanitized_extension
        elif not final_sanitized_name and not original_sanitized_extension:  # Original filename was empty or just dots
            return gr.Textbox("Error: Invalid filename after sanitization.", visible=True), file_uploads_log

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(final_sanitized_name))
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
        user_agent_header = request.headers.get("user-agent", "").lower()
        mobile_keywords = ["android", "iphone", "ipad", "mobile", "phone"]

        if any(keyword in user_agent_header for keyword in mobile_keywords):
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

    def process_csv_file(self, input_csv_file_obj, progress=gr.Progress(track_tqdm=True)):
        if input_csv_file_obj is None:
            gr.Warning("No CSV file provided for batch processing.")
            return None

        input_file_path = input_csv_file_obj.name

        try:
            df = pd.read_csv(input_file_path, keep_default_na=False)
            if "query" not in df.columns:
                gr.Error("CSV file must contain a 'query' column.")
                return None

            original_headers = df.columns.tolist()
            input_row_dicts = df.to_dict(orient="records")
            total_queries = len(input_row_dicts)

            if total_queries == 0:
                gr.Warning("CSV file contains no data rows.")
                return None

            # Determine fieldnames for the output CSV, ensuring 'answer' is last
            output_fieldnames = original_headers[:]
            if "answer" not in output_fieldnames:
                output_fieldnames.append("answer")
            elif "answer" in output_fieldnames and output_fieldnames[-1] != "answer":
                output_fieldnames.remove("answer")
                output_fieldnames.append("answer")

            # Ensure batch_outputs folder exists
            os.makedirs(self.batch_output_folder, exist_ok=True)

            # Create datetime-prefixed output filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename_base = os.path.splitext(os.path.basename(input_file_path))[0]
            # Limit filename length if needed and replace invalid characters
            safe_filename_base = re.sub(r"[^\w\-\.]", "_", original_filename_base)[:100]
            output_filename = f"{timestamp}_{safe_filename_base}_processed.csv"
            output_file_path = os.path.join(self.batch_output_folder, output_filename)

            # Open the output file at the start
            csv_output_file = open(output_file_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_output_file, fieldnames=output_fieldnames, extrasaction="ignore")
            csv_writer.writeheader()

            # Create a lock for writing to the CSV file
            csv_write_lock = threading.Lock()

            def process_row(row_data_with_index):
                idx, row_data = row_data_with_index
                agent = create_agent()  # Create a new agent for each thread/row
                query_text = row_data.get("query")
                current_output_row = row_data.copy()

                if pd.isna(query_text) or not isinstance(query_text, str) or not str(query_text).strip():
                    current_output_row["answer"] = "Skipped: Invalid, empty, or non-string query"
                    return current_output_row, idx

                final_answer_text = "No textual answer from agent for this query."
                last_assistant_content = None
                try:
                    for chat_message in stream_to_gradio(agent, task=str(query_text), reset_agent_memory=True):
                        if (
                            isinstance(chat_message, gr.ChatMessage)
                            and chat_message.role == "assistant"
                            and chat_message.content is not None
                        ):
                            last_assistant_content = chat_message.content

                    if last_assistant_content is not None:
                        final_answer_text = extract_text_from_agent_response(last_assistant_content)

                except Exception as e:
                    final_answer_text = f"Error processing query '{str(query_text)[:50]}...': {str(e)}"
                    # Optionally log detailed error to console here if needed

                current_output_row["answer"] = final_answer_text
                return current_output_row, idx

            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Store futures to retrieve results in order of submission if needed, or just process as completed
                future_to_row_index = {
                    executor.submit(process_row, (i, input_row_dicts[i])): i for i in range(total_queries)
                }

                for i, future in enumerate(concurrent.futures.as_completed(future_to_row_index)):
                    original_idx = future_to_row_index[future]
                    try:
                        processed_row_data, _ = future.result()

                        # Write each processed row immediately to the CSV file
                        with csv_write_lock:
                            csv_writer.writerow(processed_row_data)
                            csv_output_file.flush()  # Ensure data is written to disk
                    except Exception as exc:
                        # Handle exceptions from the process_row function itself
                        failed_row_data = input_row_dicts[original_idx].copy()
                        failed_row_data["answer"] = f"Failed to process row due to: {exc}"

                        # Write error row to the CSV file
                        with csv_write_lock:
                            csv_writer.writerow(failed_row_data)
                            csv_output_file.flush()

                    progress(
                        (i + 1) / total_queries,
                        desc=f"Processing row {i + 1}/{total_queries} - Saving to {output_filename}",
                    )

            # Close the output file
            csv_output_file.close()

            gr.Info(f"Processing complete. Output saved to {output_file_path}")
            return output_file_path

        except pd.errors.EmptyDataError:
            gr.Error("The uploaded CSV file is empty or not valid.")
            return None
        except Exception as e:
            gr.Error(f"Error reading or processing CSV: {str(e)}")
            print(f"Detailed error during CSV processing: {e}")

            # Close the file if it was opened
            if "csv_output_file" in locals() and not csv_output_file.closed:
                csv_output_file.close()

            return None

    def launch(self, **kwargs):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            current_device = gr.State(value="Desktop")  # Default, will be updated
            demo.load(
                self.detect_device, inputs=None, outputs=[current_device]
            )  # inputs=None if no explicit inputs needed

            gr.Markdown("# 🕵️‍♂️ Smol Research Agent 🕵️‍♀️")

            with gr.Tabs():
                with gr.TabItem("Interactive Chat"):
                    session_state = gr.State(value={})  # Session state for agent instance

                    with gr.Row():
                        with gr.Column(scale=1):  # Column for uploads
                            upload_button = gr.UploadButton(
                                "Click to Upload File(s)",
                                file_count="multiple",
                                file_types=[".pdf", ".docx", ".txt"],
                            )
                            file_output_status = gr.Textbox(
                                lines=1, label="File Upload Status", interactive=False, visible=True
                            )
                            file_uploads_log_state = gr.State([])  # State to hold uploaded file paths

                    with gr.Row():
                        with gr.Column(scale=3):  # Main chat column
                            chatbot_display = gr.Chatbot(
                                [],
                                elem_id="chatbot",
                                bubble_full_width=False,
                                show_copy_button=True,
                                avatar_images=(  # Tuple of (user_avatar, bot_avatar)
                                    None,  # Placeholder for user avatar, or path to an image
                                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",  # Bot avatar
                                ),
                                label="Smol Research Agent Chat",
                            )
                            chat_input_area = gr.Textbox(
                                scale=4,  # Relative width
                                show_label=False,
                                placeholder="Enter text and press enter, or upload a file",
                                container=False,  # Part of the row, not a separate container
                                interactive=True,
                                autofocus=True,
                            )
                            send_button_el = gr.Button("Send", variant="primary")

                    # Wire up components for interactive chat
                    upload_button.upload(
                        self.upload_file,
                        [upload_button, file_uploads_log_state],
                        [file_output_status, file_uploads_log_state],
                    )

                    # Actions for sending a message (via submit or button click)
                    chat_submit_action = chat_input_area.submit(
                        self.log_user_message,
                        [chat_input_area, file_uploads_log_state],
                        [
                            chat_input_area,
                            file_output_status,
                            send_button_el,
                        ],  # log_user_message clears input, updates status
                    ).then(
                        self.interact_with_agent,
                        [chat_input_area, chatbot_display, session_state],  # Pass NEW content of chat_input_area
                        [chatbot_display],
                    )

                    send_button_el.click(
                        self.log_user_message,
                        [chat_input_area, file_uploads_log_state],
                        [chat_input_area, file_output_status, send_button_el],
                    ).then(
                        self.interact_with_agent,
                        [chat_input_area, chatbot_display, session_state],
                        [chatbot_display],
                    )

                with gr.TabItem("Batch Processing (CSV)"):
                    with gr.Column():
                        gr.Markdown(
                            "Upload a CSV file with a 'query' column. Each query will be processed in parallel. "
                            "Results are saved in batches of 10 rows to intermediate CSV files. "
                            "Finally, all generated batch CSVs will be zipped together for download. "
                            "If there are 10 or fewer rows, a single CSV file will be provided directly."
                        )
                        input_csv_file_uploader = gr.File(label="Input CSV File", file_types=[".csv"])
                        process_csv_action_button = gr.Button("Process CSV File", variant="primary")
                        # Output for the processed file, user can download from here
                        output_csv_download_link = gr.File(label="Download Processed CSV", interactive=False)

                        process_csv_action_button.click(
                            self.process_csv_file, inputs=[input_csv_file_uploader], outputs=[output_csv_download_link]
                        )

            demo.queue().launch(**kwargs)


if __name__ == "__main__":
    ui = GradioUI(file_upload_folder="./file_uploads")
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)

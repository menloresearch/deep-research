import concurrent.futures
import csv
import datetime
import json
import logging
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
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
    VisitTool,
    VisitToolSerperAPI,
)
from scripts.visual_qa import visualizer


# Setup comprehensive logging
def setup_comprehensive_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"research_agent_{timestamp}.log")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up comprehensive logging - capture EVERYTHING
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,  # Override any existing configuration
    )

    # Set specific loggers to capture more info
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("smolagents").setLevel(logging.DEBUG)
    logging.getLogger("gradio").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pdfminer.psparser").setLevel(logging.WARNING)  # Reduce PDF parsing noise

    # Create a custom stream to capture stdout/stderr
    class LoggingStream:
        def __init__(self, original_stream, log_level):
            self.original_stream = original_stream
            self.log_level = log_level
            self.logger = logging.getLogger("STDOUT_STDERR")

        def write(self, data):
            if data.strip():  # Only log non-empty lines
                self.logger.log(self.log_level, f"[CAPTURED] {data.strip()}")
            self.original_stream.write(data)

        def flush(self):
            self.original_stream.flush()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

    # Redirect stdout and stderr to capture print statements
    sys.stdout = LoggingStream(sys.stdout, logging.INFO)
    sys.stderr = LoggingStream(sys.stderr, logging.ERROR)

    logger = logging.getLogger(__name__)
    logger.info(f"Comprehensive logging initialized. Log file: {log_file}")
    logger.info("All stdout/stderr output will be captured in logs")

    # Patch smolagents logging to ensure it uses our configuration
    def patch_smolagents_logging():
        try:
            import smolagents

            # Force smolagents to use our logging setup
            smolagents_logger = logging.getLogger("smolagents")
            smolagents_logger.setLevel(logging.DEBUG)
            logger.info("Smolagents logging patched successfully")

            # Also try to patch the agent's internal logging
            try:
                from smolagents.agent_types import logger as agent_logger

                if hasattr(agent_logger, "setLevel"):
                    agent_logger.setLevel(logging.DEBUG)
                    logger.info("Smolagents agent internal logging patched")
            except:
                pass

        except Exception as e:
            logger.warning(f"Could not patch smolagents logging: {e}")

    patch_smolagents_logging()

    return logger


logger = setup_comprehensive_logging()


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
    VisitTool(browser),
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
        interaction_start = time.time()
        logger.info(f"Starting agent interaction. Prompt length: {len(prompt)} characters")
        logger.debug(f"User prompt: {prompt[:500]}{'...' if len(prompt) > 500 else ''}")

        # Get or create session-specific agent
        if "agent" not in session_state:
            logger.info("No existing agent in session, creating new one...")
            session_state["agent"] = create_agent()
        else:
            logger.info("Using existing agent from session")

        # Adding monitoring
        try:
            # log the existence of agent memory
            has_memory = hasattr(session_state["agent"], "memory")
            logger.debug(f"Agent has memory: {has_memory}")
            if has_memory:
                memory_type = type(session_state["agent"].memory)
                logger.debug(f"Memory type: {memory_type}")

                # Log memory contents if available
                if hasattr(session_state["agent"].memory, "messages"):
                    memory_count = len(session_state["agent"].memory.messages)
                    logger.debug(f"Agent memory contains {memory_count} messages")

            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            step_count = 0
            response_start = time.time()
            logger.info("Starting agent task processing...")

            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                step_count += 1
                logger.debug(f"Agent step {step_count}: {type(msg).__name__}")

                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content_preview = str(msg.content)[:200] if msg.content else "None"
                    logger.debug(f"Message role: {msg.role}, content preview: {content_preview}...")

                    # Log tool calls and responses in detail
                    if msg.role == "assistant" and "tool_call" in str(msg.content).lower():
                        logger.info(f"Agent making tool call at step {step_count}")
                    elif msg.role == "user" and "tool_response" in str(type(msg.content)).lower():
                        logger.info(f"Tool response received at step {step_count}")

                messages.append(msg)
                yield messages

            processing_time = time.time() - response_start
            total_time = time.time() - interaction_start

            logger.info(
                f"Agent interaction completed. Steps: {step_count}, Processing time: {processing_time:.2f}s, Total time: {total_time:.2f}s"
            )
            yield messages

        except Exception as e:
            error_time = time.time() - interaction_start
            logger.error(f"Error in interaction after {error_time:.2f}s: {str(e)}")
            logger.exception("Full exception traceback:")
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
        upload_start = time.time()
        logger.info("File upload attempt started")

        if self.file_upload_folder is None:
            logger.warning("File upload attempted but no upload folder configured")
            return gr.Textbox(
                "File upload feature is not configured (no upload folder specified).", visible=True
            ), file_uploads_log

        if file is None:
            logger.warning("File upload attempted but no file provided")
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        logger.info(f"Processing file upload: {file.name}")

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
            logger.debug(f"File MIME type detected: {mime_type}")
        except Exception as e:
            logger.error(f"Error detecting MIME type for file {file.name}: {str(e)}")
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            logger.warning(f"File type {mime_type} not in allowed types: {allowed_file_types}")
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        logger.debug(f"Original filename: {original_name}")

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
            logger.error(f"Invalid filename after sanitization: {original_name}")
            return gr.Textbox("Error: Invalid filename after sanitization.", visible=True), file_uploads_log

        logger.debug(f"Sanitized filename: {final_sanitized_name}")

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(final_sanitized_name))

        try:
            file_size = os.path.getsize(file.name)
            logger.info(f"Copying file {file.name} ({file_size} bytes) to {file_path}")
            shutil.copy(file.name, file_path)

            upload_time = time.time() - upload_start
            logger.info(f"File upload completed successfully in {upload_time:.2f}s: {file_path}")

            return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

        except Exception as e:
            logger.error(f"Error copying file {file.name} to {file_path}: {str(e)}")
            return gr.Textbox(f"Error saving file: {e}", visible=True), file_uploads_log

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
        batch_start = time.time()
        logger.info("Starting CSV batch processing")

        if input_csv_file_obj is None:
            logger.warning("CSV batch processing attempted with no file")
            gr.Warning("No CSV file provided for batch processing.")
            return None

        input_file_path = input_csv_file_obj.name
        logger.info(f"Processing CSV file: {input_file_path}")

        try:
            df = pd.read_csv(input_file_path, keep_default_na=False)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            logger.debug(f"CSV columns: {df.columns.tolist()}")

            if "query" not in df.columns:
                logger.error("CSV file missing required 'query' column")
                gr.Error("CSV file must contain a 'query' column.")
                return None

            original_headers = df.columns.tolist()
            input_row_dicts = df.to_dict(orient="records")
            total_queries = len(input_row_dicts)
            logger.info(f"Total queries to process: {total_queries}")

            if total_queries == 0:
                logger.warning("CSV file contains no data rows")
                gr.Warning("CSV file contains no data rows.")
                return None

            # Determine fieldnames for the output CSV, ensuring 'answer' is last
            output_fieldnames = original_headers[:]
            if "answer" not in output_fieldnames:
                output_fieldnames.append("answer")
            elif "answer" in output_fieldnames and output_fieldnames[-1] != "answer":
                output_fieldnames.remove("answer")
                output_fieldnames.append("answer")
            logger.debug(f"Output CSV fieldnames: {output_fieldnames}")

            # Ensure batch_outputs folder exists
            os.makedirs(self.batch_output_folder, exist_ok=True)

            # Create datetime-prefixed output filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename_base = os.path.splitext(os.path.basename(input_file_path))[0]
            # Limit filename length if needed and replace invalid characters
            safe_filename_base = re.sub(r"[^\w\-\.]", "_", original_filename_base)[:100]
            output_filename = f"{timestamp}_{safe_filename_base}_processed.csv"
            output_file_path = os.path.join(self.batch_output_folder, output_filename)
            logger.info(f"Output will be saved to: {output_file_path}")

            # Open the output file at the start
            csv_output_file = open(output_file_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_output_file, fieldnames=output_fieldnames, extrasaction="ignore")
            csv_writer.writeheader()
            logger.debug("CSV output file created and header written")

            # Create a lock for writing to the CSV file
            csv_write_lock = threading.Lock()

            def process_row(row_data_with_index):
                idx, row_data = row_data_with_index
                row_start = time.time()
                logger.info(f"Processing row {idx + 1}/{total_queries}")

                agent = create_agent()  # Create a new agent for each thread/row
                query_text = row_data.get("query")
                current_output_row = row_data.copy()

                logger.debug(f"Row {idx + 1} query: {str(query_text)[:100]}...")

                if pd.isna(query_text) or not isinstance(query_text, str) or not str(query_text).strip():
                    logger.warning(f"Row {idx + 1} skipped: invalid query")
                    current_output_row["answer"] = "Skipped: Invalid, empty, or non-string query"
                    return current_output_row, idx

                final_answer_text = "No textual answer from agent for this query."
                last_assistant_content = None

                try:
                    processing_start = time.time()
                    logger.debug(f"Row {idx + 1} starting agent processing")

                    message_count = 0
                    for chat_message in stream_to_gradio(agent, task=str(query_text), reset_agent_memory=True):
                        message_count += 1
                        if (
                            isinstance(chat_message, gr.ChatMessage)
                            and chat_message.role == "assistant"
                            and chat_message.content is not None
                        ):
                            last_assistant_content = chat_message.content
                            logger.debug(f"Row {idx + 1} got assistant message {message_count}")

                    if last_assistant_content is not None:
                        final_answer_text = extract_text_from_agent_response(last_assistant_content)

                    processing_time = time.time() - processing_start
                    logger.info(
                        f"Row {idx + 1} processed successfully in {processing_time:.2f}s. Messages: {message_count}"
                    )

                except Exception as e:
                    processing_time = time.time() - processing_start if "processing_start" in locals() else 0
                    error_msg = f"Error processing query '{str(query_text)[:50]}...': {str(e)}"
                    final_answer_text = error_msg
                    logger.error(f"Row {idx + 1} failed after {processing_time:.2f}s: {str(e)}")
                    logger.exception(f"Row {idx + 1} exception details:")

                current_output_row["answer"] = final_answer_text

                row_time = time.time() - row_start
                logger.debug(f"Row {idx + 1} completed in {row_time:.2f}s total")
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
                            logger.debug(f"Row {original_idx + 1} result written to CSV ({i + 1}/{total_queries})")

                    except Exception as exc:
                        # Handle exceptions from the process_row function itself
                        failed_row_data = input_row_dicts[original_idx].copy()
                        failed_row_data["answer"] = f"Failed to process row due to: {exc}"

                        logger.error(f"Row {original_idx + 1} failed with exception: {str(exc)}")
                        logger.exception(f"Row {original_idx + 1} full exception:")

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

            processing_time = time.time() - batch_start
            logger.info(f"CSV batch processing completed successfully!")
            logger.info(f"Total time: {processing_time:.2f}s")
            logger.info(f"Average time per query: {processing_time / total_queries:.2f}s")
            logger.info(f"Output saved to: {output_file_path}")

            gr.Info(f"Processing complete. Output saved to {output_file_path}")
            return output_file_path

        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty or not valid")
            gr.Error("The uploaded CSV file is empty or not valid.")
            return None
        except Exception as e:
            error_time = time.time() - batch_start
            logger.error(f"CSV processing failed after {error_time:.2f}s: {str(e)}")
            logger.exception("Full exception traceback:")
            gr.Error(f"Error reading or processing CSV: {str(e)}")
            print(f"Detailed error during CSV processing: {e}")

            # Close the file if it was opened
            if "csv_output_file" in locals() and not csv_output_file.closed:
                csv_output_file.close()
                logger.debug("CSV output file closed due to error")

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
    logger.info("=== Starting Smol Research Agent Application ===")
    logger.info(f"Python script: {__file__}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(
        f"Environment variables: HF_TOKEN={'SET' if os.getenv('HF_TOKEN') else 'NOT SET'}, SERPAPI_API_KEY={'SET' if os.getenv('SERPAPI_API_KEY') else 'NOT SET'}"
    )

    # Log all available environment variables (excluding sensitive ones)
    sensitive_keys = ["token", "key", "secret", "password", "api"]
    env_vars = {k: v for k, v in os.environ.items() if not any(sens in k.lower() for sens in sensitive_keys)}
    logger.debug(f"Environment variables (non-sensitive): {env_vars}")

    try:
        ui = GradioUI(file_upload_folder="./file_uploads")
        logger.info("GradioUI instance created successfully")

        launch_config = {"server_name": "0.0.0.0", "server_port": 7860, "share": True, "debug": True}
        logger.info(f"Launching with config: {launch_config}")

        ui.launch(**launch_config)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.exception("Full exception traceback:")
        raise

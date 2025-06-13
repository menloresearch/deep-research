#!/usr/bin/env python3
"""
Flexible vLLM + MCP Research Agent Script (Retry-Enhanced Version)
Usage: python vllm_mcp_agent.py --csv path/to/input.csv --model_id your/model --base_url http://localhost:8000/v1
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv(override=True)

# Global logger
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


class VLLMMCPResearchAgent:
    """Research agent using vLLM + MCP + LangChain stack with retry mechanism."""
    
    def __init__(self, model_id: str, base_url: str, temperature: float = 0.7, 
                 max_tokens: Optional[int] = None, 
                 mcp_server_url: str = "http://127.0.0.1:8000/mcp",
                 initial_timeout: float = 60.0,
                 max_retries: int = 3,
                 retry_delay: float = 5.0,
                 timeout_multiplier: float = 2.0,
                 max_timeout: float = 600.0,
                 **model_kwargs):
        """
        Initialize the research agent with model parameters and retry configuration.
        
        Args:
            model_id: Model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct")
            base_url: Base URL for the vLLM server (e.g., "http://localhost:8000/v1")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            mcp_server_url: URL for MCP server
            initial_timeout: Initial timeout for HTTP requests in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            timeout_multiplier: Factor to multiply timeout on each retry
            max_timeout: Maximum timeout value in seconds
            **model_kwargs: Additional model parameters
        """
        self.model_id = model_id
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mcp_server_url = mcp_server_url
        self.initial_timeout = initial_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout_multiplier = timeout_multiplier
        self.max_timeout = max_timeout
        self.model_kwargs = model_kwargs
        self.chat_model = None
        self.mcp_client = None
        self.agent = None
        self.tools = None
        self.http_client = None
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "timeout_count": 0,
            "average_timeout_used": 0.0
        }
        
    async def initialize(self):
        """Initialize the chat model, MCP client, and agent with retry logic."""
        max_init_retries = 3
        
        for attempt in range(max_init_retries):
            try:
                logger.info(f"Initializing vLLM chat model (attempt {attempt + 1}/{max_init_retries}): {self.model_id}")
                logger.info(f"Base URL: {self.base_url}")
                logger.info(f"Temperature: {self.temperature}")
                logger.info(f"Initial timeout: {self.initial_timeout}s, Max retries: {self.max_retries}")
                
                # Create HTTP client with generous timeout for initialization
                init_timeout = min(self.initial_timeout * 3, 180.0)
                timeout = httpx.Timeout(init_timeout)
                limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
                self.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
                
                # Prepare model kwargs with timeout
                model_kwargs = {
                    "temperature": self.temperature,
                    "timeout": init_timeout,
                    **self.model_kwargs
                }
                
                if self.max_tokens:
                    model_kwargs["max_tokens"] = self.max_tokens
                
                # Initialize chat model pointing to vLLM server
                self.chat_model = init_chat_model(
                    model=self.model_id,
                    base_url=self.base_url,
                    api_key="your-key",
                    model_kwargs=model_kwargs,
                )
                
                logger.info(f"Initializing MCP client at: {self.mcp_server_url}")
                
                # Initialize MCP client with server configuration
                mcp_config = {
                    "serper-search": {
                        "url": self.mcp_server_url,
                        "transport": "streamable_http"
                    }
                }
                
                self.mcp_client = MultiServerMCPClient(mcp_config)
                
                logger.info("Getting tools from MCP servers...")
                self.tools = await asyncio.wait_for(
                    self.mcp_client.get_tools(), 
                    timeout=init_timeout
                )
                
                logger.info(f"Retrieved {len(self.tools)} tools from MCP servers")
                for tool in self.tools:
                    logger.debug(f"Available tool: {tool.name} - {tool.description}")
                
                logger.info("Creating ReAct agent...")
                
                system_prompt = """"""

                self.agent = create_react_agent(
                    self.chat_model, 
                    self.tools, 
                    prompt=system_prompt
                )
                
                logger.info("Agent initialization completed successfully")
                return
                
            except asyncio.TimeoutError:
                logger.warning(f"Initialization attempt {attempt + 1} timed out")
                if attempt < max_init_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying initialization in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError("Agent initialization failed after all retry attempts")
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_init_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying initialization in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to initialize agent after {max_init_retries} attempts: {str(e)}")
                    raise
    
    async def research_query_with_retry(self, query: str) -> Dict[str, Any]:
        """Process a single research query with retry mechanism."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.stats["total_requests"] += 1
        start_time = time.time()
        logger.info(f"Processing query with retry: {query[:100]}...")
        
        last_error = None
        current_timeout = self.initial_timeout
        total_timeout_used = 0
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if shutdown_requested:
                    logger.info("Shutdown requested, aborting query")
                    return self._create_error_result(query, "Shutdown requested", time.time() - start_time)
                
                attempt_start = time.time()
                logger.info(f"Query attempt {attempt + 1}/{self.max_retries + 1}, timeout: {current_timeout}s")
                
                # Use current timeout for this attempt
                result = await asyncio.wait_for(
                    self.agent.ainvoke({
                        "messages": [HumanMessage(content=query)]
                    }),
                    timeout=current_timeout
                )
                
                # Success! Extract the final AI response
                final_answer = "No response generated"
                messages = result.get("messages", [])
                
                # Find the last AI message with content
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content
                        break
                
                processing_time = time.time() - start_time
                actual_attempt_time = time.time() - attempt_start
                total_timeout_used += actual_attempt_time
                
                self.stats["successful_requests"] += 1
                self.stats["average_timeout_used"] = (
                    (self.stats["average_timeout_used"] * (self.stats["successful_requests"] - 1) + total_timeout_used) / 
                    self.stats["successful_requests"]
                )
                
                logger.info(f"Query completed successfully on attempt {attempt + 1}, took {actual_attempt_time:.2f}s")
                
                return {
                    "query": query,
                    "answer": final_answer,
                    "processing_time": processing_time,
                    "attempts": attempt + 1,
                    "total_timeout_used": total_timeout_used,
                    "message_count": len(messages),
                    "status": "success",
                    "model_id": self.model_id,
                    "temperature": self.temperature,
                    "full_conversation": [
                        {
                            "type": msg.__class__.__name__,
                            "content": msg.content,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        for msg in messages
                    ]
                }
                
            except asyncio.TimeoutError as e:
                attempt_time = time.time() - attempt_start
                total_timeout_used += attempt_time
                self.stats["timeout_count"] += 1
                last_error = f"Timeout after {current_timeout}s"
                
                logger.warning(f"Query attempt {attempt + 1} timed out after {current_timeout}s")
                
                if attempt < self.max_retries:
                    # Calculate next timeout and delay
                    next_timeout = min(current_timeout * self.timeout_multiplier, self.max_timeout)
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    logger.info(f"Retrying in {wait_time}s with timeout {next_timeout}s...")
                    await asyncio.sleep(wait_time)
                    
                    current_timeout = next_timeout
                    self.stats["total_retries"] += 1
                else:
                    logger.error(f"Query failed after {self.max_retries + 1} attempts (all timeouts)")
                    
            except Exception as e:
                attempt_time = time.time() - attempt_start
                total_timeout_used += attempt_time
                last_error = str(e)
                
                logger.error(f"Query attempt {attempt + 1} failed with error: {str(e)}")
                
                if attempt < self.max_retries:
                    # For non-timeout errors, wait and retry
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    self.stats["total_retries"] += 1
                else:
                    logger.error(f"Query failed after {self.max_retries + 1} attempts")
        
        # All attempts failed
        processing_time = time.time() - start_time
        self.stats["failed_requests"] += 1
        
        return self._create_error_result(
            query, 
            f"Failed after {self.max_retries + 1} attempts. Last error: {last_error}",
            processing_time,
            self.max_retries + 1,
            total_timeout_used
        )
    
    def _create_error_result(self, query: str, error_msg: str, processing_time: float, 
                           attempts: int = 1, total_timeout_used: float = 0) -> Dict[str, Any]:
        """Create a standardized error result."""
        return {
            "query": query,
            "answer": f"Error: {error_msg}",
            "processing_time": processing_time,
            "attempts": attempts,
            "total_timeout_used": total_timeout_used,
            "message_count": 0,
            "status": "error",
            "error": error_msg,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "full_conversation": []
        }
    
    async def research_query(self, query: str) -> Dict[str, Any]:
        """Backward compatibility wrapper."""
        return await self.research_query_with_retry(query)
    
    def print_stats(self):
        """Print processing statistics."""
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total requests: {self.stats['total_requests']}")
        logger.info(f"Successful: {self.stats['successful_requests']}")
        logger.info(f"Failed: {self.stats['failed_requests']}")
        logger.info(f"Total retries: {self.stats['total_retries']}")
        logger.info(f"Timeouts encountered: {self.stats['timeout_count']}")
        if self.stats['successful_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info(f"Average timeout used: {self.stats['average_timeout_used']:.2f}s")
    
    async def close(self):
        """Clean up resources."""
        logger.info("Cleaning up agent resources...")
        
        # Print final stats
        self.print_stats()
        
        if self.mcp_client:
            try:
                if hasattr(self.mcp_client, 'close'):
                    await self.mcp_client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
        
        if self.http_client:
            try:
                await self.http_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")


def setup_logging(log_dir: str = "logs", model_id: str = "unknown_model") -> logging.Logger:
    """Setup comprehensive logging."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.replace("/", "_").replace(":", "_")
    log_file = os.path.join(log_dir, f"vllm_mcp_research_{model_name}_{timestamp}.log")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"), 
            logging.StreamHandler()
        ],
        force=True,
    )

    # Set specific loggers
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def serialize_result(obj):
    """Custom JSON serializer for complex objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    try:
        return obj.__dict__
    except AttributeError:
        return str(obj)


async def process_single_query_async(agent: VLLMMCPResearchAgent, query_data: Dict[str, Any], 
                                   idx: int, output_dir: str) -> Dict[str, Any]:
    """Process a single query asynchronously with retry support."""
    global shutdown_requested
    
    if shutdown_requested:
        logger.info(f"Shutdown requested, skipping row {idx + 1}")
        return {
            **query_data,
            "answer": "Skipped: Shutdown requested",
            "status": "cancelled",
            "attempts": 0,
            "processing_time": 0
        }
    
    query_text = query_data.get("query", "")
    
    if pd.isna(query_text) or not isinstance(query_text, str) or not str(query_text).strip():
        logger.warning(f"Row {idx + 1} skipped: invalid query")
        return {
            **query_data,
            "answer": "Skipped: Invalid, empty, or non-string query",
            "status": "skipped",
            "attempts": 0,
            "processing_time": 0
        }
    
    # Process the query with retry
    result = await agent.research_query_with_retry(query_text)
    
    # Save detailed results to JSONL
    os.makedirs(output_dir, exist_ok=True)
    model_name = agent.model_id.replace("/", "_").replace(":", "_")
    jsonl_filename = f"{model_name}_row_{idx + 1}.jsonl"
    jsonl_filepath = os.path.join(output_dir, jsonl_filename)
    
    try:
        with open(jsonl_filepath, "w", encoding="utf-8") as jsonl_file:
            json.dump(result, jsonl_file, default=serialize_result, indent=2)
        logger.info(f"Row {idx + 1} detailed log saved to {jsonl_filename}")
    except Exception as e:
        logger.warning(f"Failed to save detailed log for row {idx + 1}: {e}")
    
    # Return the processed row with retry information
    return {
        **query_data,
        "answer": result["answer"],
        "processing_time": result["processing_time"],
        "attempts": result.get("attempts", 1),
        "total_timeout_used": result.get("total_timeout_used", 0),
        "status": result["status"]
    }


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


async def run_batch_processing_async(agent: VLLMMCPResearchAgent, input_csv_path: str, 
                                   output_csv_path: str, max_concurrent: int = 3,
                                   output_dir: str = "outputs"):
    """Run batch processing with async agent and progressive saving."""
    
    global shutdown_requested
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found at '{input_csv_path}'")

    batch_start = time.time()
    logger.info("Starting CSV batch processing with vLLM + MCP + LangChain (Retry-Enhanced)")
    logger.info(f"Processing CSV file: {input_csv_path}")
    logger.info(f"Output CSV: {output_csv_path}")
    logger.info(f"Max concurrent requests: {max_concurrent}")
    logger.info(f"Retry configuration: max_retries={agent.max_retries}, initial_timeout={agent.initial_timeout}s")

    # Initialize the agent
    await agent.initialize()
    
    try:
        # Load and validate CSV
        df = pd.read_csv(input_csv_path, keep_default_na=False)
        logger.info(f"CSV loaded successfully. Shape: {df.shape}")
        
        if "query" not in df.columns:
            raise ValueError("CSV file must contain a 'query' column")

        input_row_dicts = df.to_dict(orient="records")
        total_queries = len(input_row_dicts)
        logger.info(f"Total queries to process: {total_queries}")

        if total_queries == 0:
            logger.warning("CSV file contains no data rows")
            return

        # Setup output CSV headers
        original_headers = df.columns.tolist()
        output_fieldnames = original_headers[:]
        new_fields = ["answer", "processing_time", "attempts", "total_timeout_used", "status"]
        for field in new_fields:
            if field not in output_fieldnames:
                output_fieldnames.append(field)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)

        # Initialize output CSV file with headers
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames, extrasaction="ignore")
            writer.writeheader()

        # Process queries with controlled concurrency and progressive saving
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        
        async def bounded_process(query_data, idx):
            nonlocal completed_count
            async with semaphore:
                if shutdown_requested:
                    return None
                
                result = await process_single_query_async(agent, query_data, idx, output_dir)
                
                # Immediately save result to CSV (append mode)
                try:
                    with open(output_csv_path, "a", newline="", encoding="utf-8") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames, extrasaction="ignore")
                        writer.writerow(result)
                    
                    completed_count += 1
                    status_info = f"[{result.get('attempts', 1)} attempts, {result.get('total_timeout_used', 0):.1f}s]"
                    logger.info(f"Completed and saved row {idx + 1}/{total_queries} {status_info}")
                    
                except Exception as e:
                    logger.error(f"Failed to save row {idx + 1} to CSV: {e}")
                
                return result
        
        # Create tasks for all queries
        tasks = []
        for i in range(total_queries):
            if shutdown_requested:
                break
            task = asyncio.create_task(bounded_process(input_row_dicts[i], i))
            tasks.append(task)
        
        # Process with progress tracking
        results = []
        try:
            with tqdm(total=len(tasks), desc="Processing queries") as pbar:
                for coro in asyncio.as_completed(tasks):
                    if shutdown_requested:
                        logger.info("Shutdown requested, cancelling remaining tasks...")
                        break
                    
                    try:
                        result = await coro
                        if result is not None:
                            results.append(result)
                        pbar.update(1)
                        
                        # Log progress with retry stats
                        if completed_count % 5 == 0 or completed_count == total_queries:
                            logger.info(f"Progress: {completed_count}/{total_queries} queries completed")
                            agent.print_stats()
                            
                    except asyncio.CancelledError:
                        logger.info("Task was cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Task failed with error: {e}")
                        pbar.update(1)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, cancelling tasks...")
            shutdown_requested = True
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        processing_time = time.time() - batch_start
        
        if shutdown_requested:
            logger.info(f"Processing stopped due to shutdown request")
        else:
            logger.info(f"CSV batch processing completed successfully!")
        
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"Completed queries: {completed_count}/{total_queries}")
        if completed_count > 0:
            logger.info(f"Average time per query: {processing_time / completed_count:.2f}s")
        
        # Print final statistics
        agent.print_stats()
        
        logger.info(f"Output saved to: {output_csv_path}")
        print(f"\nProcessing complete. {completed_count}/{total_queries} queries processed.")
        print(f"Output saved to {output_csv_path}")

    except Exception as e:
        error_time = time.time() - batch_start
        logger.error(f"Batch processing failed after {error_time:.2f}s: {str(e)}")
        logger.exception("Full exception traceback:")
        raise
        
    finally:
        # Clean up agent
        try:
            await agent.close()
            logger.info("Agent cleanup completed")
        except Exception as e:
            logger.warning(f"Error during agent cleanup: {e}")


async def test_single_query(agent: VLLMMCPResearchAgent, test_query: str = None):
    """Test function for a single query with retry demonstration."""
    logger.info("=== Testing Single Query with Retry ===")
    
    try:
        await agent.initialize()
        
        if not test_query:
            test_query = "What are the latest developments in AI safety research in 2024?"
        
        result = await agent.research_query_with_retry(test_query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Status: {result['status']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Attempts: {result.get('attempts', 1)}")
        print(f"Total timeout used: {result.get('total_timeout_used', 0):.2f}s")
        print(f"Message count: {result['message_count']}")
        print(f"Model: {result['model_id']}")
        print(f"Temperature: {result['temperature']}")
        print(f"\nAnswer:\n{result['answer']}")
        
        # Print agent statistics
        agent.print_stats()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
        raise
    finally:
        await agent.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="vLLM + MCP Research Agent for batch CSV processing with retry mechanism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with retry
  python vllm_mcp_agent.py --csv queries.csv --model_id microsoft/Phi-3-mini-4k-instruct --base_url http://localhost:8000/v1

  # Custom retry configuration
  python vllm_mcp_agent.py --csv data.csv --model_id llama2-7b --base_url http://localhost:8000/v1 --initial_timeout 30 --max_retries 5 --max_timeout 300

  # Conservative settings for slow server
  python vllm_mcp_agent.py --csv data.csv --model_id your-model --base_url http://localhost:8000/v1 --initial_timeout 120 --max_retries 2 --max_concurrent 1

  # Test mode
  python vllm_mcp_agent.py --test --model_id your-model --base_url http://localhost:8000/v1 --query "Test query here"
        """
    )
    
    # Required arguments
    parser.add_argument("--csv", type=str, help="Path to input CSV file (required for batch processing)")
    parser.add_argument("--model_id", type=str, required=True, 
                       help="Model identifier (e.g., microsoft/Phi-3-mini-4k-instruct)")
    parser.add_argument("--base_url", type=str, required=True,
                       help="Base URL for vLLM server (e.g., http://localhost:8000/v1)")
    
    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.0 to 2.0, default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Maximum tokens to generate (default: model default)")
    parser.add_argument("--model_kwargs", type=str, default="{}",
                       help="Additional model parameters as JSON string")
    
    # Retry configuration
    parser.add_argument("--initial_timeout", type=float, default=60.0,
                       help="Initial timeout in seconds (default: 60)")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retry attempts (default: 3)")
    parser.add_argument("--retry_delay", type=float, default=5.0,
                       help="Base delay between retries in seconds (default: 5)")
    parser.add_argument("--timeout_multiplier", type=float, default=2.0,
                       help="Factor to multiply timeout on each retry (default: 2.0)")
    parser.add_argument("--max_timeout", type=float, default=600.0,
                       help="Maximum timeout value in seconds (default: 600)")
    
    # Processing parameters
    parser.add_argument("--max_concurrent", type=int, default=3,
                       help="Maximum concurrent requests (default: 3)")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path (default: auto-generated)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for detailed logs (default: outputs)")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Log directory (default: logs)")
    
    # MCP configuration
    parser.add_argument("--mcp_server_url", type=str, default="http://127.0.0.1:8000/mcp",
                       help="MCP server URL (default: http://127.0.0.1:8000/mcp)")
    
    # Test mode
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with a single query")
    parser.add_argument("--query", type=str, default=None,
                       help="Test query (for --test mode)")
    
    return parser.parse_args()


def main():
    """Main function to run the batch processing."""
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_arguments()
    
    # Validate arguments
    if not args.test and not args.csv:
        print("Error: --csv is required for batch processing (or use --test for test mode)")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_dir, args.model_id)
    
    logger.info("=== Starting vLLM + MCP Research Agent (Retry-Enhanced Version) ===")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Retry config: initial_timeout={args.initial_timeout}s, max_retries={args.max_retries}")
    logger.info(f"Timeout progression: {args.initial_timeout}s -> {args.max_timeout}s (x{args.timeout_multiplier})")
    logger.info(f"MCP Server: {args.mcp_server_url}")
    
    # Parse model_kwargs from JSON string
    try:
        model_kwargs = json.loads(args.model_kwargs)
        if not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a JSON object")
    except json.JSONDecodeError as e:
        print(f"Error parsing model_kwargs JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error with model_kwargs: {e}")
        sys.exit(1)
    
    # Create agent with retry parameters
    agent = VLLMMCPResearchAgent(
        model_id=args.model_id,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mcp_server_url=args.mcp_server_url,
        initial_timeout=args.initial_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout_multiplier=args.timeout_multiplier,
        max_timeout=args.max_timeout,
        **model_kwargs
    )

    try:
        if args.test:
            # Test mode
            asyncio.run(test_single_query(agent, args.query))
        else:
            # Batch processing mode
            input_csv_path = args.csv
            
            # Generate output CSV path if not provided
            if args.output_csv:
                output_csv_path = args.output_csv
            else:
                base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_csv_path = f"{base_name}_answered_{timestamp}.csv"
            
            logger.info(f"Input CSV: {input_csv_path}")
            logger.info(f"Output CSV: {output_csv_path}")
            logger.info(f"Max concurrent: {args.max_concurrent}")
            
            # Run batch processing directly with asyncio.run
            asyncio.run(run_batch_processing_async(
                agent=agent,
                input_csv_path=input_csv_path,
                output_csv_path=output_csv_path,
                max_concurrent=args.max_concurrent,
                output_dir=args.output_dir
            ))
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Full traceback:")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
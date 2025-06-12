#!/usr/bin/env python3
"""
Flexible vLLM + MCP Research Agent Script
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
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
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


class VLLMMCPResearchAgent:
    """Research agent using vLLM + MCP + LangChain stack."""
    
    def __init__(self, model_id: str, base_url: str, temperature: float = 0.7, 
                 max_tokens: Optional[int] = None, 
                 mcp_server_url: str = "http://127.0.0.1:8000/mcp",
                 **model_kwargs):
        """
        Initialize the research agent with model parameters.
        
        Args:
            model_id: Model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct")
            base_url: Base URL for the vLLM server (e.g., "http://localhost:8000/v1")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            mcp_server_url: URL for MCP server
            **model_kwargs: Additional model parameters (top_p, frequency_penalty, presence_penalty, etc.)
        """
        self.model_id = model_id
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mcp_server_url = mcp_server_url
        self.model_kwargs = model_kwargs
        self.chat_model = None
        self.mcp_client = None
        self.agent = None
        self.tools = None
        
    async def initialize(self):
        """Initialize the chat model, MCP client, and agent."""
        try:
            logger.info(f"Initializing vLLM chat model: {self.model_id}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Temperature: {self.temperature}")
            
            # Prepare model kwargs
            model_kwargs = {
                "temperature": self.temperature,
                **self.model_kwargs  # Include any additional model parameters
            }
            
            if self.max_tokens:
                model_kwargs["max_tokens"] = self.max_tokens
            
            # Initialize chat model pointing to vLLM server
            self.chat_model = init_chat_model(
                model=self.model_id,
                base_url=self.base_url,
                api_key="not-needed-but-required",  # Required by langchain-openai
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
            self.tools = await self.mcp_client.get_tools()
            
            logger.info(f"Retrieved {len(self.tools)} tools from MCP servers")
            for tool in self.tools:
                logger.debug(f"Available tool: {tool.name} - {tool.description}")
            
            logger.info("Creating ReAct agent...")
            
            # Create the ReAct agent with custom prompt
            system_prompt = """"""

            self.agent = create_react_agent(
                self.chat_model, 
                self.tools, 
                prompt=system_prompt
            )
            
            logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise
    
    async def research_query(self, query: str) -> Dict[str, Any]:
        """Process a single research query and return detailed results."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        start_time = time.time()
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Invoke the agent with the query
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=query)]
            })
            
            # Extract the final AI response
            final_answer = "No response generated"
            messages = result.get("messages", [])
            
            # Find the last AI message with content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break
            
            processing_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": final_answer,
                "processing_time": processing_time,
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
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed: {str(e)}")
            
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "processing_time": processing_time,
                "message_count": 0,
                "status": "error",
                "error": str(e),
                "model_id": self.model_id,
                "temperature": self.temperature,
                "full_conversation": []
            }
    
    async def close(self):
        """Clean up resources."""
        if self.mcp_client:
            if hasattr(self.mcp_client, 'close'):
                await self.mcp_client.close()


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
    """Process a single query asynchronously."""
    query_text = query_data.get("query", "")
    
    if pd.isna(query_text) or not isinstance(query_text, str) or not str(query_text).strip():
        logger.warning(f"Row {idx + 1} skipped: invalid query")
        return {
            **query_data,
            "answer": "Skipped: Invalid, empty, or non-string query",
            "status": "skipped"
        }
    
    # Process the query
    result = await agent.research_query(query_text)
    
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
    
    # Return the processed row
    return {
        **query_data,
        "answer": result["answer"],
        "processing_time": result["processing_time"],
        "status": result["status"]
    }


async def run_batch_processing_async(agent: VLLMMCPResearchAgent, input_csv_path: str, 
                                   output_csv_path: str, max_concurrent: int = 3,
                                   output_dir: str = "outputs"):
    """Run batch processing with async agent."""
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found at '{input_csv_path}'")

    batch_start = time.time()
    logger.info("Starting CSV batch processing with vLLM + MCP + LangChain")
    logger.info(f"Processing CSV file: {input_csv_path}")
    logger.info(f"Output CSV: {output_csv_path}")
    logger.info(f"Max concurrent requests: {max_concurrent}")

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

        # Setup output CSV
        original_headers = df.columns.tolist()
        output_fieldnames = original_headers[:]
        if "answer" not in output_fieldnames:
            output_fieldnames.append("answer")
        if "processing_time" not in output_fieldnames:
            output_fieldnames.append("processing_time")
        if "status" not in output_fieldnames:
            output_fieldnames.append("status")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)

        # Process queries with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(query_data, idx):
            async with semaphore:
                return await process_single_query_async(agent, query_data, idx, output_dir)
        
        # Create tasks for all queries
        tasks = [
            bounded_process(input_row_dicts[i], i) 
            for i in range(total_queries)
        ]
        
        # Process with progress tracking
        results = []
        with tqdm(total=total_queries, desc="Processing queries") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # Log progress
                completed = len(results)
                if completed % 10 == 0 or completed == total_queries:
                    logger.info(f"Completed {completed}/{total_queries} queries")
        
        # Write results to CSV
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        processing_time = time.time() - batch_start
        logger.info(f"CSV batch processing completed successfully!")
        logger.info(f"Total time: {processing_time:.2f}s")
        if total_queries > 0:
            logger.info(f"Average time per query: {processing_time / total_queries:.2f}s")
        logger.info(f"Output saved to: {output_csv_path}")
        print(f"\nProcessing complete. Output saved to {output_csv_path}")

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


def run_batch_processing(agent: VLLMMCPResearchAgent, input_csv_path: str, 
                        output_csv_path: str, max_concurrent: int = 3,
                        output_dir: str = "outputs"):
    """Wrapper to run async batch processing from sync context."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
        # If we're already in an async context, create a new thread
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(
                    run_batch_processing_async(agent, input_csv_path, output_csv_path, 
                                             max_concurrent, output_dir)
                )
            except Exception as e:
                exception = e
            finally:
                new_loop.close()
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
            
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        asyncio.run(
            run_batch_processing_async(agent, input_csv_path, output_csv_path, 
                                     max_concurrent, output_dir)
        )


async def test_single_query(agent: VLLMMCPResearchAgent, test_query: str = None):
    """Test function for a single query."""
    logger.info("=== Testing Single Query ===")
    
    try:
        await agent.initialize()
        
        if not test_query:
            test_query = "What are the latest developments in AI safety research in 2024?"
        
        result = await agent.research_query(test_query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Status: {result['status']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Message count: {result['message_count']}")
        print(f"Model: {result['model_id']}")
        print(f"Temperature: {result['temperature']}")
        print(f"\nAnswer:\n{result['answer']}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
        raise
    finally:
        await agent.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="vLLM + MCP Research Agent for batch CSV processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python vllm_mcp_agent.py --csv queries.csv --model_id microsoft/Phi-3-mini-4k-instruct --base_url http://localhost:8000/v1

  # With custom parameters
  python vllm_mcp_agent.py --csv data.csv --model_id llama2-7b --base_url http://192.168.1.100:8000/v1 --temperature 0.3 --max_tokens 2048

  # With additional model parameters
  python vllm_mcp_agent.py --csv data.csv --model_id your-model --base_url http://localhost:8000/v1 --model_kwargs '{"top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.1}'

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
                       help="Additional model parameters as JSON string (e.g., '{\"top_p\": 0.9, \"frequency_penalty\": 0.1}')")
    
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
    args = parse_arguments()
    
    # Validate arguments
    if not args.test and not args.csv:
        print("Error: --csv is required for batch processing (or use --test for test mode)")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_dir, args.model_id)
    
    logger.info("=== Starting vLLM + MCP Research Agent ===")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Additional model kwargs: {args.model_kwargs}")
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
    
    # Create agent with parameters
    agent = VLLMMCPResearchAgent(
        model_id=args.model_id,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mcp_server_url=args.mcp_server_url,
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
            
            run_batch_processing(
                agent=agent,
                input_csv_path=input_csv_path,
                output_csv_path=output_csv_path,
                max_concurrent=args.max_concurrent,
                output_dir=args.output_dir
            )
            
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
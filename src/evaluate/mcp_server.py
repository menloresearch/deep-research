#!/usr/bin/env python3
"""
MCP Server Proxy using FastMCP
Proxies the Serper search and scrape MCP server via HTTP
"""

import os
import logging
from fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_mcp_proxy():
    """Create and configure the MCP proxy."""
    
    # Get Serper API key from environment
    # serper_api_key = os.getenv("SERPER_API_KEY")
    # if not serper_api_key:
    #     raise ValueError("SERPER_API_KEY environment variable is required")
    
    # logger.info(f"Configuring MCP proxy with Serper API key: {serper_api_key[:10]}...")
    
    # Create a proxy configuration
    config = {
        "mcpServers": {
            "serper-search": {
                "command": "npx",
                "args": ["-y", "serper-search-scrape-mcp-server"],
                "env": {
                    "SERPER_API_KEY": "e75ad9dd891de5d51e95a74abee61ccfd78d0653"
                }
            }
        }
    }
    
    # Create a proxy to the configured server
    proxy = FastMCP.as_proxy(config, name="Serper-Search-Proxy")
    
    logger.info("MCP proxy created successfully")
    return proxy


def main():
    """Main function to run the MCP proxy server."""
    try:
        logger.info("Starting MCP proxy server...")
        
        # Create the proxy
        proxy = create_mcp_proxy()
        
        # Run the proxy with HTTP transport
        logger.info("Running MCP proxy on http://127.0.0.1:8000/mcp")
        proxy.run(transport='streamable-http', host='127.0.0.1', port=2323)
        
    except KeyboardInterrupt:
        logger.info("MCP proxy server stopped by user")
    except Exception as e:
        logger.error(f"MCP proxy server failed: {e}")
        raise


if __name__ == "__main__":
    main()
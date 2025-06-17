---
layout: single
title: Welcome to Jan Nano Docs
---

# Welcome to Jan Nano Docs

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/menloresearch/deep-research)

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/65713d70f56f9538679e5a56/wC7Xtolp7HOFIdKTOJhVt.png" width="300" alt="Jan-Nano">
</div>

## Overview

Jan-Nano is a compact 4-billion parameter language model specifically designed and trained for deep research tasks. This model has been optimized to work seamlessly with Model Context Protocol (MCP) servers, enabling efficient integration with various research tools and data sources.

## Demo

![Jan-Nano Demo](assets/replay.gif)

Jan-Nano is supported by [Jan](https://github.com/menloresearch/jan), an open-source ChatGPT alternative that runs entirely on your computer. Jan provides a user-friendly interface for running local AI models with full privacy and control.

## System Requirements

- **Minimum Requirements**:
    - 8GB RAM (with iQ4_XS quantization)
    - 12GB VRAM (for Q8 quantization)
    - CUDA-compatible GPU

- **Recommended Setup**:
    - 16GB+ RAM
    - 16GB+ VRAM
    - Latest CUDA drivers
    - RTX 30/40 series or newer

## Quick Start

1. Install [Jan Beta](https://jan.ai/docs/desktop/beta)
2. Download Jan-Nano from Hugging Face:
   - [Model Repository](https://huggingface.co/Menlo/Jan-nano)
   - [GGUF Versions](https://huggingface.co/Menlo/Jan-nano-gguf)
3. Setup in Jan:
   - Go to Settings -> Model Provider -> llama.cpp
   - Add Jan-Nano model
   - Select appropriate quantization

## Setup Guidelines

### MCP Server (Serper) Setup

- Recommended: [Serper MCP Server](https://github.com/marcopesani/mcp-server-serper)
- Requirements: Node.js ≥ 18, Serper API key ([get your API key here](https://serper.dev/api-key))

### Using with Jan

![Jan Beta Guidelines](assets/jan-beta-usage.excalidraw.png)

1. Start the Serper MCP server as above.
2. In Jan, go to **Settings → MCP Servers**.
3. Add a new MCP server, set the command to:

   ```
   env SERPER_API_KEY=your_api_key_here npx -y serper-search-scrape-mcp-server
   ```

4. Save and ensure Jan can connect to the MCP server.

## Performance

Jan-Nano has been evaluated on the SimpleQA benchmark using our MCP-based benchmark methodology:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65713d70f56f9538679e5a56/sdRfF9FX5ApPow9gZ31No.png)

The evaluation was conducted using our MCP-based benchmark approach, which assesses the model's performance on SimpleQA tasks while leveraging its native MCP server integration capabilities. This methodology better reflects Jan-Nano's real-world performance as a tool-augmented research model.

## FAQ

- **What are the recommended GGUF quantizations?**
    - Q8 GGUF is recommended for best performance
    - iQ4_XS GGUF for very limited VRAM setups
    - Avoid Q4_0 and Q4_K_M as they show significant performance degradation

- **Can I run this on a laptop with 8GB RAM?**
    - Yes, but use the recommended quantizations (iQ4_XS)
    - Note that performance may be limited with Q4 quantizations

- **How much did the training cost?**
    - Training was done on internal A6000 clusters
    - Estimated cost on RunPod would be under $100 using H200
    - Hardware used:
        - 8xA6000 for training code
        - 4xA6000 for vllm server (inferencing)

- **What frontend should I use?**
    - Jan Beta (recommended) - Minimalistic and polished interface
    - Download link: <https://jan.ai/docs/desktop/beta>

- **Getting Jinja errors in LM Studio?**
    - Use Qwen3 template from other LM Studio compatible models
    - Disable "thinking" and add the required system prompt
    - Fix coming soon in future GGUF releases

- **Having model loading issues in Jan?**
    - Use latest beta version: Jan-beta_0.5.18-rc6-beta
    - Ensure proper CUDA support for your GPU
    - Check VRAM requirements match your quantization choice

## Resources

- [Jan-Nano Model on Hugging Face](https://huggingface.co/Menlo/Jan-nano)
- [Jan-Nano GGUF on Hugging Face](https://huggingface.co/Menlo/Jan-nano-gguf)

## Contact

- For support, questions, and community chat, join the [Menlo Discord Community](https://discord.com/invite/FTk2MvZwJH)

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.llms.vllm import VLLMOpenAI
from langchain.chat_models import init_chat_model
import pprint
async def main():

    from langchain.chat_models import init_chat_model

    # chat = init_chat_model(
    #     model="openai:jan-hq/Qwen3-4B-v0.4-deepresearch-no-think",
    #     base_url="http://127.0.0.1:8008/v1",
    #     api_key="ok",  # Required by langchain-openai even if not 
    #     top_k=20,
    #     top_p=0.8,
    # )
    chat = VLLMOpenAI(openai_api_base='http://127.0.0.1:8008/v1',openai_api_key="ok", top_k=20, top_p=0.8)

    client = MultiServerMCPClient(
        {
            "serper-search": {
                "url": "http://127.0.0.1:2323/mcp",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    # print(tools)
    agent = create_react_agent(chat, tools, prompt="")
    result = await agent.ainvoke({"messages": "Who is Alan Dao?"})
    pprint.pprint(result)
    # Extract and print only the final AI response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            print(msg.content)
            break

if __name__ == "__main__":
    asyncio.run(main())
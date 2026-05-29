import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEEKNEWS_SERVER_PATH = os.path.join(BASE_DIR, "GeekNews-MCP-Server", "main.py")


async def main():
    client = MultiServerMCPClient(
        {
            "geeknews": {
                "command": "python",
                "args": [GEEKNEWS_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    print("====== Available Tools List ======")
    for tool in tools:
        print("-", tool.name)


if __name__ == "__main__":
    asyncio.run(main())
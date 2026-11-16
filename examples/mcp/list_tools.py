"""Start the local MCP server as an assistant's app does, list its tools, read the budgets.
Needs: pip install "nalvera[mcp]"; once: nalvera.login(remember=True).
Spends nothing. Run: python list_tools.py
"""
import asyncio
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters

# The server gets this process's NALVERA_* settings; a key stored by
# nalvera.login(remember=True) is found by the server itself.
SETTINGS = {k: v for k, v in os.environ.items() if k.startswith("NALVERA_")}
SERVER = StdioServerParameters(command=sys.executable, args=["-m", "nalvera.mcp"],
                               env=SETTINGS)


async def main():
    async with Client(SERVER) as client:
        listed = await client.list_tools()
        for tool in listed.tools:
            print(f"{tool.name}: {tool.title}")
        budgets = await client.call_tool("get_budgets", {})
        print("free tasks left:", budgets.structured_content["free"]["free_tasks_left"])


asyncio.run(main())

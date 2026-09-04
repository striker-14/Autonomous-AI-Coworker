"""Tools for the Sidekick.

The Sidekick uses:
- Google Search
- Wikipedia
- Playwright MCP for browser automation
- Filesystem MCP for working with the sandbox
"""

import asyncio
from contextlib import AsyncExitStack

import wikipedia
from dotenv import load_dotenv

from langchain_community.tools import GoogleSerperRun, WikipediaQueryRun
from langchain_community.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


load_dotenv(override=True)


# ---------------------------------------------------------
# Google Search
# ---------------------------------------------------------

search = GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper()
)


# ---------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------

# Wikimedia requires a proper user agent.
wikipedia.set_user_agent(
    "sidekick-agent"
)

wikipedia_lookup = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


# ---------------------------------------------------------
# MCP Server Configuration
# ---------------------------------------------------------

def mcp_connections(sandbox: str) -> dict:
    """Return the MCP servers used by Sidekick."""

    return {
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "@playwright/mcp@latest",
                "--isolated",
            ],
        },

        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                sandbox,
            ],
        },
    }


# ---------------------------------------------------------
# Persistent MCP Sessions
# ---------------------------------------------------------

class McpSessions:
    """Keep MCP server sessions alive while Sidekick is running."""

    def __init__(self, connections: dict):
        self.connections = connections
        self.tools = []

        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task = None

    async def _run(self):
        """Start the MCP servers and keep them alive."""

        client = MultiServerMCPClient(
            self.connections
        )

        async with AsyncExitStack() as stack:

            for name in self.connections:

                session = await stack.enter_async_context(
                    client.session(name)
                )

                tools = await load_mcp_tools(
                    session,
                    server_name=name,
                )

                self.tools += tools

            # Tell start() that all servers are ready.
            self._ready.set()

            # Keep the sessions alive.
            await self._stop.wait()

    async def start(self) -> list:
        """Start the MCP servers and return their tools."""

        self._task = asyncio.create_task(
            self._run()
        )

        ready = asyncio.create_task(
            self._ready.wait()
        )

        await asyncio.wait(
            [ready, self._task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        ready.cancel()

        # If the background task finished unexpectedly,
        # re-raise the actual MCP startup error.
        if self._task.done():
            self._task.result()

        return self.tools

    def stop(self):
        """Stop the MCP servers."""

        self._stop.set()


# ---------------------------------------------------------
# Get All Sidekick Tools
# ---------------------------------------------------------

async def get_all_tools(sandbox: str):
    """Create the MCP sessions and return all Sidekick tools."""

    sessions = McpSessions(
        mcp_connections(sandbox)
    )

    # Start Playwright + Filesystem MCP servers.
    mcp_tools = await sessions.start()

    # Our normal LangChain tools.
    our_tools = [
        search,
        wikipedia_lookup,
    ]

    # Return everything to the Sidekick agent.
    return our_tools + mcp_tools, sessions


import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware


load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")


WORKER_PROMPT = """You are Sidekick, a capable personal assistant who completes tasks for the user.

You have access to:
- a real web browser
- a sandbox filesystem
- web search
- Wikipedia

Use your tools whenever they are useful for completing the user's request.

When using the browser, navigate to pages and read their content efficiently.

Work on the user's task until you have completed it.

When you are finished, give the user a clear final answer explaining:
- what you did
- what you found
- what you produced, if anything.
"""


class TolerateToolErrors(AgentMiddleware):
    """Allow the agent to recover from tool failures instead of crashing."""

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            return ToolMessage(
                content=f"That tool call failed: {error}. Try another approach.",
                tool_call_id=request.tool_call["id"],
            )


class Sidekick:

    def __init__(self):
        self.sidekick_id = str(uuid.uuid4())
        self.memory = InMemorySaver()
        self.tools = None
        self.sessions = None
        self.worker = None

    async def setup(self):
        os.makedirs(SANDBOX, exist_ok=True)

        self.tools, self.sessions = await get_all_tools(SANDBOX)

        self.worker = create_agent(
            model="google_genai:gemini-3.1-flash-lite",
            tools=self.tools,
            system_prompt=(
                f"{WORKER_PROMPT}\n"
                f"Today is {datetime.now():%A %d %B %Y}."
            ),
            middleware=[
                TolerateToolErrors(),
            ],
            checkpointer=self.memory,
        )

    async def run(self, message: str):
        config = {
            "configurable": {
                "thread_id": self.sidekick_id
            }
        }

        result = await self.worker.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            },
            config=config,
        )

        return result["messages"][-1].content

    def cleanup(self):
        if self.sessions:
            self.sessions.stop()

import asyncio


async def main():
    sidekick = Sidekick()

    try:
        print("Starting Sidekick...")
        await sidekick.setup()

        print("Sidekick is ready!")
        print("Type 'exit' to quit.\n")

        while True:
            message = input("You: ")

            if message.lower() in {"exit", "quit"}:
                break

            response = await sidekick.run(message)

            print(f"\nSidekick: {response}\n")

    finally:
        sidekick.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

"""MCP server (stdio) exposing get_context as a native agent tool."""

import threading
from contextlib import asynccontextmanager
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import core

ROOT = None


@asynccontextmanager
async def _warm_on_start(_server):
    # runs once the stdio transport has taken over stdout; a thread so the agent's handshake isn't held up.
    # The first query then waits only for whatever is left, instead of for the model load and index update.
    threading.Thread(target=core.warm, args=(ROOT,), daemon=True).start()
    yield {}


mcp = MCPServer(
    "gus",
    instructions="Gus finds the code relevant to a question in this project. Call get_context to locate code before "
    "reading whole files; open files when you need the surrounding code, and use grep when you need every occurrence.",
    lifespan=_warm_on_start,
)


# read-only for the agent: it only refreshes Gus's own cache in .gus/, never the project's files.
# stray prints can't corrupt the JSON-RPC stream: the SDK's stdio transport diverts fd 1 to stderr while serving.
@mcp.tool(annotations=ToolAnnotations(title="Get relevant context", read_only_hint=True, open_world_hint=False))
def get_context(query: str, top_k: int = 5, detail: Literal["full", "outline"] = "full") -> list[dict]:
    """Find the code, docs and PDF passages in this project that answer a question.

    Returns up to top_k chunks, best first: {path, line_start, line_end, snippet, score, kind}. Use it to locate code
    instead of reading whole files, then open the file at that range when you need what's around it (callers,
    imports, the rest of the class).

    Asking well:
    - Use exact names (functions, classes, config keys) when you know them: they match most reliably.
    - Ask about one thing per call; don't pack several phrasings into one query.
    - If the results mention a likely function or class, search again by that name.

    Limits: these are the best matches, not every occurrence, so use grep to find all usages. If the snippets don't
    clearly answer the question, read the file or grep before concluding the code doesn't exist. Not indexed:
    git-ignored files, files over 1 MB (PDFs over 25 MB), binaries, symlinks, and the contents of images.

    detail="full" (default) returns each chunk's code. detail="outline" (~3x smaller) returns signatures and matching
    lines only: use it just to locate something, then read the chosen range.
    """
    return core.get_context(query, top_k, ROOT, detail)


def serve(root):
    global ROOT
    ROOT = root
    mcp.run("stdio")

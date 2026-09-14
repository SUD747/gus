"""`gus init`: register `gus serve` with each agent's MCP config and add a fallback note to convention files."""

import json
import re
import shutil
import sys
from pathlib import Path

MARKER = "<!-- gus -->"
# Antigravity CLI loads MCP servers only from its global config (its project-level .agents/mcp_config.json is ignored as
# of agy 1.2.2), and starts them in the project directory, so one path-less `gus serve` entry serves every project
GEMINI_DIR = Path.home() / ".gemini"
NOTE = f"""
{MARKER}
Use the `get_context` tool (Gus) to find the code relevant to a question
before opening whole files. Use exact names when you know them, and search
again with names from the first results. Open the file when you need the
surrounding code, and use grep to find every usage. If Gus isn't available
as a tool, run `gus search "<query>"` from the shell instead.
"""


def server_entry(root: Path | None) -> dict:
    """root=None: `gus serve` finds the project from the working directory the agent starts it in."""
    serve = ["serve", str(root)] if root else ["serve"]
    # agents spawn MCP servers with their own PATH, so pin an absolute command
    exe = shutil.which("gus")
    if exe:
        return {"command": exe, "args": serve}
    return {"command": sys.executable, "args": ["-m", "gus", *serve]}


def _merge_json(path: Path, entry: dict) -> str:
    data = json.loads(path.read_text() or "{}") if path.exists() else {}  # invalid JSON raises: never clobber
    data.setdefault("mcpServers", {})["gus"] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path)


def _merge_codex(path: Path, entry: dict) -> str:
    text = path.read_text() if path.exists() else ""
    # json.dumps output is a valid TOML basic string / array
    block = f"[mcp_servers.gus]\ncommand = {json.dumps(entry['command'])}\nargs = {json.dumps(entry['args'])}\n"
    # replace our previous block (up to the next table header) so a moved gus install doesn't leave a dead entry
    text, n = re.subn(r"^\[mcp_servers\.gus\]\n(?:(?!\[).*\n?)*", lambda _: block, text, flags=re.M)
    if not n:
        text += ("\n" if text else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def init(root: Path, all_agents: bool = False) -> list[str]:
    """Wire Gus into the project. Returns human-readable lines describing what was done."""
    entry = server_entry(root)
    done = [f"MCP  {_merge_json(root / '.mcp.json', entry)}  (Claude Code)"]
    for d, name, agent in ((".cursor", "mcp.json", "Cursor"), (".agents", "mcp_config.json", "Antigravity IDE")):
        if all_agents or (root / d).is_dir():
            done.append(f"MCP  {_merge_json(root / d / name, entry)}  ({agent})")
    if all_agents or (GEMINI_DIR / "antigravity-cli").is_dir():  # not just ~/.gemini: Gemini CLI uses it too
        done.append(f"MCP  {_merge_json(GEMINI_DIR / 'config' / 'mcp_config.json', server_entry(None))}  (Antigravity CLI, all projects)")
    if all_agents or (root / ".codex").is_dir():
        done.append(f"MCP  {_merge_codex(root / '.codex' / 'config.toml', entry)}  (Codex — project must be trusted in ~/.codex/config.toml)")

    for name in ("CLAUDE.md", "AGENTS.md", ".cursorrules"):
        f = root / name
        if f.is_file() and MARKER not in f.read_text():
            f.write_text(f.read_text().rstrip("\n") + "\n" + NOTE)
            done.append(f"note {f}")
    rules = root / ".cursor" / "rules"
    if rules.is_dir() and not (rules / "gus.mdc").exists():
        (rules / "gus.mdc").write_text("---\ndescription: Gus context retrieval\nalwaysApply: true\n---" + NOTE)
        done.append(f"note {rules / 'gus.mdc'}")
    return done

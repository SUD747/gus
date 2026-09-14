"""End-to-end check of get_context on a tiny fixture project. Run: python tests/test_gus.py (or pytest)."""

import json
import os
import tempfile
from pathlib import Path

from gus import agents, chunker, core, embed, index

agents.GEMINI_DIR = Path(tempfile.mkdtemp()) / ".gemini"  # never touch the real global Antigravity config

FILES = {
    "src/auth/session.py": '''import time


class SessionStore:
    """Keeps user sessions in memory."""

    def __init__(self):
        self.sessions = {}

    def refresh_token(self, session_id):
        """Renew an expired access token for the session."""
        s = self.sessions[session_id]
        s["token"] = s["token"] + "-renewed"
        s["expires"] = time.time() + 3600
        return s["token"]
''',
    "src/billing/invoice.ts": """// Invoice helpers
export function computeInvoiceTotal(items: {price: number, qty: number}[]): number {
  let total = 0;
  for (const it of items) total += it.price * it.qty;
  return total;
}
""",
    "docs/guide.md": """# Guide

Setup notes.

## Deploying

Push to main and the pipeline ships it.

![diagram](arch.png)

```bash
# not a heading
deploy.sh
```
""",
    "tests/test_session.py": "def test_refresh_token():\n    assert refresh_token('x')\n    assert refresh_token('y')\n    assert refresh_token('z')\n",
}


def make_project(d: Path):
    for rel, text in FILES.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
        root = Path(tmp)
        make_project(root)
        (Path(outside) / "secret.py").write_text("def leaked_credentials():\n    return 'x'\n")
        (root / "src/link.py").symlink_to(Path(outside) / "secret.py")  # must not be indexed

        top = core.get_context("renew expired access token", 3, root)
        assert top[0]["path"] == "src/auth/session.py", top
        assert top[0]["line_start"] <= 10 <= top[0]["line_end"], top[0]
        assert set(top[0]) >= {"path", "line_start", "line_end", "snippet", "score"}

        assert core.get_context("computeInvoiceTotal", 1, root)[0]["path"] == "src/billing/invoice.ts"
        brief = core.get_context("renew expired access token", 1, root, detail="outline")[0]
        # the whole small class is one chunk: the outline must still surface the method that matched
        assert brief["snippet"].startswith("class SessionStore:") and "def refresh_token(self, session_id):" in brief["snippet"], brief
        assert (brief["line_start"], brief["line_end"]) == (top[0]["line_start"], top[0]["line_end"])
        assert core.get_context("session store", 1, root)[0]["snippet"].startswith("class SessionStore")

        doc = core.get_context("how do we deploy", 1, root)[0]
        assert doc["path"] == "docs/guide.md" and "image(s) not indexed" in doc["note"], doc
        assert "# not a heading" in doc["snippet"]  # fenced comment didn't split the section

        # incremental: an edit shows up in the next query without a manual reindex
        (root / "src/billing/invoice.ts").write_text("export function applyDiscountCoupon() {}\n")
        assert core.get_context("applyDiscountCoupon", 1, root)[0]["path"] == "src/billing/invoice.ts"

        from PIL import Image  # pdfplumber's own dependency

        Image.new("RGB", (100, 100), "white").save(root / "docs/scan.pdf")  # a scanned page: images, no text
        scan = core.get_context("scanned page", 1, root)[0]
        assert scan["path"] == "docs/scan.pdf" and scan["page"] == 1 and "image(s) not indexed" in scan["note"], scan
        assert index.stats(root)["files"] == len(FILES) + 1  # the symlink was skipped
        assert (root / ".gus/.gitignore").read_text() == "*\n"

        assert len(core.get_context("token", 0, root)) == 1  # top_k is clamped, not sliced from the end
        try:
            core.get_context("   ", 5, root)
            raise AssertionError("empty query accepted")
        except ValueError:
            pass


def test_chunking_splits_big_class():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "big.py"
        methods = "\n".join(f"    def m{i}(self):\n" + "        x = 1\n" * 8 for i in range(12))
        f.write_text("class Big:\n" + methods)
        chunks = chunker.chunk_file(f, "big.py")
        assert len(chunks) > 1 and all(c.line_end - c.line_start < chunker.MAX_LINES for c in chunks)
        assert chunks[0].text.startswith("class Big:")  # header kept with first member
        assert chunks[1].context == "class Big:"

        # a long method after another member: its signature line holds several nodes (name, parameters, return type).
        # It must stay one construct: no one-line signature chunk and no overlapping chunks.
        ts = Path(tmp) / "long.ts"
        first = "".join(f"    a += {i};\n" for i in range(8))
        body = "".join(f"    if (x > {i}) {{\n      y += {i};\n    }}\n" for i in range(30))
        ts.write_text(f"class K {{\n  small(a: number): number {{\n{first}    return a;\n  }}\n\n"
                      f"  async run(x: number): Promise<number> {{\n    let y = 0;\n{body}    return y;\n  }}\n}}\n")
        chunks = chunker.chunk_file(ts, "long.ts")
        spans = [(c.line_start, c.line_end) for c in chunks]
        assert all(b[0] > a[1] for a, b in zip(spans, spans[1:], strict=False)), spans
        assert (14, 14) not in spans and any(s == 14 for s, _ in spans), spans


def test_index_keeps_its_model():
    """Built with GUS_MODEL=X, a later process without the variable (e.g. an agent's `gus serve`) must reuse X."""
    other = "minishlab/potion-base-2M"  # any second model; the smallest one keeps the test download tiny
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)
        saved = os.environ.get("GUS_MODEL")
        try:
            os.environ["GUS_MODEL"] = other
            embed.use(other)
            core.get_context("renew token", 1, root)
            chunk_ids = index.connect(root).execute("SELECT group_concat(id) FROM chunks").fetchone()

            del os.environ["GUS_MODEL"]
            embed.use(embed.DEFAULT_MODEL)  # what a fresh process without the variable starts with
            core.get_context("renew token", 1, root)
            assert index.stats(root)["model"] == other and embed.MODEL == other
            assert index.connect(root).execute("SELECT group_concat(id) FROM chunks").fetchone() == chunk_ids  # no rebuild

            os.environ["GUS_MODEL"] = embed.DEFAULT_MODEL  # explicitly asking for another model does rebuild
            core.get_context("renew token", 1, root)
            assert index.stats(root)["model"] == embed.DEFAULT_MODEL
        finally:
            os.environ.pop("GUS_MODEL", None) if saved is None else os.environ.update(GUS_MODEL=saved)
            embed.use(saved or embed.DEFAULT_MODEL)


def test_setup_choice():
    from gus import cli

    answers = iter(["2", "2"])
    assert cli.ask_model_and_device(embed.DEFAULT_MODEL, "cpu", ask=lambda _: next(answers)) == (embed.QUALITY_MODEL, "gpu")
    assert cli.ask_model_and_device(embed.QUALITY_MODEL, "gpu", ask=lambda _: "") == (embed.QUALITY_MODEL, "gpu")  # Enter keeps current
    assert cli.ask_model_and_device(embed.DEFAULT_MODEL, "gpu", ask=lambda _: "1") == (embed.DEFAULT_MODEL, "cpu")  # fast model: no device question
    assert embed.missing_runtime("fast", "gpu") is None  # the static model needs no extra runtime anywhere

    with tempfile.TemporaryDirectory() as tmp:  # non-interactive init: flags are saved and survive a flag-less re-run
        root = Path(tmp)
        make_project(root)
        saved = {k: os.environ.get(k) for k in ("GUS_MODEL", "GUS_DEVICE")}
        try:
            cli.main(["init", str(root), "--model", "minishlab/potion-base-2M"])
            assert index.load_config(root) == {"model": "minishlab/potion-base-2M", "device": "cpu"}
            for k in saved:
                os.environ.pop(k, None)
            cli.main(["init", str(root)])
            assert index.load_config(root)["model"] == "minishlab/potion-base-2M" and index.stats(root)["model"] == "minishlab/potion-base-2M"
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
            embed.use(embed.DEFAULT_MODEL)


def test_mcp_server():
    """The real stdio server: warms up at start, answers get_context, and describes how to query it."""
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call(root):
        params = StdioServerParameters(command=sys.executable, args=["-m", "gus", "serve", str(root)])
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tool = (await s.list_tools()).tools[0]
            res = await s.call_tool("get_context", {"query": "renew expired access token", "top_k": 1})
            return tool, res

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)
        tool, res = asyncio.run(call(root))
        assert tool.annotations.read_only_hint and "exact names" in tool.description and "grep" in tool.description
        assert not res.is_error and res.structured_content["result"][0]["path"] == "src/auth/session.py", res


def test_cli_help_and_version():
    import contextlib
    import io

    from gus import cli

    for argv, expect in ((["help"], "<command>"), ([], "<command>"), (["help", "search"], "--top-k")):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(argv)
        assert expect in out.getvalue(), (argv, out.getvalue())
    for flag in ("-v", "--version"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            cli.main([flag])
        assert out.getvalue().startswith("gus "), out.getvalue()


def test_init_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "CLAUDE.md").write_text("# Project\n")
        (root / ".mcp.json").write_text('{"mcpServers": {"other": {"command": "x"}}}')
        (root / ".codex").mkdir()
        (root / ".codex/config.toml").write_text('[mcp_servers.gus]\ncommand = "/old/python"\nargs = []\n\n[other]\nk = 1\n')
        (agents.GEMINI_DIR / "config").mkdir(parents=True, exist_ok=True)
        (agents.GEMINI_DIR / "config/mcp_config.json").write_text('{"mcpServers": {"other": {"serverUrl": "https://example.com/mcp"}}}')
        for _ in range(2):
            agents.init(root, all_agents=True)
        codex = (root / ".codex/config.toml").read_text()
        assert "/old/python" not in codex and "[other]\nk = 1" in codex, codex
        cfg = json.loads((root / ".mcp.json").read_text())["mcpServers"]
        assert set(cfg) == {"other", "gus"} and cfg["gus"]["args"][-1] == str(root)
        assert (root / "CLAUDE.md").read_text().count(agents.MARKER) == 1
        assert (root / ".codex/config.toml").read_text().count("[mcp_servers.gus]") == 1
        assert "gus" in json.loads((root / ".agents/mcp_config.json").read_text())["mcpServers"]
        agy = json.loads((agents.GEMINI_DIR / "config/mcp_config.json").read_text())["mcpServers"]
        assert set(agy) == {"other", "gus"} and agy["gus"]["args"][-1] == "serve"  # path-less: Antigravity starts it in whichever project is open


if __name__ == "__main__":
    test_end_to_end()
    test_chunking_splits_big_class()
    test_index_keeps_its_model()
    test_setup_choice()
    test_init_is_idempotent()
    test_cli_help_and_version()
    test_mcp_server()
    print("ok")

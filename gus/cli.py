import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import core, embed, index


def main(argv=None):
    p = argparse.ArgumentParser(prog="gus", description="Relevant-context retrieval for AI coding agents.")
    p.add_argument("-v", "--version", action="version", version=f"gus {_version()}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    s = sub.add_parser("search", help="retrieve the chunks most relevant to a query")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--format", choices=["text", "json"], default="text")
    s.add_argument("--detail", choices=["full", "outline"], default="full", help="outline: doc line, signature, nested signatures and query-matching lines per hit (~3x smaller)")
    s.add_argument("--root", help="project root (default: nearest indexed/git ancestor of cwd)")

    for name, help_ in (("index", "build or update the index"), ("stats", "show index size and freshness"),
                        ("serve", "run the MCP server over stdio"), ("init", "wire Gus into agent configs")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("path", nargs="?", help="project directory (default: the project containing the current directory)")
    sub.choices["init"].add_argument("--all", action="store_true", help="also write Cursor/Codex/Antigravity configs even if those dirs don't exist")
    sub.choices["init"].add_argument("--model", help="fast (default), quality, or a model id; asked interactively when omitted in a terminal")
    sub.choices["init"].add_argument("--device", choices=["cpu", "gpu"], help="where the quality model runs (default cpu)")
    sub.add_parser("help", help="show this help, or a command's help: gus help search").add_argument("command", nargs="?")

    a = p.parse_args(argv)
    if a.cmd in (None, "help"):  # bare `gus` prints help instead of an argparse error
        name = getattr(a, "command", None)
        if name and name not in sub.choices:
            p.error(f"unknown command {name!r}")
        (sub.choices[name] if name else p).print_help()
        return
    root = Path(a.path).resolve() if getattr(a, "path", None) else index.find_root(getattr(a, "root", None))

    if a.cmd == "search":
        try:
            results = core.get_context(a.query, a.top_k, root, a.detail)
        except ValueError as e:
            sys.exit(f"gus: {e}")
        print(json.dumps(results, indent=2) if a.format == "json" else core.format_text(results))
    elif a.cmd == "index":
        r = index.build(root)
        index.log(f"{r['files']} files ({r['changed']} updated, {r['removed']} removed)")
    elif a.cmd == "stats":
        if not index.exists(root):
            print(f"no index at {root} — run `gus index`")
            return
        st = index.stats(root)
        ago = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st['last_indexed']))} ({_ago(st['last_indexed'])})" if st["last_indexed"] else "never"
        print(f"root          {st['root']}\nfiles         {st['files']}\nchunks        {st['chunks']}\n"
              f"index size    {st['index_bytes'] / 1e6:.1f} MB\nlast indexed  {ago}\nmodel         {st['model']}\ndevice        {st['device']}")
    elif a.cmd == "serve":
        from . import server

        server.serve(root)
    elif a.cmd == "init":
        from . import agents

        current = index.load_config(root)
        model, device = a.model, a.device
        if model is None and sys.stdin.isatty():
            model, device = ask_model_and_device(current.get("model", embed.DEFAULT_MODEL), device or current.get("device", "cpu"))
        model = embed.resolve(model or current.get("model", embed.DEFAULT_MODEL))
        device = device or current.get("device", "cpu")
        if problem := embed.missing_runtime(model, device):
            sys.exit(f"gus: {problem}, then run `gus init` again")
        index.save_config(root, model=model, device=device)
        os.environ.update(GUS_MODEL=model, GUS_DEVICE=device)  # this run's choice beats any value exported in the shell

        for line in agents.init(root, a.all):
            print(line)
        index.build(root)  # build now (and fetch the model) so the agent's first call is fast
        print(f"done ({model} on {device}) — restart your agent so it picks up the `gus` MCP server")


def ask_model_and_device(model: str, device: str, ask=input) -> tuple[str, str]:
    """Interactive setup choice. Defaults are the project's current choice (fast / cpu for a new project)."""
    print(
        "Embedding model for this project:\n"
        "  1) fast     potion-base-8M       30 MB download, indexes even large repos in seconds on any CPU\n"
        "  2) quality  EmbeddingGemma-300M  1.2 GB download, needs gus-mcp[onnx] (CPU) or [gpu] (NVIDIA);\n"
        "              puts the right code first 70% of the time vs 45% (see README, Embedding models)"
    )
    default = "2" if model == embed.QUALITY_MODEL else "1"
    model = embed.QUALITY_MODEL if (ask(f"Choose [{default}]: ").strip() or default) == "2" else embed.DEFAULT_MODEL
    if model != embed.QUALITY_MODEL:
        return model, "cpu"
    print(
        "Run the quality model on:\n"
        "  1) cpu  works everywhere; ~12 chunks/s, so a 20,000-chunk repo takes ~30 min to index the first time\n"
        "  2) gpu  NVIDIA CUDA via gus-mcp[gpu]; ~190 chunks/s on a laptop RTX 4050"
    )
    default = "2" if device == "gpu" else "1"
    return model, "gpu" if (ask(f"Choose [{default}]: ").strip() or default) == "2" else "cpu"


def _version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("gus-mcp")
    except PackageNotFoundError:  # running from a source checkout without installing
        return "unknown"


def _ago(ts):
    s = int(time.time() - ts)
    return f"{s}s ago" if s < 120 else f"{s // 60}m ago" if s < 7200 else f"{s // 3600}h ago" if s < 172800 else f"{s // 86400}d ago"

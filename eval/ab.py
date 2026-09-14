"""A/B benchmark: does a coding agent find the right code better, or with fewer tokens, when it has Gus?

Every eval question (cases.json) becomes a task for Claude Code in headless mode, run in two arms that differ only
in whether Gus is available:

    off   Read, Grep, Glob
    on    Read, Grep, Glob + Gus's get_context over MCP

Each run gets a fresh copy of the pinned repo, no user or project settings, and no other MCP servers. The agent ends
its reply with `ANSWER: path:start-end` lines, graded against the same hand-checked targets as run.py. The report
compares correctness, tokens, turns, time and cost per arm, and pairs the arms per question.

This spends API usage (see --max-budget-usd) and is not part of CI.

    python eval/ab.py --limit 4                      # pilot: 4 test questions, haiku
    python eval/ab.py --split test --model sonnet     # full run; results go to ~/.cache/gus/ab/<time>/
    python eval/ab.py --out ~/.cache/gus/ab/<time>    # resume an interrupted run (finished runs are skipped)
    python eval/ab.py --report ~/.cache/gus/ab/<time> # print the report again
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402

from gus import chunker, index  # noqa: E402

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gus"
TOOLS = ["Read", "Grep", "Glob"]
GUS_TOOL = "mcp__gus__get_context"
MAX_RANGE = 150  # an answer spanning more lines than this is too vague to count as locating the code
PROMPT = """You are working in the {repo} repository (the current directory). Question: {q}

Find the code in this repository that answers the question. You can't edit files. When you're done, end your reply
with one line per location, most relevant first:
ANSWER: <path relative to the repository root>:<first line>-<last line>"""
NOTE = (
    "Use the `get_context` tool (Gus) to find the code relevant to a question before opening whole files. Use exact "
    "names when you know them, and search again with names from the first results. Open the file when you need the "
    "surrounding code, and use grep to find every usage."
)
_write = threading.Lock()


def case_id(case):
    return f"{case['repo']}-{zlib.crc32(case['q'].encode()):08x}"


def template(repo, src, arm, a):
    """A clean copy of the pinned checkout per arm (the on arm with a prebuilt index), copied again for every run."""
    dest = Path(a.out) / "templates" / f"{repo}-{arm}"
    if dest.exists():
        return dest
    shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns(index.GUS_DIR))
    if arm == "on":
        os.environ.update(GUS_MODEL=a.gus_model, GUS_DEVICE=a.gus_device)
        index.build(dest)  # copies keep mtimes (copy2), so runs reuse this index instead of re-embedding
    return dest


def parse_answers(text, root):
    answers = []
    for m in re.finditer(r"^\W*ANSWER:\s*`?([^`\s:]+):(\d+)(?:\s*-\s*(\d+))?", text or "", re.M):
        path = m.group(1).removeprefix("./")
        if path.startswith(str(root)):
            path = os.path.relpath(path, root)
        start = int(m.group(2))
        answers.append((path, start, int(m.group(3) or start)))
    return answers


ENCLOSING = re.compile(r"function|method|class|impl|trait|interface|struct|enum|decorated|arrow")


def target_spans(root, case):
    """Each target as (path, first line, last line) of the construct it points at.

    An agent may point at the relevant lines inside a function rather than at its signature, and Gus returns whole
    definitions; grading by the whole construct keeps the two arms on equal terms.
    """
    spans = []
    for path, line in (run.resolve(root, t) for t in case["targets"]):
        spans.append((path, *construct_span(root / path, line)) if line else (path, None, None))
    return spans


def construct_span(file, line):
    """A target on a line that starts a construct (def, class, decorator, const = () => {...}) spans that construct;
    a target inside one (a statement in a function) spans the innermost enclosing function or class."""
    ext = file.suffix.lower()
    if ext in chunker.GRAMMARS:
        text = file.read_text(errors="replace")
        row, lines = line - 1, text.splitlines()
        src = lines[row] if row < len(lines) else ""
        indent = len(src) - len(src.lstrip())  # an indented method starts after its indent, not at column 0
        node = chunker._parser(ext).parse(text.encode()).root_node.named_descendant_for_point_range((row, indent), (row, len(src)))
        if node is not None and node.start_point[0] == row:
            while node.parent is not None and node.parent.parent is not None and node.parent.start_point[0] == row:
                node = node.parent
            if node.end_point[0] > row:
                return row + 1, node.end_point[0] + 1
        while node is not None and node.parent is not None:
            if ENCLOSING.search(node.type):
                return node.start_point[0] + 1, node.end_point[0] + 1
            node = node.parent
    inside = [c for c in chunker.chunk_file(file, file.name) if c.line_start <= line <= c.line_end]
    return (inside[0].line_start, inside[0].line_end) if inside else (line, line)


def grade(answers, spans):
    if not answers:
        return "no_answer", None

    def hits(ans):
        path, start, end = ans
        return end - start < MAX_RANGE and any(
            path == p and (lo is None or (start <= hi and end >= lo)) for p, lo, hi in spans  # ranges overlap
        )

    rank = next((i for i, ans in enumerate(answers, 1) if hits(ans)), None)
    return ("correct" if rank else "wrong"), rank


def run_one(case, arm, rep, templates, a):
    repo, cid = case["repo"], case_id(case)
    work = Path(a.out) / "work" / f"{cid}-{arm}-{rep}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(templates[repo][arm], work, symlinks=True)
    cfg = work.parent / f"{work.name}.mcp.json"
    servers = {"gus": {"command": sys.executable, "args": ["-m", "gus", "serve", str(work)]}} if arm == "on" else {}
    cfg.write_text(json.dumps({"mcpServers": servers}))
    allowed = TOOLS + ([GUS_TOOL] if arm == "on" else [])
    cmd = [
        "claude", "-p", PROMPT.format(repo=repo, q=case["q"]), "--model", a.model,
        "--setting-sources", "", "--strict-mcp-config", "--mcp-config", str(cfg),
        "--tools", ",".join(TOOLS), "--allowedTools", *allowed, "--no-session-persistence",
        "--max-budget-usd", str(a.max_budget_usd), "--output-format", "stream-json", "--verbose",
    ]
    if arm == "on" and a.note:
        cmd += ["--append-system-prompt", NOTE]
    started = time.time()
    try:
        out = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=a.timeout).stdout
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out, timed_out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), True
    seconds = time.time() - started

    tools, final = {}, {}
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for item in ev.get("message", {}).get("content", []):
                if item.get("type") == "tool_use":
                    tools[item["name"]] = tools.get(item["name"], 0) + 1
        elif ev.get("type") == "result":
            final = ev
    usage = final.get("usage") or {}
    answers = parse_answers(final.get("result"), work)
    status, rank = grade(answers, target_spans(work, case))
    failed = timed_out or bool(final.get("is_error")) or not final
    if failed:
        status = "error"
    row = {
        "case": cid, "repo": repo, "q": case["q"], "split": run.split_of(case), "arm": arm, "rep": rep,
        "model": a.model, "status": status, "rank": rank, "answers": answers, "failed": failed,
        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0), "turns": final.get("num_turns"), "seconds": round(seconds, 1),
        "cost_usd": final.get("total_cost_usd"), "tools": tools, "subtype": final.get("subtype"),
    }
    logs = Path(a.out) / "logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{cid}-{arm}-{rep}.jsonl").write_text(out)
    with _write, open(Path(a.out) / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    cfg.unlink(missing_ok=True)
    print(f"  {arm:<3} {status:<9} {row['input_tokens']:>7} tok {row['turns'] or 0:>3} turns {seconds:>5.0f}s  {case['q'][:60]}", flush=True)
    return row


def load_cases(public):
    spec = json.loads((HERE / "cases.json").read_text())
    local = HERE / "cases.local.json"
    if local.exists() and not public:
        extra = json.loads(local.read_text())
        spec["repos"] |= extra["repos"]
        spec["cases"] += extra["cases"]
    return spec


def report(out):
    rows = [json.loads(line) for line in (Path(out) / "runs.jsonl").read_text().splitlines() if line.strip()]
    # grade again from the stored answers, so a grading fix never needs new (paid) runs
    cases = {case_id(c): c for c in load_cases(public=False)["cases"]}
    spans = {}
    for r in rows:
        if r.get("failed") or r["case"] not in cases:
            continue
        if r["case"] not in spans:
            spans[r["case"]] = target_spans(Path(out) / "templates" / f"{r['repo']}-off", cases[r["case"]])
        r["status"], r["rank"] = grade([tuple(x) for x in r["answers"]], spans[r["case"]])
    print(f"\n{len(rows)} runs, model {', '.join(sorted({r['model'] for r in rows}))}\n")
    med = lambda xs: statistics.median(xs) if xs else 0
    print(f"{'arm':<4} {'n':>3} {'correct':>8} {'first':>6} {'wrong':>6} {'no ans':>7} {'error':>6}  "
          f"{'tokens (median / mean)':>23} {'turns':>6} {'secs':>5} {'cost $':>7}  tool calls per run")
    for arm in ("off", "on"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        status = {s: sum(r["status"] == s for r in rs) / n for s in ("correct", "wrong", "no_answer", "error")}
        tok = [r["input_tokens"] + r["output_tokens"] for r in rs]
        calls = {}
        for r in rs:
            for name, c in r["tools"].items():
                calls[name] = calls.get(name, 0) + c
        used = sum(GUS_TOOL in r["tools"] for r in rs)
        print(f"{arm:<4} {n:>3} {status['correct']:>8.0%} {sum(r['rank'] == 1 for r in rs) / n:>6.0%} {status['wrong']:>6.0%} "
              f"{status['no_answer']:>7.0%} {status['error']:>6.0%}  {med(tok):>11,.0f} / {statistics.mean(tok):>9,.0f} "
              f"{med([r['turns'] or 0 for r in rs]):>6.0f} {med([r['seconds'] for r in rs]):>5.0f} "
              f"{sum(r['cost_usd'] or 0 for r in rs):>7.2f}  "
              + ", ".join(f"{k.removeprefix('mcp__gus__')} {v / n:.1f}" for k, v in sorted(calls.items()))
              + (f"  (Gus used in {used}/{n} runs)" if arm == "on" else ""))

    # paired by question (and repeat): the same task with and without Gus
    by = {}
    for r in rows:
        by.setdefault((r["case"], r["rep"]), {})[r["arm"]] = r
    pairs = [p for p in by.values() if len(p) == 2]
    if not pairs:
        return
    ok = lambda r: r["status"] == "correct"
    only_on = sum(ok(p["on"]) and not ok(p["off"]) for p in pairs)
    only_off = sum(ok(p["off"]) and not ok(p["on"]) for p in pairs)
    both = [p for p in pairs if ok(p["on"]) and ok(p["off"])]
    # exact two-sided sign test on the questions where exactly one arm was right
    k, m = min(only_on, only_off), only_on + only_off
    p_value = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    print(f"\npaired: {len(pairs)} tasks · both right {len(both)} · only with Gus {only_on} · only without {only_off} · "
          f"neither {len(pairs) - len(both) - only_on - only_off} · sign test p={p_value:.2f}")
    if both:
        total = lambda r: r["input_tokens"] + r["output_tokens"]
        ratios = [total(p["on"]) / max(total(p["off"]), 1) for p in both]
        print(f"when both arms were right: tokens with Gus / without, median {med(ratios):.2f} "
              f"(Gus cheaper in {sum(x < 1 for x in ratios)}/{len(ratios)}); "
              f"turns median {med([p['on']['turns'] or 0 for p in both]):.0f} vs {med([p['off']['turns'] or 0 for p in both]):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--model", default="haiku", help="Claude model for the agent (default haiku)")
    ap.add_argument("--limit", type=int, help="only the first N questions (a pilot)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per question per arm (agents are nondeterministic)")
    ap.add_argument("--jobs", type=int, default=3, help="runs in parallel (timings get noisier with more)")
    ap.add_argument("--gus-model", default="fast", help="Gus embedding model for the on arm (fast, quality, or an id)")
    ap.add_argument("--gus-device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--note", action="store_true", help="also give the on arm the `gus init` usage note as a system prompt")
    ap.add_argument("--public", action="store_true", help="ignore cases.local.json")
    ap.add_argument("--max-budget-usd", type=float, default=1.0, help="per-run spending cap passed to claude")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per run")
    ap.add_argument("--out", help="results directory (default ~/.cache/gus/ab/<time>); reusing one resumes it")
    ap.add_argument("--report", metavar="DIR", help="print the report for an earlier run and exit")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    if not shutil.which("claude"):
        sys.exit("ab.py needs Claude Code's `claude` command on PATH")

    a.out = str(Path(a.out or CACHE / "ab" / time.strftime("%Y%m%d-%H%M%S")).expanduser())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    spec = load_cases(a.public)
    cases = [c for c in spec["cases"] if a.split in ("all", run.split_of(c))][: a.limit]
    repos_dir = CACHE / "eval-repos"
    repos_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: run.checkout(name, spec["repos"][name], repos_dir) for name in {c["repo"] for c in cases}}
    cases = [c for c in cases if sources[c["repo"]]]
    templates = {repo: {arm: template(repo, src, arm, a) for arm in ("off", "on")} for repo, src in sources.items()}

    done = set()
    runs_file = Path(a.out) / "runs.jsonl"
    if runs_file.exists():
        done = {(r["case"], r["arm"], r["rep"]) for r in map(json.loads, runs_file.read_text().splitlines()) if r}
    todo = [(c, arm, rep) for c in cases for rep in range(a.repeats) for arm in ("off", "on") if (case_id(c), arm, rep) not in done]
    (Path(a.out) / "config.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "report"}, indent=2))
    print(f"{len(cases)} questions × 2 arms × {a.repeats} = {len(todo)} runs to go, model {a.model}, results in {a.out}")
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda t: run_one(*t, templates, a), todo))
    report(a.out)


if __name__ == "__main__":
    main()

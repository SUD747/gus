"""Retrieval eval: quality (hit@k, MRR) and context size for get_context on pinned real repos.

    python eval/run.py                 # all splits, top-5
    python eval/run.py --split dev -v  # tune on dev only; list every non-#1 case

Cases come from cases.json (public repos, cloned at a pinned commit) plus eval/cases.local.json if it exists
(same shape; git-ignored, for private repos with a local path as url).

Tune on `dev`, report `test`, and never tune on `test`: once a choice is made by looking at it, it stops being a
held-out measure. Tokens are counted with tiktoken (o200k_base) when installed, else estimated as bytes / 4. Both "+read" columns
charge the target file on a miss, since the agent then has to go find the code itself.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from gus import core  # noqa: E402


def split_of(case):
    # stable, question-derived split for cases that don't pin one: ~40% dev, ~60% test
    return case.get("split") or ("dev" if zlib.crc32(case["q"].encode()) % 5 < 2 else "test")


def checkout(name, spec, repos_dir):
    dest = repos_dir / name
    url = os.path.expanduser(spec["url"])
    if not dest.exists() and not url.startswith(("http://", "https://", "git@")) and not os.path.isdir(url):
        print(f"skipping {name}: local repo {url} not found", file=sys.stderr)
        return None
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    have = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
    if have != spec["commit"]:
        subprocess.run(["git", "fetch", "-q", "origin", spec["commit"]], cwd=dest)
        subprocess.run(["git", "checkout", "-q", spec["commit"]], cwd=dest, check=True)
    return dest


def resolve(root, target):
    """[path] | [path, line] | [path, regex] -> (path, line or None). Fails loudly so a stale case can't silently pass."""
    path, *rest = target
    if not rest:
        return path, None
    if isinstance(rest[0], int):
        return path, rest[0]
    for n, line in enumerate((root / path).read_text().splitlines(), 1):
        if re.search(rest[0], line):
            return path, n
    raise SystemExit(f"target not found: {path} /{rest[0]}/")


def rank_of(results, targets):
    for i, r in enumerate(results, 1):
        if any(r["path"] == p and (ln is None or r["line_start"] <= ln <= r["line_end"]) for p, ln in targets):
            return i
    return None


try:  # a real BPE tokenizer when available (uv run --with tiktoken); Claude's own tokenizer isn't public
    import tiktoken

    _enc = tiktoken.get_encoding("o200k_base")
    TOKENIZER = "tiktoken o200k_base"

    def count(text):
        return len(_enc.encode(text, disallowed_special=()))
except ImportError:
    TOKENIZER = "bytes / 4 estimate (install tiktoken for real counts)"

    def count(text):
        return len(text.encode()) // 4


def tokens(results):
    return count(core.format_text(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--top-k", type=int, default=5)
    # clones live in the user cache, never inside this repo: they are other people's (or private) code
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    ap.add_argument("--repos-dir", type=Path, default=cache / "gus" / "eval-repos")
    ap.add_argument("--public", action="store_true", help="ignore cases.local.json (reproduce the published numbers)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    spec = json.loads((HERE / "cases.json").read_text())
    local = HERE / "cases.local.json"  # git-ignored: cases for private repos you have checked out locally
    if local.exists() and not a.public:
        extra = json.loads(local.read_text())
        spec["repos"] |= extra["repos"]
        spec["cases"] += extra["cases"]
    a.repos_dir.mkdir(parents=True, exist_ok=True)
    roots = {name: checkout(name, r, a.repos_dir) for name, r in spec["repos"].items()}

    rows = []  # (repo, split, q, rank, full+read_tok, outline_tok, outline+read_tok, file_tok, outline_shows_target)
    for case in spec["cases"]:
        split = split_of(case)
        if a.split not in ("all", split) or roots[case["repo"]] is None:
            continue
        root = roots[case["repo"]]
        targets = [resolve(root, t) for t in case["targets"]]
        full = core.get_context(case["q"], a.top_k, root)
        outline = core.get_context(case["q"], a.top_k, root, detail="outline")
        rank = rank_of(full, targets)
        file_tok = count((root / targets[0][0]).read_text(errors="replace"))
        # outline, then open the right hit; on a miss the agent still has to go read the file
        read = tokens(outline) + (tokens([full[rank - 1]]) if rank else file_tok)
        # can an agent recognise the right hit from the outline alone? (the target's own line is visible in it)
        shows = None
        if rank:
            hit_lines = {(root / p).read_text().splitlines()[ln - 1].strip()[:80] for p, ln in targets if ln and p == full[rank - 1]["path"]}
            shows = not hit_lines or any(h in outline[rank - 1]["snippet"] for h in hit_lines)
        full_read = tokens(full) + (0 if rank else file_tok)
        rows.append((case["repo"], split, case["q"], rank, full_read, tokens(outline), read, file_tok, shows))

    def report(label, rs):
        n = len(rs)
        hit = lambda k: sum(1 for r in rs if r[3] and r[3] <= k) / n
        mean = lambda i: int(statistics.mean(r[i] for r in rs))  # mean, not median: the miss cost lives in the tail
        mrr = sum(1 / r[3] for r in rs if r[3]) / n
        seen = [r[8] for r in rs if r[8] is not None]
        ks = sorted({1, 3, a.top_k})
        print(f"{label:<24} n={n:<3} " + "  ".join(f"hit@{k} {hit(k):>4.0%}" for k in ks) + f"  MRR {mrr:.2f}"
              f"  | tokens (mean): full+read {mean(4):>5}  outline {mean(5):>4}  outline+read {mean(6):>5}  target file {mean(7):>5}"
              f"  | outline shows target {sum(seen) / max(len(seen), 1):.0%}")

    print(f"tokens: {TOKENIZER}\n")
    for split in ("dev", "test"):
        rs = [r for r in rows if r[1] == split]
        if not rs:
            continue
        for repo in spec["repos"]:
            if any(r[0] == repo for r in rs):
                report(f"{split}/{repo}", [r for r in rs if r[0] == repo])
        report(f"{split}/ALL", rs)
        print()

    if a.verbose:
        for repo, split, q, rank, *_ in rows:
            if rank != 1:
                print(f"  {split:<4} {repo:<18} {'#' + str(rank) if rank else 'MISS':>5}  {q}")
        for repo, split, q, *_, shows in rows:
            if shows is False:
                print(f"  {split:<4} {repo:<18} outline hides target  {q}")


if __name__ == "__main__":
    main()

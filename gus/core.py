"""get_context: the one thing Gus does. The CLI and MCP server are thin wrappers around this."""

import re
import threading
from pathlib import Path

from . import embed, index

CANDIDATES = 50
RRF_K = 60
# weights below were chosen on the eval dev split (eval/run.py), where they sit on a flat plateau of MRR
TEST_WEIGHT = 0.8
PROSE_WEIGHT = 0.8  # docs restate what the code does and otherwise outrank it on code questions
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_|_test\.\w+$|\.(test|spec)\.\w+$")
PROSE_PATH = re.compile(r"\.(md|mdx|markdown|rst|txt)$|(^|/)docs?/", re.I)
DOC_QUERY = re.compile(r"\b(how (do|can|to)|docs?|documentation|guide|example|tutorial|explain|why)\b", re.I)
# a line that opens a definition: `def f(`, `class X`, `export const f = (`, `  async #method(...) {`
SIGNATURE = re.compile(
    r"\b(def|class|function|fn|func|struct|interface|trait|type|enum|impl|module|const|let|var)\b|=>"
    r"|^((public|private|protected|static|async|override|readonly)\s+)*(?!(if|for|while|switch|catch|return)\b)#?\w+\s*\(.*\)\s*(:[^={]+)?\{?$"
)
# a nested definition worth listing in an outline (stricter than SIGNATURE, which also matches body statements)
NESTED_DEF = re.compile(r"^(export |pub |public |private |protected |static |async |override )*(def|class|function|fn|func|interface|struct|impl|trait|enum)\s+\w")
OUTLINE_WIDTH = 160
OUTLINE_NESTED = 8
_lock = threading.Lock()


def refresh(root: str | Path | None = None) -> Path:
    """Bring the index up to date (building it on first use) and return the project root."""
    root = index.find_root(root)
    with _lock:
        if not index.exists(root):
            if root in (Path.home(), Path(root.anchor)):  # no project found: don't quietly embed a whole home directory
                raise ValueError(f"no project found ({root} is not inside a git repo); run `gus init` in the project first")
            index.log(f"no index found, building one for {root} (first run only)")
        # incremental: only files changed since the last call are re-embedded, so results track the agent's edits
        index.build(root, quiet=index.exists(root))
    return root


def warm(root: str | Path | None = None) -> None:
    """Update the index and load the model ahead of the first query (the MCP server runs this at startup)."""
    try:
        refresh(root)
        embed.embed_query(["warm up"])  # an unchanged index embeds nothing, so load the model explicitly
    except (Exception, SystemExit) as e:  # the first real query reports the same problem to the agent
        index.log(f"warm-up skipped: {e}")


def get_context(query: str, top_k: int = 5, root: str | Path | None = None, detail: str = "full") -> list[dict]:
    """Return the top_k chunks most relevant to query, as dicts: path, line_start, line_end, snippet, score, kind (+ page, note).

    detail="full" returns each chunk's code; detail="outline" returns only its leading lines (decorators, doc comment,
    signature, first body line) so the caller can pick a hit and read just that line range.
    """
    if detail not in ("full", "outline"):
        raise ValueError("detail must be 'full' or 'outline'")
    if not query.strip():
        raise ValueError("query must not be empty")
    top_k = max(1, min(int(top_k), CANDIDATES))  # at most the fused candidates; also guards top_k<=0 slicing
    root = refresh(root)

    conn = index.connect(root)
    try:
        # fusing extra phrasings of the query was measured too (eval/cases.json "rephrasings"): it lowered MRR for every model
        ranked = [
            (embed.VECTOR_WEIGHT, index.vector_candidates(conn, embed.embed_query([query])[0], CANDIDATES)),
            (1.0, index.keyword_candidates(conn, query, CANDIDATES)),
        ]
        # weighted reciprocal rank fusion: semantic matches + keyword matches
        scores = {}
        for weight, ids in ranked:
            for rank, cid in enumerate(ids):
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank + 1)
        rows = index.fetch(conn, list(scores)) if scores else {}
    finally:
        conn.close()

    # a chunk that *defines* a symbol named in the query counts as a top hit in a third ranking
    names = _symbol_names(query)
    if names:
        defines = re.compile(rf"\b(?:def|class|function|fn|func|struct|interface|trait|type|enum|const|let|var)\s+(?:{'|'.join(map(re.escape, names))})\b", re.I)
        for cid, row in rows.items():
            if defines.search(row[4]):
                scores[cid] += 1.0 / (RRF_K + 1)
    # ponytail: path heuristics, not a learned prior. Tests repeat every identifier; docs paraphrase the code.
    wants_tests, wants_docs = re.search(r"\btest", query, re.I), DOC_QUERY.search(query)
    for cid, row in rows.items():
        if not wants_tests and TEST_PATH.search(row[0]):
            scores[cid] *= TEST_WEIGHT
        if not wants_docs and PROSE_PATH.search(row[0]):
            scores[cid] *= PROSE_WEIGHT
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]

    results = []
    for cid in best:
        path, start, end, kind, text, images = rows[cid]
        snippet = text if detail == "full" else outline(text, kind, query)
        r = {"path": path, "line_start": start, "line_end": end, "snippet": snippet, "score": round(scores[cid], 4), "kind": kind}
        if kind == "pdf_page":
            r["page"] = start
        if images:
            where = "page" if kind == "pdf_page" else "section"
            r["note"] = f"this {where} also contains {images} image(s) not indexed by Gus — open the source file directly if visual content is needed"
        results.append(r)
    return results


def _symbol_names(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z_]\w*", query)
    names = [w for w in words if len(w) > 2 and ("_" in w or re.search(r"[a-z][A-Z]", w) or len(words) == 1)]
    if 1 < len(words) <= 3:  # "session store" -> SessionStore / session_store
        names += ["".join(words), "_".join(words)]
    return names


def outline(text: str, kind: str = "", query: str = "") -> str:
    """What a hit is, without its body: first line of its doc comment, decorators, the signature and the line after
    it, the signatures of definitions nested inside (so a matched method inside a small class stays visible), and
    the lines that best match the query (so a detail buried in the body is visible too)."""
    raw = [line.strip() for line in text.splitlines()]
    if kind in ("section", "text", "pdf_page"):  # prose: the heading/opening lines say what it is
        head = [line for line in raw if line][:2]
        return _clip(head + _matching_lines(raw, query, head))
    i, doc = 0, None
    if raw and raw[0].startswith("/*"):  # skip a leading block comment whole: its examples look like code
        while i < len(raw) and "*/" not in raw[i]:
            doc = doc or raw[i].strip("/* ") or None
            i += 1
        i += 1
    while i < len(raw) and (raw[i].startswith(("//", "# ")) or raw[i] in ("", "#")):
        doc = doc or raw[i].lstrip("/# ") or None
        i += 1
    lines = [line for line in raw[i:] if line.strip("\"'/*{}()[];, ")]  # drop bare `"""`, `/**`, `}`
    sig = next((n for n, line in enumerate(lines[:8]) if SIGNATURE.search(line)), None)
    if sig is None:
        picked = [doc] * bool(doc) + lines[: 2 - bool(doc)]
    else:
        decorators = [line for line in lines[:sig] if line.startswith("@")][:2]
        nested = [line for line in lines[sig + 2 :] if NESTED_DEF.match(line)][:OUTLINE_NESTED]
        picked = [doc] * bool(doc) + decorators + lines[sig : sig + 2] + nested
    return _clip(picked + _matching_lines(raw, query, picked))


def _matching_lines(lines, query, shown, limit=2):
    """Up to `limit` lines (in order) sharing at least two query words, best matches first, skipping ones already shown."""
    terms = {t for t in re.findall(r"\w+", index.fts_text(query).lower()) if len(t) > 2 and t not in index.STOPWORDS}
    if len(terms) < 2:
        return []
    scored = []
    for n, line in enumerate(lines):
        words = set(re.findall(r"\w+", index.fts_text(line).lower()))
        hits = len(terms & words)
        if hits >= 2 and line not in shown:
            scored.append((-hits, n, line))
    return ["… " + line for _, n, line in sorted(sorted(scored)[:limit], key=lambda x: x[1])]


def _clip(lines):
    return "\n".join(line if len(line) <= OUTLINE_WIDTH else line[: OUTLINE_WIDTH - 1] + "…" for line in lines)


def format_text(results: list[dict]) -> str:
    out = []
    for r in results:
        loc = f"page {r['page']}" if "page" in r else f"{r['line_start']}-{r['line_end']}"
        body = "\n".join("  " + line for line in r["snippet"].splitlines())
        note = f"\n  [note] {r['note']}" if "note" in r else ""
        out.append(f"{r['path']}:{loc}\n{body}{note}")
    return "\n\n".join(out) if out else "no results"

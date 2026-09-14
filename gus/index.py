"""Walk a project, chunk + embed changed files, and store everything in .gus/index.db."""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import chunker, embed

GUS_DIR = ".gus"
FORMAT_VERSION = 6  # bump when chunking/storage changes so old indexes rebuild
MAX_FILE_BYTES = 1_000_000
MAX_PDF_BYTES = 25_000_000  # PDFs carry far less text per byte than source files
STORE_EVERY = 256  # chunks per embed + commit
SKIP = re.compile(r"(\.(lock|min\.js|min\.css|map|svg|png|jpe?g|gif|ico|webp|zip|gz|tar|whl|so|dll|exe|bin|pyc)$|-lock\.json$)", re.I)


def log(msg):
    print(f"gus: {msg}", file=sys.stderr)  # never stdout: it is the MCP JSON-RPC channel


def find_root(start=None) -> Path:
    """Nearest ancestor holding a .gus index, else the git top level, else start itself."""
    start = Path(start or os.getcwd()).resolve()
    for d in (start, *start.parents):
        if (d / GUS_DIR).is_dir():
            return d
    for d in (start, *start.parents):
        if (d / ".git").exists():
            return d
    return start


def connect(root: Path) -> sqlite3.Connection:
    d = root / GUS_DIR
    d.mkdir(exist_ok=True)
    if not (d / ".gitignore").exists():
        (d / ".gitignore").write_text("*\n")  # keeps the index out of git without touching the user's .gitignore
    conn = sqlite3.connect(d / "index.db", timeout=30)
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER);
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, path TEXT, line_start INTEGER, line_end INTEGER,
            kind TEXT, text TEXT, images INTEGER, vec BLOB);
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(body);
        """
    )
    return conn


def load_config(root: Path) -> dict:
    path = root / GUS_DIR / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_config(root: Path, **values) -> None:
    (root / GUS_DIR).mkdir(exist_ok=True)
    (root / GUS_DIR / "config.json").write_text(json.dumps({**load_config(root), **values}, indent=2) + "\n")


def exists(root: Path) -> bool:
    return (root / GUS_DIR / "index.db").exists()


def list_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True,
        ).stdout.decode()
        files = out.split("\0")
    except (OSError, subprocess.CalledProcessError):
        files = _walk(root)
    return sorted(f for f in files if f and not f.startswith(GUS_DIR + "/") and not SKIP.search(f) and _regular(root / f))


def _regular(path: Path) -> bool:
    # symlinks are skipped: they can point outside the project (or at a file indexed under its real path anyway)
    return path.is_file() and not path.is_symlink()


def _walk(root):
    # ponytail: only the root .gitignore is honoured outside git repos; nested ones need per-dir specs
    import pathspec

    gi = root / ".gitignore"
    spec = pathspec.GitIgnoreSpec.from_lines(gi.read_text().splitlines() if gi.exists() else [])
    for dirpath, dirs, names in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules" and not spec.match_file(f"{(rel / d).as_posix()}/")]
        yield from ((rel / n).as_posix() for n in names if not spec.match_file((rel / n).as_posix()))


def _is_text(path: Path) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    with path.open("rb") as f:
        return b"\0" not in f.read(8192)


def build(root: Path, quiet=False) -> dict:
    """Incrementally (re)index root: only new/changed files are chunked and embedded."""
    conn = connect(root)
    say = (lambda m: None) if quiet else log
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    # the project's choice lives in .gus/config.json, so an agent-launched `gus serve` or a later shell without
    # GUS_MODEL uses it too (and never silently re-embeds a slow model's index with the default one)
    cfg = load_config(root)
    embed.use(
        os.environ.get("GUS_MODEL") or cfg.get("model") or meta.get("model") or embed.DEFAULT_MODEL,
        os.environ.get("GUS_DEVICE") or cfg.get("device") or "cpu",
    )
    if (cfg.get("model"), cfg.get("device")) != (embed.MODEL, embed.DEVICE):
        save_config(root, model=embed.MODEL, device=embed.DEVICE)
    fmt = f"{FORMAT_VERSION}:{embed.MODEL}"
    if meta.get("format", fmt) != fmt:
        log(f"index was built with {meta.get('model')} (format {meta['format'].split(':')[0]}); rebuilding with {embed.MODEL}")
        conn.close()  # table definitions may have changed too, so start from an empty database
        for f in (root / GUS_DIR).glob("index.db*"):
            f.unlink()
        conn = connect(root)
    # recorded before embedding, so an interrupted first build resumes with the same model
    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [("format", fmt), ("model", embed.MODEL)])
    conn.commit()

    known = {p: (m, s) for p, m, s in conn.execute("SELECT path, mtime, size FROM files")}
    current = {}
    for rel in list_files(root):
        try:
            st = (root / rel).stat()
        except OSError:  # deleted since it was listed (an agent editing while we index)
            continue
        if st.st_size <= (MAX_PDF_BYTES if rel.lower().endswith(".pdf") else MAX_FILE_BYTES):
            current[rel] = (st.st_mtime, st.st_size)
    changed = [p for p, sig in current.items() if known.get(p) != sig]
    removed = [p for p in known if p not in current]

    for p in removed + changed:
        ids = [(i,) for (i,) in conn.execute("SELECT id FROM chunks WHERE path=?", (p,))]
        conn.executemany("DELETE FROM fts WHERE rowid=?", ids)
        conn.execute("DELETE FROM chunks WHERE path=?", (p,))
        conn.execute("DELETE FROM files WHERE path=?", (p,))

    if changed:
        say(f"indexing {len(changed)} file(s) in {root} with {embed.MODEL}")
    batch, embedded, started, last_note = [], 0, time.time(), time.time()
    for n, rel in enumerate(changed, 1):
        path = root / rel
        try:
            chunks = chunker.chunk_file(path, rel) if _is_text(path) else []
        except Exception as e:  # one unparsable file must not kill the index
            say(f"skipped {rel}: {e}")
            chunks = []
        batch += [(rel, c) for c in chunks]
        conn.execute("INSERT INTO files VALUES (?, ?, ?)", (rel, *current[rel]))
        if len(batch) >= STORE_EVERY or n == len(changed):
            _store(conn, batch)
            conn.commit()  # a slow model's first build survives interruption: finished files aren't redone
            embedded += len(batch)
            batch = []
            if time.time() - last_note > 10:  # transformer models take minutes; show signs of life
                last_note = time.time()
                say(f"  {n}/{len(changed)} files, {embedded} chunks, {embedded / (last_note - started):.0f} chunks/s")

    if changed or removed or not conn.execute("SELECT 1 FROM meta WHERE key='indexed_at'").fetchone():
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('indexed_at', ?)", (str(time.time()),))
    conn.commit()
    conn.close()
    return {"changed": len(changed), "removed": len(removed), "files": len(current)}


def _store(conn, batch):
    if not batch:
        return
    # path and enclosing signatures carry a lot of the signal for code
    texts = [f"{rel}\n{c.context}\n{c.text}" for rel, c in batch]
    vecs = embed.embed_documents(texts)
    for (rel, c), v, t in zip(batch, vecs, texts, strict=True):
        cur = conn.execute(
            "INSERT INTO chunks (path, line_start, line_end, kind, text, images, vec) VALUES (?,?,?,?,?,?,?)",
            (rel, c.line_start, c.line_end, c.kind, c.text, c.images, v.tobytes()),
        )
        conn.execute("INSERT INTO fts (rowid, body) VALUES (?, ?)", (cur.lastrowid, fts_text(t)))


def fts_text(s: str) -> str:
    """Split camelCase and snake_case so 'refresh token' matches refreshToken and refresh_token."""
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s).replace("_", " ")


# --- queries used by core.get_context ---

def vector_candidates(conn, qvec: np.ndarray, n: int) -> list[int]:
    # ponytail: brute-force cosine over every vector; fine to ~100k chunks, switch to sqlite-vec/FAISS beyond
    rows = conn.execute("SELECT id, vec FROM chunks").fetchall()
    if not rows:
        return []
    mat = np.frombuffer(b"".join(v for _, v in rows), dtype=np.float32).reshape(len(rows), -1)
    top = np.argsort(-(mat @ qvec))[:n]
    return [rows[i][0] for i in top]


STOPWORDS = set("a an and are as at be by can do does for from how i if in is it of on or should so that the this to use used we what when where which who why will with".split())


def keyword_candidates(conn, query: str, n: int) -> list[int]:
    # stemming and exact-phrase matching for identifiers were measured too (eval/): both lowered dev MRR
    terms = [t for t in re.findall(r"\w+", fts_text(query).lower()) if t not in STOPWORDS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    return [r[0] for r in conn.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?", (match, n))]


def fetch(conn, ids: list[int]) -> dict[int, tuple]:
    q = f"SELECT id, path, line_start, line_end, kind, text, images FROM chunks WHERE id IN ({','.join('?' * len(ids))})"
    return {r[0]: r[1:] for r in conn.execute(q, ids)}


def stats(root: Path) -> dict:
    conn = connect(root)
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    out = {
        "root": str(root),
        "files": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
        "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "index_bytes": sum(f.stat().st_size for f in (root / GUS_DIR).iterdir() if f.is_file()),
        "last_indexed": float(meta["indexed_at"]) if "indexed_at" in meta else None,
        "model": meta.get("model"),
        "device": load_config(root).get("device", "cpu"),
    }
    conn.close()
    return out

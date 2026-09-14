# Contributing to Gus

Thanks for helping. Bug reports, bad-result reports ("I asked X and it returned Y"), and pull requests are
all welcome. For anything bigger than a bug fix (a new output mode, a ranking change, a new dependency), open
an issue first so we can agree on the approach before you spend time on it.

## Setup

```bash
git clone https://github.com/SUD747/gus && cd gus
uv venv && uv pip install -e '.[onnx]'   # or '.[gpu]' with an NVIDIA GPU, or '.' for the fast model only
```

Before opening a pull request:

```bash
uvx ruff check .
python tests/test_gus.py
python eval/run.py            # required if you touched chunking, embedding, indexing or ranking (see below)
```

CI runs ruff and the tests on Python 3.10 and 3.13.

## Layout

```
gus/core.py      get_context(): hybrid retrieval + ranking   <- the one shared function
gus/server.py    MCP server (thin wrapper over core)
gus/cli.py       CLI (thin wrapper over core)
gus/index.py     file walk, incremental indexing, SQLite storage, candidate queries
gus/chunker.py   code-aware chunking (tree-sitter, markdown, PDF)
gus/embed.py     embedding backends (model2vec static models, ONNX transformers)
gus/agents.py    `gus init` config wiring for Claude Code, Cursor, Codex, Antigravity
tests/           end-to-end tests on a tiny fixture project
eval/            retrieval eval: cases.json (pinned repos + questions) and run.py
```

## How it works

```
your repo ─▶ git ls-files ─▶ tree-sitter chunks ─▶ embeddings + full-text index ─▶ hybrid ranking ─▶ get_context
             (.gitignore)    (functions, classes,   (SQLite in .gus/, only          (meaning + exact
                              headings, PDF pages)   changed files re-embedded)      identifiers)
```

1. **Find files.** Uses `git ls-files`, so `.gitignore` is respected. Outside git, only the root `.gitignore`
   applies. Binaries, symlinks, lockfiles, minified assets, and files over 1 MB (PDFs over 25 MB) are skipped.
2. **Chunk.** Code is split into functions and classes with tree-sitter (Python, JS/JSX, TS/TSX, Go, Rust,
   Java, C, C++, Ruby), keeping doc comments attached and large classes split into methods. Markdown is split
   by heading, PDFs by page, and everything else into 60-line windows.
3. **Embed and store** in a SQLite database at `.gus/index.db`, with a full-text index next to the vectors.
   `.gus/` ignores itself in git.
4. **Retrieve.** Semantic search and keyword search (which catches exact identifiers like `refreshToken` or
   `refresh_token`) are merged into one ranking. Chunks that define a symbol named in the query rank higher;
   tests and docs rank lower unless the question is about them.

**The index never goes stale.** Every `get_context` / `search` call first does a cheap incremental update
(mtime + size check, re-embedding only changed files), so results keep up with the agent's own edits. The
first call on a project builds the index automatically. `gus serve` also starts that update and loads the model
in a background thread as soon as the agent connects (`core.warm`), so the agent's first question doesn't pay for
them: on the EmbeddingGemma GPU index, a first query 25 s after start took 0.11 s instead of 5 s.

**The tool description teaches agents to query well.** Agents read the MCP tool description on every call, so
`gus/server.py` tells them to use exact names, to search again with names from the first results, that results
are the best matches rather than every occurrence (use grep for that), and what isn't indexed. On the public eval,
bare-identifier queries put the right code first 5 times out of 5, against 47% for plain sentences.

## Ground rules

- **Keep the wrappers thin.** Behaviour belongs in `core.get_context`; the CLI and MCP server only translate
  arguments and output.
- **Never write to stdout from library code.** In `gus serve`, stdout is the MCP JSON-RPC channel. Log with
  `index.log()`, which writes to stderr.
- **Quality before token savings.** Returning less context only helps if the agent still gets the code it
  needs. A change that saves tokens but lowers hit@k or MRR is a regression. Aggressive savings belong in an
  opt-in mode (like `outline`), not the default. Report both numbers side by side.
- **Bump `FORMAT_VERSION`** in `gus/index.py` when you change chunking or the storage schema, so existing
  indexes rebuild instead of mixing old and new chunks.
- **Keep dependencies small.** Gus installs as a CLI tool on every developer's machine. No torch, and nothing
  that downloads at runtime except the embedding model.

## Evaluation

`eval/run.py` measures whether `get_context` returns the right code, and how much context it costs.

### Cases

`eval/cases.json` holds 62 questions with hand-checked answers about two repos pinned to exact commits:
[httpx](https://github.com/encode/httpx) (Python) and [ky](https://github.com/sindresorhus/ky) (TypeScript).
Most are phrased the way an agent asks ("strip the Authorization header when redirected to a different
origin"); a few are bare identifiers or docs questions. Answers were written from reading the code, before
running Gus.

Each case's `targets` lists acceptable answers as `[path]` (any chunk of the file), `[path, line]`, or
`[path, regex]` (the first matching line). A hit is a result whose line range contains the target line. A
regex that no longer matches fails the run loudly, so a stale case can't pass by accident.

To evaluate on your own or private code, put cases in `eval/cases.local.json`. It has the same shape, is
git-ignored, and its repo `url` may be a local path; `run.py` merges it in when present. Ranking was tuned
with 42 such questions on a private FastAPI + React app in addition to the public ones.

### Dev and test

Questions are split into **dev** and **test** (a stable hash of the question, or a pinned `"split"`).

- Tune and compare ideas on **dev** only: `python eval/run.py --split dev -v` lists every case that wasn't
  ranked first.
- Run **test** once per final configuration to confirm the gain holds. Never pick between options by looking
  at test; once you do, it stops being a held-out measure.

### Running

```bash
python eval/run.py                     # clones the pinned repos into ~/.cache/gus/eval-repos on first run
python eval/run.py --split dev -v
uv run --with tiktoken python eval/run.py --public   # the published numbers: real token counts, public cases only
GUS_MODEL=quality GUS_DEVICE=gpu python eval/run.py --repos-dir ~/.cache/gus/eval-repos-gemma   # one index dir per model
```

The report shows hit@1/3/5 and MRR, plus mean tokens for: `full` results, `outline`, outline plus
opening the right hit, and the target file alone. Both "+read" columns charge reading the target file when
the answer was missed. "Outline shows target" is how often the outline contains the answer's own line. Tokens are counted with
tiktoken (`o200k_base`) when it's installed and estimated as bytes / 4 otherwise; the report's first line says
which. Claude's tokenizer isn't public, so tiktoken is the closest reproducible stand-in.

### Current results (test split, public cases, top 5)

| Configuration | hit@1 | hit@3 | hit@5 | MRR | Tokens: full + miss | outline + read + miss |
|---|---|---|---|---|---|---|
| Reading the file that holds the answer | | | | | 6,472 | |
| Default (`potion-base-8M`, vector weight 0.6) | 45% | 70% | 82% | 0.59 | 2,059 (−68%) | 1,597 (−75%) |
| `quality` (EmbeddingGemma-300M fp32, vector weight 1.0) | 70% | 82% | 85% | 0.76 | 2,118 (−67%) | 1,758 (−73%) |

On hits alone, the right chunk is a median 98% smaller than the file it lives in, and the full top-5 results a
median 80% smaller; misses and small files pull the like-for-like averages above down to about −68%.

Per repo, EmbeddingGemma raised MRR from 0.67 to 0.85 on httpx and from 0.50 to 0.65 on ky. On the earlier
three-repo set, ranking tuning (weighted fusion, definition boost, test/doc down-weighting) raised the
default's test MRR from 0.50 to 0.59. Dropping `top_k` from 5 to 3 costs 12 points of hit@5, so the default
stays 5.

### Tried and rejected

Measured on the three-repo dev split (including the private cases), so these figures can't be reproduced
exactly from `cases.json` alone:

- **Fusing several phrasings of the query.** Each case has two blind rephrasings written by a separate model
  that saw only that question. They lowered dev MRR for every embedding model (EmbeddingGemma 0.85 → 0.79
  even at quarter weight): guessed identifiers and synonyms pulled in wrong code.
- **Porter stemming and exact-phrase identifier matching** in keyword search. Both lowered dev MRR.
- **A general-purpose cross-encoder reranker** (ms-marco-MiniLM). Worse, and about 1.4 s slower per query.
- **Other embedding models** (dev MRR against 0.75 for the default and 0.85 for EmbeddingGemma):
  bge-small-en-v1.5 0.80, gte-modernbert-base 0.78, jina-embeddings-v2-base-code 0.78,
  snowflake-arctic-embed-xs 0.77, LateOn-Code-edge 0.72. potion-code-16M tied with the default (0.60 vs 0.59
  on test) at twice the download. 4-bit EmbeddingGemma was about 6× slower than fp32 on CPU for the same
  quality.

### Agent A/B benchmark

The eval above scores retrieval. `eval/ab.py` scores what matters in the end: whether an agent finds the right code,
and at what cost, with Gus versus without. Each question becomes a Claude Code task run twice, in a fresh copy of the
pinned repo with no user settings and no other MCP servers:

- **off:** Read, Grep, Glob
- **on:** the same tools plus Gus's `get_context`

The agent ends with `ANSWER: path:start-end` lines. An answer is correct if it overlaps the definition that contains
the target line (so pointing at the right lines inside a function counts, same as Gus's whole-function ranges). The
report compares correctness, tokens, turns, time and cost per arm, pairs the arms per question with a sign test,
and grades again from saved answers, so a grading fix never needs new runs.

```bash
python eval/ab.py --limit 4                          # pilot
python eval/ab.py --split test --public --jobs 4     # full test split; results in ~/.cache/gus/ab/<time>/
python eval/ab.py --report ~/.cache/gus/ab/<time>    # report again
```

It spends API usage (capped per run with `--max-budget-usd`) and isn't part of CI. Agents are nondeterministic,
so use `--repeats` before trusting small differences.

**First result** (test split, public cases, Claude Haiku, one run per question, 66 runs, $2.15 total):

| Arm | Correct | Tokens per task (median / mean) | Turns (median) | Seconds (median) | Cost |
|---|---|---|---|---|---|
| Without Gus | 85% | 43,414 / 63,626 | 6 | 15 | $1.28 |
| With Gus | 88% | 39,570 / 48,099 | 5 | 13 | $0.87 |

- **Correctness didn't change measurably:** 3 questions only right with Gus, 2 only without (sign test p = 1.0).
- **Tokens:** where both arms were right, Gus used a median 0.89× the tokens and was cheaper on 14 of 26. The bigger
  saving is in the tail: mean tokens −24% and cost −32%, because without Gus the agent sometimes reads its way
  through many large files.
- **The agent used Gus in 27 of 33 runs** without being told to (no `--note`).
- Agent tokens fall far less than the retrieval eval's −68%, because every turn also carries the system prompt, the
  tool definitions and the conversation so far, which Gus doesn't shrink.
- These runs used chunker format 5; the signature fix in format 6 left retrieval results unchanged on the eval.

Worth running next: `--repeats 3`, a stronger model (`--model sonnet`), and `--note`.

Building this benchmark caught three problems, now fixed: a mislabeled case (the httpx encoding-setter target
matched `Headers.encoding` instead of `Response.encoding`, so both Gus and the agents were marked wrong for the
right answer), grading that rewarded Gus's whole-function ranges over an agent's precise lines, and a chunker bug
that split a long method's signature line into its own one-line chunk. A related change, keeping small statements
whole when a group of them is cut into windows, was measured too and reverted: it lowered ky's dev MRR from 0.67 to
0.63 by moving doc-heavy type chunks above implementations.

## Agent integrations

`gus init` writes MCP configs in each agent's own format (`gus/agents.py`). What's been checked by hand, with
the agent actually running:

| Agent | Checked | How |
|---|---|---|
| Claude Code | Tool call works | `claude -p` in a project after `gus init` called `mcp__gus__get_context` |
| Antigravity CLI 1.2.2 | Tool call works | an `agy` session called `get_context` via the global entry |
| Codex CLI 0.154 | Config loads | `codex mcp list` shows `gus` once the project is trusted; no tool call (needs a login) |
| Cursor | Not run | `.cursor/mcp.json` follows Cursor's documented format |

Antigravity CLI ignores project-level `.agents/mcp_config.json` (upstream issue
[google-antigravity/antigravity-cli#60](https://github.com/google-antigravity/antigravity-cli/issues/60)), so
`gus init` also adds a path-less `gus serve` entry to `~/.gemini/config/mcp_config.json`; `agy` starts MCP
servers in the project directory, and `gus serve` finds the project from there. Tests point `agents.GEMINI_DIR`
at a temp directory so they never touch the real file. If you change an integration, re-check it with the real
agent and update this table.

## Adding an embedding model

- **model2vec static models** work by name already: `gus init --model minishlab/<model>`.
- **Transformers:** add an entry to `ONNX_MODELS` in `gus/embed.py` with the ONNX file, the output to use (or
  the pooling), any query/document prefixes, and a `vector_weight` chosen on dev. Then run the eval with
  `GUS_MODEL=<id>` and include the dev and test numbers, plus indexing speed, in the pull request. Nothing
  outside `embed.py` should need to change.

## Adding a language

1. Add the grammar wheel (`tree-sitter-<lang>`) to `dependencies` in `pyproject.toml`. Prefer individual
   grammar packages over bundles that download grammars at runtime.
2. Map its file extensions in `GRAMMARS` in `gus/chunker.py`.
3. Check that the language's function/class node types match `DEFINITION` in the same file, and extend the
   pattern if not.
4. Add a small case to `tests/test_gus.py`, and bump `FORMAT_VERSION`.

## Releasing (maintainers)

1. Update `version` in `pyproject.toml` and move the `Unreleased` notes in `CHANGELOG.md` under the new
   version.
2. `uv build`, then check the sdist contains only `gus/`, `tests/`, `eval/run.py`, `eval/cases.json`,
   `README.md`, `LICENSE` and `pyproject.toml`: `tar tzf dist/*.tar.gz`.
3. `uv publish`, then tag the release (`git tag v0.x.y && git push --tags`).

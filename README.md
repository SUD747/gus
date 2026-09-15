<h1 align="center">Gus</h1>

<p align="center">
  <em>Your agent doesn't need the whole file. It needs the six lines that answer the question.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MCP-server%20%2B%20CLI-111111?style=flat-square" alt="MCP server and CLI">
  <img src="https://img.shields.io/badge/runs-100%25%20local-111111?style=flat-square" alt="Runs 100% local">
  <img src="https://img.shields.io/badge/python-3.10%2B-111111?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-111111?style=flat-square" alt="MIT license">
</p>

<p align="center">
  <strong>68% less context per question &middot; the right code in the top 5, 4 times in 5 &middot; no API keys, no services</strong><br>
  <sub>Measured on 33 held-out, hand-checked questions about <a href="https://github.com/encode/httpx">httpx</a> and <a href="https://github.com/sindresorhus/ky">ky</a> at pinned commits, with the default model, against reading the one file that holds the answer. Tokens are real tiktoken counts. When Gus misses, the file read is still counted. The optional <code>quality</code> model puts the right code <em>first</em> 70% of the time instead of 45%. <a href="https://github.com/SUD747/gus/blob/main/CONTRIBUTING.md#evaluation">How it's measured</a>.</sub>
</p>

<p align="center">
  Wires into <strong>Claude Code</strong> &middot; <strong>Cursor</strong> &middot; <strong>Codex CLI</strong> &middot; <strong>Antigravity CLI</strong> &middot; any MCP client &middot; any shell
</p>

---

You ask your agent where cookies are persisted between requests. It opens `httpx/_client.py`: 2,019 lines,
about 13,700 tokens, to find a six-line property. Then it does the same thing for the next question, and the
one after that, until the context window is full of code it read once and never needed.

Gus is a local context-retrieval layer for AI coding agents. The agent asks a question and gets back only the
functions, classes, doc sections and PDF passages that matter, with file paths and line ranges.
Agents call it as an MCP tool (`get_context`); humans, scripts and shell-only agents use `gus search`.

## Before / after

```console
$ cat httpx/_client.py                         # what an agent reads without Gus
                                               # 13,702 tokens, 2,019 lines

$ gus search "where are cookies persisted between requests"
                                               # 282 tokens, 38 lines, top 5 hits
```

**49× less context**, and the hits are the ones you'd have picked by hand:

```console
$ gus search "where are cookies persisted between requests" --top-k 3
httpx/_client.py:318-323
      @property
      def cookies(self) -> Cookies:
          """
          Cookie values to include when sending requests.
          """
          return self._cookies

httpx/_client.py:413-422
      def _merge_cookies(self, cookies: CookieTypes | None = None) -> CookieTypes | None:
          """
          Merge a cookies argument together with any cookies on the client,
          to create the cookies used for the outgoing request.
          """
          if cookies or self.cookies:
              merged_cookies = Cookies(self.cookies)
              merged_cookies.update(cookies)
              return merged_cookies
          return cookies

httpx/_client.py:325-327
      @cookies.setter
      def cookies(self, cookies: CookieTypes) -> None:
          self._cookies = Cookies(cookies)
```

## Numbers

One lucky example proves nothing, so Gus is scored on questions phrased the way agents ask them ("strip the
Authorization header when redirected to a different origin"), with answers written from reading the code
before any tuning, and a held-out test split that ranking changes are never tuned on.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/numbers-dark.svg">
    <img src="https://cdn.jsdelivr.net/gh/SUD747/gus@main/assets/numbers.svg" width="860" alt="Left: tokens the agent reads per question. Reading the file that holds the answer: 6,472. Gus results (default): 2,059, 68% less. Gus outline then the chosen lines: 1,597, 75% less. Right: how often the right code is ranked first. fast model: 45%. quality model: 70%.">
  </picture>
</p>

| Model | Right code first | In top 3 | In top 5 | Tokens per question |
|---|--:|--:|--:|--:|
| No Gus: read the file that holds the answer | | | | 6,472 |
| **`fast`** (default, any CPU) | 45% | 70% | 82% | **2,059 (−68%)** |
| **`quality`** (EmbeddingGemma, CPU or GPU) | **70%** | **82%** | **85%** | 2,118 (−67%) |

Tokens are counted with tiktoken (`o200k_base`). They include the file the agent still has to open when the
answer isn't in Gus's results, so a miss costs more than a hit, not nothing. On hits alone, the right chunk is a
median 98% smaller than its file. Ranking was also tuned against 42 questions on a private FastAPI + React
app, which aren't in these numbers.

**Quality is the constraint, token savings are the goal inside it.** `full` results are the default because
they give the agent the code itself. `outline` mode trims that to 1,597 tokens (−75%), but it shows what each hit
*is* rather than its body, so it's opt-in for locating things, not the default for understanding them.

### With a real agent

Claude Code answered the same 33 questions twice in a fresh copy of each repo: once with Read, Grep and Glob, and
once with those plus Gus. No one told it to use Gus; it chose to in 27 of 33 runs.

| Claude Haiku, 33 questions | Correct | Tokens per task (median / mean) | Turns (median) | Cost |
|---|--:|--:|--:|--:|
| Without Gus | 85% | 43,414 / 63,626 | 6 | $1.28 |
| With Gus | 88% | 39,570 / 48,099 | 5 | $0.87 |

Accuracy was the same within noise (3 questions right only with Gus, 2 only without). On a typical question the
saving is small (0.89× the tokens where both were right), and larger on hard ones where the agent would otherwise
read through many big files: −24% tokens and −32% cost on average. Every agent turn also carries the system prompt,
tool definitions and conversation so far, which Gus doesn't shrink, so the saving is smaller than the −68% above.
These are early numbers, from one run per question on a small model. [How it's run](https://github.com/SUD747/gus/blob/main/CONTRIBUTING.md#agent-ab-benchmark).

## Install

Requires Python 3.10+. The package is `gus-mcp`; the command it installs is `gus`.

```bash
uv tool install gus-mcp            # fast model only (small, CPU)
uv tool install 'gus-mcp[onnx]'    # + quality model on CPU
uv tool install 'gus-mcp[gpu]'     # + quality model on an NVIDIA GPU (~2 GB of CUDA libs)
pipx install gus-mcp               # pipx works the same way (extras too)
```

Or `pip install 'gus-mcp[onnx]'`, etc. From GitHub: `uv tool install 'gus-mcp[onnx] @ git+https://github.com/SUD747/gus'`.
Install either `onnx` or `gpu`, not both: they provide the same `onnxruntime` module. If `gus` isn't found
after installing, run `uv tool update-shell` (or `pipx ensurepath`) and open a new terminal.

## Quick start

```console
$ cd your-project
$ gus init
```

`gus init` asks which embedding model to use, builds the index, and wires Gus into every agent it finds.
Restart your agent. It now has a `get_context` tool.

Models download from Hugging Face on first use (30 MB fast, 1.2 GB quality). After that, Gus works offline.

<details>
<summary><strong>What <code>gus init</code> asks, and what it writes</strong></summary>

```console
$ gus init
Embedding model for this project:
  1) fast     potion-base-8M       30 MB download, indexes even large repos in seconds on any CPU
  2) quality  EmbeddingGemma-300M  1.2 GB download, needs gus-mcp[onnx] (CPU) or [gpu] (NVIDIA);
              puts the right code first 70% of the time vs 45% (see README, Embedding models)
Choose [1]: 2
Run the quality model on:
  1) cpu  works everywhere; ~12 chunks/s, so a 20,000-chunk repo takes ~30 min to index the first time
  2) gpu  NVIDIA CUDA via gus-mcp[gpu]; ~190 chunks/s on a laptop RTX 4050
Choose [1]: 2
...
done (onnx-community/embeddinggemma-300m-ONNX on gpu) — restart your agent so it picks up the `gus` MCP server
```

Without a terminal (scripts, CI, an agent running the command), `gus init` asks nothing: it keeps the
project's existing choice, or uses fast/CPU. Pass `--model fast|quality` and `--device cpu|gpu` to choose
explicitly. If the chosen model or device needs a package that isn't installed, `gus init` stops and prints
the install command.

| Agent | File written | When |
|---|---|---|
| Claude Code | `.mcp.json` | always |
| Cursor | `.cursor/mcp.json` | if `.cursor/` exists (or `--all`) |
| Antigravity CLI | `~/.gemini/config/mcp_config.json` (global, one entry for all projects) | if Antigravity CLI is installed (or `--all`) |
| Antigravity IDE | `.agents/mcp_config.json` | if `.agents/` exists (or `--all`) |
| Codex CLI | `.codex/config.toml` | if `.codex/` exists (or `--all`); Codex only loads project config for trusted projects |

It merges into existing files without touching other servers, and running it again is safe. It also appends
a short note to any of `CLAUDE.md`, `AGENTS.md`, `.cursorrules` that already exist, and to
`.cursor/rules/gus.mdc` if that directory exists. The note tells agents how to query Gus well (exact names,
then search again with names from the results), to open the file for surrounding code and use grep to find
every usage, and to fall back to `gus search` from the shell when the MCP tool isn't available.

The config entries use absolute paths to the `gus` executable and the project root, because agents launch
MCP servers with their own `PATH` (the Antigravity CLI entry has no project path; it starts servers in the
project directory). Re-run `gus init` on each machine instead of committing those entries.

To wire it up by hand, any MCP client works with:

```json
{ "mcpServers": { "gus": { "command": "gus", "args": ["serve", "/abs/path/to/project"] } } }
```

</details>

## Commands

```
gus init   [path] [--all] [--model fast|quality|<id>] [--device cpu|gpu]
                                              wire into agent configs + build index
gus serve  [path]                             MCP server over stdio (what agents launch)
gus search "<query>" [--top-k N] [--format text|json] [--detail full|outline] [--root DIR]
gus index  [path]                             build / update the index
gus stats  [path]                             files, chunks, size, last indexed time
gus help   [command]                          all commands, or one command's options
gus -v                                        version
```

With no path, commands use the nearest ancestor that has a `.gus/` index, then the git top level, then cwd.
You rarely need `gus index`: queries keep the index up to date.

<details>
<summary><strong>Result shape</strong></summary>

`get_context` and `--format json` return:

```json
[
  {
    "path": "httpx/_client.py",
    "line_start": 318,
    "line_end": 323,
    "snippet": "    @property\n    def cookies(self) -> Cookies: ...",
    "score": 0.0313,
    "kind": "decorated_definition"
  }
]
```

</details>

<details>
<summary><strong>Outline mode</strong></summary>

`detail="outline"` (MCP) or `--detail outline` (CLI) returns what each hit *is* instead of its whole body:
the first line of its doc comment, the signature, the signatures of anything defined inside it, and up to two
lines that best match the query (marked `…`). The agent picks a hit and reads just that line range.

```console
$ gus search "error when an async request is sent with the sync client" --detail outline --top-k 2
httpx/_client.py:1717-1749
  async def _send_single_request(self, request: Request) -> Response:
  Sends a single request, without handling any redirections.
  … if not isinstance(request.stream, AsyncByteStream):
  … "Attempted to send a sync request with an AsyncClient instance."

httpx/_client.py:1001-1034
  def _send_single_request(self, request: Request) -> Response:
  Sends a single request, without handling any redirections.
  … if not isinstance(request.stream, SyncByteStream):
  … "Attempted to send an async request with a sync Client instance."
```

`full` is the default. Outline plus opening the right hit uses about 20% fewer tokens than `full`, but it
hides code the agent may need, so it's meant for locating things, not for understanding them.

</details>

<details>
<summary><strong>PDFs and images</strong></summary>

PDF hits also carry `"page"`. If a hit's source page or section contains images, it carries a `"note"`:

```
docs/manual.pdf:page 12
  3.2.2 Memory optimisation
  ...
  [note] this page also contains 1 image(s) not indexed by Gus — open the source file directly if visual content is needed
```

Pages with no text at all (scans, full-page figures) are still indexed as a note saying the page is only
images, so the agent knows to open it. Gus only extracts text. It doesn't OCR or interpret images; it tells
the agent where they are.

</details>

## Embedding models

| Choice | Top result | Top 3 | Top 5 | Indexing speed | Query right after editing a 2,000-line file | Download |
|---|--:|--:|--:|---|---|---|
| `fast`: potion-base-8M (default) | 45% | 70% | 82% | ~20,000 chunks/s (CPU) | 90 ms | 30 MB |
| `quality` on `cpu`: EmbeddingGemma-300M | **70%** | **82%** | **85%** | ~12 chunks/s | 8.6 s | 1.2 GB + `gus-mcp[onnx]` |
| `quality` on `gpu`: EmbeddingGemma-300M | **70%** | **82%** | **85%** | ~190 chunks/s | 0.7 s | 1.2 GB + `gus-mcp[gpu]` |

Speeds are from a 20-thread laptop CPU and an RTX 4050 laptop GPU. CPU and GPU give identical results; only
speed differs.

**Pick `fast`** for large repos, machines without much memory, or if you want indexing to be instant.
**Pick `quality`** if you want the agent to get the right code first more often and can afford the setup:

- On CPU, the first index of a 20,000-chunk repo takes about half an hour (about 2 minutes on a GPU), using
  around 1.2 GB of memory. Progress is shown, and an interrupted index resumes where it stopped.
- Gus re-indexes changed files before every query, so a query right after editing a big file waits for that
  file to be re-embedded.
- GPU means NVIDIA CUDA. If a GPU is requested but can't be used, Gus warns and falls back to the CPU.
- The model is distributed under Google's [Gemma terms of use](https://ai.google.dev/gemma/terms).

The choice is saved in `.gus/config.json` (per machine, never committed), and every later `gus serve`,
`gus search` and `gus index` in the project uses it. To switch, run `gus init --model … --device …` again;
changing the model rebuilds the index. `GUS_MODEL` and `GUS_DEVICE` override the saved choice for one command.

## Known limits

- Very large repos (over ~100,000 chunks) will get slower queries: vector search is brute force.
- Git-ignored files, binaries, symlinks, lockfiles, minified assets, and files over 1 MB (PDFs over 25 MB)
  aren't indexed. Outside a git repo, only the root `.gitignore` is honoured.
- Files in languages without a bundled grammar are chunked into line windows instead of functions.
- Images in PDFs and Markdown are not read; Gus flags them so the agent can open the source.
- GPU acceleration requires an NVIDIA GPU with CUDA.
- Tested on Linux, with macOS in CI. Windows hasn't been tested yet.
- Gus won't index your home directory or the filesystem root on its own. Run it inside a project (a git repo,
  or a directory where you ran `gus init`).

## FAQ

**Does my code leave my machine?**
No. Indexing, embedding and search all run locally. The only network access is downloading the embedding model
from Hugging Face the first time.

**Isn't this just grep?**
Grep finds a string you already know, and every line that contains it. Gus ranks whole functions by what they
do, so a question phrased in plain words works even when you don't know the names. Exact identifiers still
work too: keyword search is part of the ranking, and it matches `refreshToken` and `refresh_token` alike.

**Will results go stale while the agent edits files?**
No. Every query re-indexes changed files first, so the agent sees its own edits.

**What if the answer isn't in the results?**
The agent reads the file, the way it would have anyway. That cost is counted in the numbers above.

**My agent has no MCP support.**
Use `gus search` from the shell. `gus init` leaves a note in `CLAUDE.md` / `AGENTS.md` telling agents to do so.

## Contributing

Bug reports, questions about results, and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/SUD747/gus/blob/main/CONTRIBUTING.md)
for setup, the evaluation, and how retrieval changes are judged. To report a security issue, see
[SECURITY.md](https://github.com/SUD747/gus/blob/main/SECURITY.md).

## License

MIT, see [LICENSE](https://github.com/SUD747/gus/blob/main/LICENSE). The optional EmbeddingGemma model has its own terms (see above).

<sub>Gus changes how an agent gathers context, and that can change its results in either direction. In our A/B benchmark (33 questions, Claude Haiku), overall accuracy was about the same with and without Gus, but 2 questions were answered correctly only without it. How it affects your agent depends on your codebase, model and questions.</sub>

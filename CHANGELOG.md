# Changelog

All notable changes to Gus are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

First release.

- `get_context(query, top_k, detail)` as an MCP tool (`gus serve`) and a CLI (`gus search`), both over one
  shared function.
- Hybrid retrieval: semantic and keyword search merged into one ranking, with a boost for chunks that
  define a symbol named in the query and lower weight for tests and docs.
- Code-aware chunking with tree-sitter for Python, JavaScript, TypeScript, Go, Rust, Java, C, C++ and Ruby;
  Markdown by heading; PDFs by page, with images flagged.
- Incremental indexing before every query, in SQLite under `.gus/`. PDF pages that are only images (scans,
  full-page figures) are indexed as a flagged placeholder instead of being dropped.
- `outline` detail mode for locating code in fewer tokens.
- The MCP tool description and the `gus init` note tell agents how to query well and when to read files or grep
  instead; `gus serve` updates the index and loads the model at startup so the first query is fast.
- `gus help [command]` and `gus -v`.
- `gus init` wires Claude Code, Cursor, Codex CLI and Antigravity CLI (via its global config, since the CLI
  ignores project-level MCP configs), and asks for the embedding model and device: `fast` (potion-base-8M,
  default) or `quality` (EmbeddingGemma-300M on CPU or NVIDIA GPU).
- Retrieval eval with pinned repos, a dev/test split and tiktoken token counts (`eval/run.py`), and an agent A/B
  benchmark that runs Claude Code on the same questions with and without Gus (`eval/ab.py`).

# Security policy

## Supported versions

Security fixes go into the latest release only.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub: open the repository's **Security** tab
and choose **Report a vulnerability**. Include the Gus version (`gus --version`), your OS, and steps to
reproduce. You should get a reply within a week.

## What Gus does on your machine

This is what to keep in mind when judging whether something is a vulnerability:

- **Reads** files in the project it's run in: git-tracked and untracked files that aren't git-ignored. It
  skips symlinks, so it doesn't follow links out of the project.
- **Writes** only inside `.gus/` in the project, plus the agent config and convention files that `gus init`
  lists (`.mcp.json`, `.cursor/mcp.json`, `.codex/config.toml`, `.agents/mcp_config.json`, `CLAUDE.md`,
  `AGENTS.md`, `.cursorrules`, `.cursor/rules/gus.mdc`) and, if Antigravity CLI is installed (`~/.gemini/antigravity-cli/`), one `gus` entry merged into
  Antigravity's global `~/.gemini/config/mcp_config.json`. Existing entries in these files are kept; a config
  file that isn't valid JSON is left untouched and `gus init` stops with an error.
- **Network:** downloads embedding models from Hugging Face on first use, into the Hugging Face cache
  (`~/.cache/huggingface` by default). Gus itself sends nothing else anywhere.
- **MCP server:** `gus serve` talks only over stdio to the agent that launched it. It opens no ports.
- **Returned content:** snippets come straight from project files. If a repository contains untrusted text
  (for example, instructions aimed at an AI agent), Gus passes it through like any file read would.

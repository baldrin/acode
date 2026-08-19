# AGENTS.md — working on acode

acode is a deliberately minimal agentic coding harness (Python 3.12+, uv).
The README is the authoritative user-facing description — keep it updated
when behavior or flags change. This file is for things an agent working on
the code should know without rediscovering them.

## Commands

- Tests: `uv run pytest -q` (fast, ~5s; run before claiming any change works)
- Run the CLI: `uv run acode` (needs `ANTHROPIC_API_KEY`; a dummy key is
  enough to see the startup banner and exercise /commands that don't hit
  the API)
- No linter or formatter is configured; match the existing style by eye.

## Layout (one module per role)

| File | Role |
| --- | --- |
| `acode/agent.py` | the agent loop top to bottom, prompt construction, compaction/summaries, the real API backend at the bottom |
| `acode/tools.py` | the seven tools: schemas, implementations, previews |
| `acode/safety.py` | path confinement, command denylist, command execution funnel |
| `acode/sandbox.py` | OS sandbox (Seatbelt on macOS, bubblewrap on Linux) + startup probes |
| `acode/handoff.py` | per-workspace handoff notes (save/load) |
| `acode/resume.py` | session resume from --log-full transcripts (replay, trimming, validation) |
| `acode/display.py` | terminal rendering, approval prompt (plain ANSI — no terminal libraries) |
| `acode/transcript.py` | JSONL session logs |
| `acode/cli.py` | argument parsing, the REPL, API error handling |

## Design rules (violating these is a bug, not a style choice)

- **The approval gate is the real control.** Nothing mutating (edit_file,
  write_file, run_command) may execute without it, except the explicitly
  opted-in `--trust-edits` for file edits. Denylist and sandbox are
  defense-in-depth, never a substitute.
- **All shell execution funnels through `safety.execute_command`.** Do not
  add a second path to `subprocess` for model-driven commands.
- **The system prompt must be static per session** — nothing volatile
  (timestamps, counters) interpolated, or the prompt-cache prefix breaks on
  every request. Per-session-constant values (workspace path, handoff date)
  are fine.
- **Everything visible, nothing silent.** New behavior shows up in the
  banner, warnings, or the transcript. Failures degrade loudly (warn and
  continue) rather than pretending; logging/persistence must never block
  the session.
- Minimalism: prefer extending an existing module over adding one; new
  runtime dependencies need a strong reason (currently: anthropic, httpx;
  optional truststore).

## Testing conventions

- The model is always faked (`FakeModelClient` in `tests/test_agent.py`,
  reused by `tests/test_cli.py`) — no network in tests, ever.
- File and shell tools run only inside `tmp_path` workspaces (fixtures in
  `tests/conftest.py`); tests must never write to the real `~/.acode` or
  this repository.
- Sandbox integration tests exercise the real OS mechanism and skip where
  none is available (macOS always has Seatbelt; Linux needs bubblewrap).
- New user-visible behavior gets a test and a README mention in the same
  change.

## Commit style

Single imperative summary line, optionally a short body explaining what and
why (see `git log`). One feature per commit.

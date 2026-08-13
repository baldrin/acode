# acode

A minimal agentic coding harness. Run it inside a repository, give it a task
in plain English, and it works on the task by reading files, editing files,
and running shell commands — in a loop, until the task is done or it needs
your input. Every file change and shell command is shown to you and requires
your explicit approval before it runs.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
cd acode
uv sync
```

## Configuration

The only required configuration is an API key for your Claude endpoint, in
the environment:

```sh
export ANTHROPIC_API_KEY=...
```

Put the export in `~/.zshenv` (sourced by every zsh invocation, unlike
`~/.zshrc`, which interactive shells only) so the tool works from anywhere.

## Usage

Run it from the directory you want to work on — the current directory becomes
the workspace boundary:

```sh
cd ~/some/repository
uv run --project ~/projects/acode acode "add a --verbose flag to the CLI"
```

The initial task is optional; without it you are prompted. After each task
completes, you can give a follow-up in the same conversation. `Ctrl+C`
interrupts the current turn; `Ctrl+D` or `/quit` exits.

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `claude-sonnet-5` | Claude model ID (e.g. `claude-opus-5`) |
| `--timeout` | `120` | per-command timeout for `run_command`, in seconds |
| `--max-tokens` | `32000` | maximum output tokens per model response |

### Reading the output

- Plain text streams in as the model produces it; a dim `thinking…` shows
  while waiting on the model.
- `● tool_name(args)` lines are tool calls; dimmed lines under them are result
  previews (errors in red).
- `⏸ approval required` means the tool is waiting for **you**: it shows a
  diff for edits (and for overwrites), the full content for new files, or the
  command for shell — then waits for `y`/`N`. Anything but yes is a no, and a
  declined action is fed back to the model as guidance, not an error.
- The dim `[tokens: …]` line after each model response reports usage.
  `cache read` should be nonzero from the second request of a session onward —
  that is prompt caching working (the system prompt, tool definitions, and
  earlier turns are cached and reused, cutting cost and latency).
- The tool warns plainly when the conversation approaches the model's context
  limit instead of failing mysteriously.

## Safety model

Understand what is and is not protecting you:

- **The approval gate is the real control.** No file edit, file write, or
  shell command executes without your explicit `y`. There is no auto-approve
  mode.
- **The workspace boundary**: file tools resolve every path (including
  symlinks and `..`) and refuse anything outside the launch directory.
  `.git` internals are off limits to the file tools; git operations go
  through `run_command` like everything else, so they hit the approval gate.
- **The denylist is a tripwire, not a security boundary.** Before you are
  even prompted, obvious footguns are rejected: `sudo`, `rm -rf` aimed at the
  workspace root or above, and piping a download into a shell. It catches
  accidents, not an adversary — read what you approve.
- Every command runs with the launch directory as its working directory and
  is killed (with its children) if it exceeds the timeout.
- Command execution is funneled through a single function
  (`acode.safety.execute_command`) so a sandboxed executor can replace it
  later without touching the rest of the code.

## Development

```sh
uv run pytest
```

Tests use a fake model client (no network) and run file/shell tools only
inside temporary directories. Layout:

| File | Contents |
| --- | --- |
| `acode/agent.py` | the agent loop, top to bottom, plus the real API backend |
| `acode/tools.py` | the seven tools: schemas, implementations, previews |
| `acode/safety.py` | path confinement, command denylist, command execution |
| `acode/display.py` | terminal rendering and the approval prompt |
| `acode/cli.py` | argument parsing, the REPL, API error handling |

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

Configuration is environment variables only. For a direct Claude endpoint,
an API key is all that is required:

```sh
export ANTHROPIC_API_KEY=...
```

Put exports in `~/.zshenv` (sourced by every zsh invocation, unlike
`~/.zshrc`, which interactive shells only) so the tool works from anywhere.

### Gateway routing

To route through an internal gateway that fronts Claude models (for example a
workspace serving endpoint), set:

```sh
export ANTHROPIC_BASE_URL=https://<host>/serving-endpoints/anthropic
export ANTHROPIC_AUTH_TOKEN=...   # sent as an Authorization: Bearer token
```

When `ANTHROPIC_BASE_URL` is set, the default model switches to the gateway's
naming (`databricks-claude-sonnet-4-6`); override with `--model` if your
endpoint serves different names. The startup banner shows the active endpoint
so it is always clear where requests are going.

If the gateway sits behind a private CA, point `ACODE_CA_BUNDLE` at the CA
bundle file — `CHUNKER_CA_BUNDLE` is honored as a fallback, so companion
tools on the same endpoint can share one setting. Alternatively install the
optional `truststore` package (`uv add truststore`) to use the operating
system's trust store.

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

Commands at the prompt:

| Command | Effect |
| --- | --- |
| `/new` | clear the conversation and start fresh (the response to the context warning) |
| `/cost` | show session token totals and the estimated cost so far |
| `/quit` | exit (also `Ctrl+D`) |

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | route-aware | Claude model ID; defaults to `claude-sonnet-5` direct, `databricks-claude-sonnet-4-6` via gateway |
| `--trust-edits` | off | file edits run without prompting (previews still shown, snapshots still taken); commands always stay gated |
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
- Usage accumulates across the whole session: `/cost` shows the running
  totals and an estimated dollar figure at any time, and the same summary is
  printed when you exit. Estimates use per-token rates for known model-name
  prefixes (including gateway-served names); an unknown model reports tokens
  only rather than guessing.
- The tool warns plainly when the conversation approaches the model's context
  limit instead of failing mysteriously.

## Safety model

Understand what is and is not protecting you:

- **The approval gate is the real control.** By default no file edit, file
  write, or shell command executes without your explicit `y`. The one
  deliberate relaxation is `--trust-edits`, opted into per invocation: file
  edits then run without prompting — but every edit's diff is still printed
  as it happens, the pre-task git snapshot makes them reversible, and shell
  commands (the unbounded tool) are always gated regardless. Starting it in a
  non-git workspace warns loudly that the snapshot net is absent.
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
- **Commands run with a scrubbed environment**: variables that look like
  credentials (names ending in `_TOKEN`, `_KEY`, `_SECRET`, `_PASSWORD`,
  `_CREDENTIALS`, or starting with `ANTHROPIC_`/`AWS_`) are removed before
  any command executes, so approved-but-hostile code cannot simply read your
  keys out of the environment. Flip side: a command that legitimately needs
  such a credential won't find it — run those yourself.
- **A workspace snapshot is taken before each task's first approved
  mutation** (git repositories only): tracked and untracked files are
  committed to `refs/acode/snapshot` without touching your index, working
  tree, or branches. If a task clobbers uncommitted work, restore with the
  `git restore --source <hash> -- .` command printed alongside the snapshot.
  Non-git workspaces get no safety net — the tool says nothing rather than
  pretending otherwise.
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

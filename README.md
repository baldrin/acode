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
| `/new` | clear the conversation and start fresh |
| `/compact` | summarize the conversation into a handoff note and continue from it (see Context management) |
| `/cost` | show session token totals, the estimated cost so far, and the current context size |
| `/help` | list the commands |
| `/quit` | exit (also `Ctrl+D`) |

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | route-aware | Claude model ID; defaults to `claude-sonnet-5` direct, `databricks-claude-sonnet-4-6` via gateway |
| `--trust-edits` | off | file edits run without prompting (previews still shown, snapshots still taken); commands always stay gated |
| `--no-sandbox` | sandbox on | run shell commands without the OS sandbox — see Safety model |
| `--log-full` | off | transcript logs everything, unclipped — see Transcripts |
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
  limit, and compacts it automatically at the next task boundary — see
  Context management.

## Context management

Long sessions no longer die at the context limit. The conversation's size is
tracked from each response's usage numbers, and when it crosses ~750k of the
~1M-token window at a task boundary, acode **compacts** it: the model writes
a handoff summary of the session (tasks and their status, changes made, key
decisions, failed approaches, next steps), and the conversation is replaced
by that summary. Work continues from there — the model re-reads files when
it needs details the summary dropped.

Things worth knowing:

- Compaction is one extra model request over the full conversation; its
  tokens are counted in `/cost` like any other request. The summary itself is
  logged to the transcript (a `compact` event) but not printed.
- Mid-task the tool only warns (compacting mid-task would derail the model);
  the compaction happens when the task finishes.
- `/compact` triggers it manually at any prompt — useful before starting a
  large new task on a long conversation. `/new` remains the blunt option:
  clear everything, keep nothing.
- After compacting, the prompt cache starts cold on the next request (the
  conversation prefix changed), so expect one request with a large
  `cache write` before reads resume.
- A failed or interrupted compaction changes nothing — the conversation is
  only replaced once a summary actually exists.

## Transcripts

Every session is logged to `~/.acode/logs/<timestamp>-<workspace>.jsonl`
(the exact path is printed at startup). One JSON object per line: your tasks,
the model's text, every tool call, **every approval outcome and the preview
you were shown to make it** (`approved` / `denied` / `auto_approved`), tool
results, per-request usage, and the session totals. Large payloads are
clipped — it is an audit trail, not a data store. If the log directory is
unwritable the session warns and continues; logging never blocks work.

With `--log-full`, nothing is clipped and the log becomes a complete record
of the model exchange: the system prompt and tool definitions at session
start, every message sent to and received from the model as `message` events
(complete content blocks — thinking summaries, `tool_use` ids, full tool
results), plus everything the default logs. The conversation state at any
point is reconstructible from the file. Two things to know: sessions can
produce multi-megabyte logs, and **a full log contains everything the model
read — including the contents of any file it opened**, so treat these logs
with the same care as the repositories they describe.

The model's reasoning is requested in summarized form (`thinking.display:
"summarized"` — a server-written summary; the raw chain of thought is never
returned by the API, and the setting does not change what thinking happens
or what it costs). Summaries appear as `thinking` blocks inside `--log-full`
message events; the default transcript and the terminal ignore them.

Useful one-liners:

```sh
# What did it change without asking, and what did the diffs look like?
jq -r 'select(.event=="approval" and .outcome=="auto_approved") | .preview' <log>

# Every command it ran (or tried to)
jq -r 'select(.event=="approval" and .tool=="run_command") | "\(.outcome): \(.preview)"' <log>

# What did each session cost?
jq -r 'select(.event=="session_end") | .estimated_cost_usd' ~/.acode/logs/*.jsonl
```

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
- **Commands run inside an OS sandbox that confines filesystem writes** to
  the workspace plus temp and cache directories (so builds and package
  managers keep working). macOS uses Seatbelt (`sandbox-exec`, ships with
  the OS); Linux uses [bubblewrap](https://github.com/containers/bubblewrap)
  (`apt install bubblewrap` / `dnf install bubblewrap`). Reads and network
  are deliberately left open in this first version: commands are individually
  approved, and the credential scrub (below) covers the obvious exfiltration
  path. The sandbox is probed at startup — a trivial command must succeed
  *and* a write outside the workspace must fail — so a missing or broken
  mechanism is reported in the banner rather than silently trusted, and the
  session runs unconfined with a warning (as it always did). `--no-sandbox`
  opts out explicitly. Like everything here, this is defense-in-depth behind
  the approval gate, not a substitute for reading what you approve.
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
  (`acode.safety.execute_command`), which is where the sandbox wraps the
  command; a stricter executor can still replace it in one place.

## Development

```sh
uv run pytest
```

Tests use a fake model client (no network) and run file/shell tools only
inside temporary directories. Sandbox integration tests exercise the real OS
mechanism where one is present (always on macOS; on Linux when bubblewrap is
installed) and are skipped elsewhere. Layout:

| File | Contents |
| --- | --- |
| `acode/agent.py` | the agent loop, top to bottom, plus the real API backend |
| `acode/tools.py` | the seven tools: schemas, implementations, previews |
| `acode/safety.py` | path confinement, command denylist, command execution |
| `acode/sandbox.py` | the OS sandbox: Seatbelt/bubblewrap wrapping, writable roots, startup probes |
| `acode/display.py` | terminal rendering and the approval prompt |
| `acode/cli.py` | argument parsing, the REPL, API error handling |

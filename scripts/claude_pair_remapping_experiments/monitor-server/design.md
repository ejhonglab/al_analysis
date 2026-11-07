# Queue monitor: design

## Two sources: what agents declare, and what the harness sees

The queue already records a lot in the shared directory: claims, the experiment index,
chat.md, and each host's `<host>_status.md`. Only the agents can write the reasons in those
files ("blocked on Tom's OK on the footnote"). But they are only as fresh as the agent's
last update, and they cannot tell a session that is idle and listening from one that
crashed, hit an API error, or sits on a permission prompt. In those states the agent cannot
write anything.

Claude Code hooks fill that gap. The harness runs them, not the model, so they cost no tokens
and do not depend on the agent remembering. The monitor shows both sources side by side: the
state of each session from the hooks, and the state each host's agent declared in the header
of its status file.

## Hooks write to the shared folder, not to the server

Hooks write `presence/<host>__<session>.json` into the shared directory. They do not send
anything to the server. So:

- the server is a pure reader of one folder, and no agent depends on it being up;
- no port has to be reachable from every host (hal runs ufw), and any host that mounts tensor
  takes part;
- history survives server restarts, and the server can move to another host.

The cost is latency. NFS attribute caching delays what the server sees by up to about a
minute, and the server polls (inotify does not see writes from other NFS clients). Agents
change state over minutes, so this does not matter.

## Keeping hooks off the critical path

The hooks fire on every tool call (PreToolUse and PostToolUse), so:

- they are registered with `async: true`, so Claude Code does not wait for them;
- each call is wrapped in `timeout -k 2 15`. tensor is a `hard` NFS mount, so a stalled NAS
  would otherwise leave hook processes hanging, and Claude Code does not enforce the timeout
  of async hooks;
- the full record lives in a local cache, and the shared file is rewritten only when the
  state, title or detail changes, on lifecycle events, or when it is over a minute old. A
  busy session writes about once a minute, not once per tool call;
- async hooks can finish out of order, so a per-session `flock` serializes them, and an event
  that started before the last recorded one is dropped;
- the hook never prints (stdout of UserPromptSubmit and SessionStart hooks goes into Claude's
  context) and always exits 0 (exit 2 from a Stop hook would keep Claude working).

## Liveness while idle: the chat_watch.sh heartbeat

No hook fires while a session is idle, so a session idle for hours looks the same as one that
died. Queue sessions already run `chat_watch.sh` whenever they are idle, polling every 60 s,
so it writes a heartbeat (`presence/<host>__<session>.watch`, keyed by
`$CLAUDE_CODE_SESSION_ID`) on each poll and marks it `exited` on exit. A current heartbeat
means "listening". A stale one, with no hook events, eventually means "gone". Having run
`chat_watch.sh` (or the queue's session title) is also what marks a session as being on the
queue: other sessions started in this folder are listed as "not on queue".

## Model and effort level from the transcript

The hook input carries the model only on some SessionStart events (not after `/clear`) and the
effort level never, and neither follows a mid-session switch (`/model`, `/effort`, or the move
from Fable to Opus when Fable's usage cap is hit). Every assistant message in the session
transcript records the model the API served and the session's effort level, so the hook reads
them from the last main-session (not subagent) assistant message, scanning back from the end
of the transcript (64 kB, then 1 MB, then 4 MB) rather than reading files that run to tens of
MB. The model in the SessionStart input is used only at SessionStart, when a fresh transcript
has no assistant message yet (or a resumed one may end on an older model).

## Declared status as front matter

Hosts wrote their status files in different layouts (`**Now:**` on hal, `## Now` on atlas),
which is fragile to parse. A small `key: value` header (`state`, `since`, `eta`,
`blocked_on`, `now`, `next`) gives the server fixed fields while the free text stays as it
is. For files without the header, the server falls back to the first line of the "Now"
section.

## Known limits

- A session killed mid-turn stays "working" until 15 min without events ("silent"), and then
  until 6 h without events ("gone"). A clean exit is recorded at once (SessionEnd).
- Ages are computed from the hosts' own timestamps, so they assume synchronized clocks (NTP).
- "joined" is the earliest of the host's first chat.md post and its earliest recorded session
  start. Sessions record their real start (from the transcript), even if they were already
  running when the hooks were installed.

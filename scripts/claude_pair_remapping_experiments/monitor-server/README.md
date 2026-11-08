# Model experiments monitor

A web page on the lab network, titled "Model experiments", showing which hosts take part in the
KC-remapping work queue (see `../CLAUDE.md`), what is claimed or running, how many tokens each
host's Claude sessions use and how close each claude.ai account is to its plan's limits, and
what the sessions are doing. It reads only the shared directory on tensor. Claude Code hooks,
the queue sessions' status line and `chat_watch.sh` keep the session records there current,
and each host's `<host>_status.md` supplies the rest.

## Viewing it

Open **http://hal:17486** on any computer on the lab network. The page reloads itself every
30 s. The second line under the title says which folder on tensor the page is built from
(`Tom/claude_kc_remapping`, and where the serving host mounts it).

- **Hosts**: one row per host. It shows the most urgent state of its sessions, the model and
  effort level of each of its live sessions (e.g. "Opus 5.5" over "xhigh"; together they set
  what the session may do on the queue, see "Permissions by model and effort" in
  `../CLAUDE.md`; greyed if the session is not on the queue, hover for the session names), the
  status the host's agent last declared (with what it is doing, ETA, and what it is blocked
  on), any experiments it runs and work it has claimed, and when its status file was updated,
  it last posted to chat.md ("last chat") and it joined, each with how long ago below. Click
  "status file" to see the whole `<host>_status.md`.
- **Running**: the experiments running now (from the index in experiment_priorities.md).
- **Queue**: experiments approved but unclaimed or awaiting approval, and the work claims table.
- **Usage**: three plots over the last 1, 3 or 7 days (links above them), each with a legend
  and hover text on every sample:
  - tokens per host, cumulative over the window: all tokens, all but cache reads, or output
    tokens only (links); tick "group hosts by Claude account" to sum the hosts that use one
    claude.ai account. By default only queue work in queue sessions is counted: "queue work
    only" leaves out the turns that a prompt typed into a session started (a request of Tom's)
    until the agent runs `presence_hook.py --mark queue`, and "queue sessions only" leaves out
    sessions that never ran `chat_watch.sh`; untick either to see everything;
  - the same for tokens on Fable models only;
  - plan usage per claude.ai account: the percentage of the 5-hour (dashed) and weekly (solid)
    limits, with a table of the latest reading, its reset times, and each host's first reading
    (its usage when it joined). Claude Code reports no separate Fable or Opus limit, so there
    is no Fable plot of this kind.
- **Recent chat.md entries**: the latest posts; click one for the whole message, again to fold
  it.
- **Claude sessions**: one row per session, from the hooks, with its model and effort level, its
  tokens so far (hover for the output tokens and the manual-work share) and whether its current
  turns count as queue or manual work:

  | state | meaning |
  |---|---|
  | needs you | waiting for a person (a permission prompt or a question) |
  | error | the last turn ended on an API error (rate limit, overload, ...) |
  | working | hook events in the last 15 min |
  | silent | last seen working, but no events for 15 min (a long command, or it died) |
  | listening | idle, with a current `chat_watch.sh` heartbeat |
  | idle | idle and not listening to chat.md |
  | gone | no events or heartbeat for 6 h |
  | ended | the session exited |

  Sessions in this folder that never ran `chat_watch.sh` and are not titled
  `al_analysis modeling - <host>` are marked "not on queue".
`http://hal:17486/api/state` returns the same data as JSON.

## Setting it up

**On every host that runs queue sessions**, pull both repositories:
```
git -C ~/src/al_analysis pull
git -C ~/src/dotfiles pull
```
The dotfiles' `claude_settings.json` (linked as `~/.claude/settings.json`) registers
`presence_hook.py` for the session and tool events. Running sessions reload the settings, so
no restart is needed. On machines without Tom's dotfiles (lab members' hosts set up with
`join_queue.sh`, where the checkout is ~/src/kc_queue/al_analysis), `.claude/settings.json` in
the experiments folder registers the same hooks for sessions started there. Its commands run
the copy of `presence_hook.py` in the session's own checkout, so they differ from the dotfiles'
ones, and in that folder on Tom's hosts each hook runs twice; that is harmless (the second run
rewrites the same session record). Giving the dotfiles the folder's commands would make it run
once again.

**On the host serving the page (hal)**, install the systemd user service and start it:
```
systemctl --user link ~/src/al_analysis/scripts/claude_pair_remapping_experiments/monitor-server/claude-queue-monitor.service
systemctl --user enable --now claude-queue-monitor
```
Then, once, allow the user service to run without a login (and start at boot), and open the
port to the LAN (hal runs ufw):
```
sudo loginctl enable-linger tom
sudo ufw allow from 10.0.0.0/24 to any port 17486 proto tcp
```
Logs: `journalctl --user -u claude-queue-monitor`.

## Applying code updates

Everything runs straight from each host's al_analysis checkout (`~/src/kc_queue/al_analysis`
on lab members' machines), so an update is a pull there, plus a restart for the server only:

| changed | pull on | then |
|---|---|---|
| `server.py` | hal (the host serving the page) | `systemctl --user restart claude-queue-monitor` |
| `presence_hook.py` | every host running queue sessions | nothing: every hook call runs the file afresh, so running sessions use the new version from their next event |
| `statusline_capture.py` | every host running queue sessions | nothing: each status-line update runs the file afresh |
| `../chat_watch.sh` | every host running queue sessions | nothing: a running watcher finishes on the old version and the next one it is restarted as uses the new one |
| hook registration (dotfiles `claude_settings.json`, `../.claude/settings.json`) | every host (also `git -C ~/src/dotfiles pull` on Tom's) | nothing: running sessions reload their settings |

On hal, for example:
```
git -C ~/src/al_analysis pull
systemctl --user restart claude-queue-monitor
systemctl --user status claude-queue-monitor
```
The status line should say `active (running) since` a time after the pull; if it says
`activating (auto-restart)` or `failed`, the new `server.py` does not start, and
`journalctl --user -u claude-queue-monitor -n 50` shows why. A session record only gains a field
the new hook writes (such as the effort level) at that session's next hook event after the pull,
so a session that stays idle shows the old record until it next does something.

## Configuration

- `server.py` options: `--shared` (default `$KC_SHARED_DIR`, else `Tom/claude_kc_remapping`
  under `/local/tensor` or `/mnt/tensor`, whichever has it), `--port` (17486), `--bind` (0.0.0.0) and `--refresh` (page reload, s;
  0 disables it). Edit `ExecStart` in the unit to change them. The thresholds behind the
  session states are constants at the top of `server.py`.
- Hook: `presence_hook.py` records sessions whose cwd or transcript is under this folder
  (`$KC_PRESENCE_MATCH`). It writes `presence/<host>__<session>.json` in the shared
  directory (found as for the server; nothing is written on a host without it) and keeps a local cache in `~/.cache/claude-queue-presence/`
  (`$KC_PRESENCE_CACHE`). Set `KC_PRESENCE_DEBUG=1` to make it raise instead of failing
  silently.
- Heartbeat: `../chat_watch.sh` writes `presence/<host>__<session>.watch` each poll, and
  marks it `exited` when it exits. It also runs `presence_hook.py --sample <session>` each poll,
  which writes a usage sample when one is due.
- Usage samples: `presence_hook.py` appends one to `usage/<host>__<session>.jsonl` every 15 min
  (`USAGE_SAMPLE_S`), at session start and end, and as soon as the status line has the
  session's first plan-usage reading: the session's tokens so far by model (from its
  transcript and its subagents', read incrementally; parse state in the local cache), the
  share of them in manual work (`tokens_manual`), the work mode in effect, whether the session
  is a queue session (it has run `chat_watch.sh`), the claude.ai account (a 12-character hash
  of the account ID, its display name and plan, from `~/.claude.json`; no email, no
  credentials), and the plan's 5-hour and weekly usage.
- Queue vs manual work: the transcript tags every prompt a person typed, so the turns from such
  a prompt to the next one are manual work unless the prompt was a queue command (the join
  command, the session-title tick, a bare acknowledgement); task notifications, scheduled
  prompts and peer messages keep the mode in effect. `presence_hook.py --mark queue` (the
  session ID from `$CLAUDE_CODE_SESSION_ID`) ends a manual segment, `--mark manual` starts one.
- Theme: the page follows the system or browser dark / light setting; the "theme" links at the
  top pin one (kept in the browser).
- Status line: `../.claude/settings.json` sets `statusLine` to `statusline_capture.py` for
  sessions started in the experiments folder, refreshed every 300 s. It saves the `rate_limits`
  Claude Code passes it to the local cache and prints the plan usage ("5h 23% (to 21:00) · week
  41% (to Mon 09:00)"). A project's `statusLine` replaces the user's own, so if the user has
  one in `~/.claude/settings.json`, the script runs that command and prints its output
  instead.
- Declared status: the header block at the top of each `<host>_status.md` (format in
  `../CLAUDE.md`, "Heartbeat"). A status file without one is shown as "unknown", with the
  first line of its "Now" section.

## Development

Only the standard library is needed (Python >= 3.8). To run it in the foreground on another
port:
```
python3 server.py --port 17487
```
Why it is built this way (hooks writing to the shared folder rather than to the server, async
hooks with throttled writes, the chat_watch.sh heartbeat) is in [design.md](design.md).

Tests (from the al_analysis root):
```
./venv/bin/python -m pytest scripts/claude_pair_remapping_experiments/monitor-server
```

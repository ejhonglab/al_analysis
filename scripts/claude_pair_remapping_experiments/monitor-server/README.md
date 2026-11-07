# KC queue monitor

A web page on the lab network showing which hosts take part in the KC-remapping work queue
(see `../CLAUDE.md`), what their Claude sessions are doing, and what is claimed or running.
It reads only the shared directory on tensor. Claude Code hooks and `chat_watch.sh` keep the
session records there current, and each host's `<host>_status.md` supplies the rest.

## Viewing it

Open **http://hal:17486** on any computer on the lab network. The page reloads itself every
30 s. The second line under the title says which folder on tensor the page is built from
(`Tom/claude_kc_remapping`, and where the serving host mounts it).

- **Hosts**: one row per host. It shows the most urgent state of its sessions, the model and
  effort level of each of its live sessions (e.g. "Opus 5.5 · xhigh"; together they set what the
  session may do on the queue, see "Permissions by model and effort" in `../CLAUDE.md`; greyed
  if the session is not on the queue, hover for the session names), the status the
  host's agent last declared (with what it is doing, ETA, and what it is blocked on), any
  experiments it runs and work it has claimed, when it last posted to chat.md, and when it
  joined. Click "status file" to see the whole `<host>_status.md`.
- **Claude sessions**: one row per session, from the hooks, with its model and effort level:

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
- **Queue**: experiments running, approved but unclaimed, or awaiting approval (from the
  index in experiment_priorities.md), plus the work claims table and the latest chat.md
  posts.

`http://hal:17486/api/state` returns the same as JSON.

## Setting it up

**On every host that runs queue sessions**, pull both repositories:
```
git -C ~/src/al_analysis pull
git -C ~/src/dotfiles pull
```
The dotfiles' `claude_settings.json` (linked as `~/.claude/settings.json`) registers
`presence_hook.py` for the session and tool events. Running sessions reload the settings, so
no restart is needed. On machines without Tom's dotfiles (lab members' hosts set up with
`join_queue.sh`), `.claude/settings.json` in the experiments folder registers the same hooks
for sessions started there; its entries are identical to the dotfiles' ones, so where both
apply each hook runs once. Keep the two identical when changing either.

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
  marks it `exited` when it exits.
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

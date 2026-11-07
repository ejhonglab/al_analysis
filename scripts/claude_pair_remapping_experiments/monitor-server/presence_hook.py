#!/usr/bin/env python3
"""Claude Code hook recording what a queue session is doing, for the queue monitor.

Registered (async) for the session lifecycle and tool events in ~/.claude/settings.json
(~/src/dotfiles/claude_settings.json). Each call reads the hook input JSON from stdin and
keeps one file per session in the shared directory, `presence/<host>__<session_id>.json`,
which `server.py` reads. Only sessions whose transcript or cwd is under this folder are
recorded (a session recorded once stays recorded, wherever it `cd`s).

To keep tool calls cheap and the NAS quiet, the full record lives in a local cache
(~/.cache/claude-queue-presence/) and the shared file is rewritten only when the state or
title changes, on lifecycle events, or when the shared copy is over a minute old. The hook
never prints anything (stdout of some events would reach Claude's context) and always exits
0. It is wrapped in `timeout` in the settings, since the NAS is a hard NFS mount.

Environment overrides (used by the tests): KC_SHARED_DIR, KC_PRESENCE_CACHE,
KC_PRESENCE_MATCH.
"""
import time

T_EVENT = time.time()

import fcntl
import json
import os
import re
import socket
import sys
from datetime import datetime

SHARED_DIR = os.environ.get('KC_SHARED_DIR', '/local/tensor/Tom/claude_kc_remapping')
CACHE_DIR = os.environ.get('KC_PRESENCE_CACHE') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
    'claude-queue-presence')
MATCH = os.environ.get('KC_PRESENCE_MATCH', 'claude_pair_remapping_experiments')
# rewrite the shared file at least this often while events keep coming
SHARED_REFRESH_S = 60
CACHE_MAX_AGE_S = 14 * 24 * 3600

# state after each event (Notification and a few others are handled in next_state)
EVENT_STATE = {
    'UserPromptSubmit': 'working',
    'PreToolUse': 'working',
    'PostToolUse': 'working',
    'PostToolUseFailure': 'working',
    'PermissionDenied': 'working',
    'PermissionRequest': 'waiting_permission',
    'PreCompact': 'compacting',
    'Stop': 'idle',
    'StopFailure': 'error',
    'SessionEnd': 'ended',
}
# Notification types that mean the session waits for a person
NEEDS_INPUT_NOTIFICATIONS = {
    'permission_prompt': 'waiting_permission',
    'elicitation_dialog': 'waiting_input',
    'elicitation_url_dialog': 'waiting_input',
    'agent_needs_input': 'waiting_input',
}
# events always written through to the shared file
LIFECYCLE_EVENTS = {
    'SessionStart', 'SessionEnd', 'Stop', 'StopFailure', 'Notification',
    'PermissionRequest', 'PreCompact', 'PostCompact', 'UserPromptSubmit',
}
TITLE_TICK = re.compile(r'\s*__set_session_title__\s+(\S.{0,63}?)\s*')


def iso(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec='seconds')


def short(text, n=200):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + '...'


def claude_pid():
    """PID of the `claude` process running this hook (env, else the parent chain)."""
    pid = os.environ.get('CLAUDE_PID')
    if pid and pid.isdigit():
        return int(pid)
    pid = os.getppid()
    for _ in range(8):
        try:
            with open('/proc/{}/stat'.format(pid)) as f:
                stat = f.read()
        except OSError:
            return None
        comm = stat[stat.index('(') + 1:stat.rindex(')')]
        if comm == 'claude':
            return pid
        pid = int(stat[stat.rindex(')') + 2:].split()[1])
        if pid <= 1:
            return None
    return None


def transcript_start(path):
    """Time of the first timestamped transcript entry (the session's real start), or None."""
    try:
        with open(path) as f:
            for _, line in zip(range(200), f):
                if '"timestamp"' not in line:
                    continue
                t = json.loads(line).get('timestamp')
                if t:
                    return datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
    except (OSError, ValueError, AttributeError):
        pass
    return None


def custom_title(d):
    """The session's custom title (hook input, else the file Claude Code keeps next to the
    transcript), or None."""
    if d.get('session_title'):
        return d['session_title']
    path = d.get('transcript_path') or ''
    if not path.endswith('.jsonl'):
        return None
    try:
        with open(path[:-len('.jsonl')] + '/custom-title.json') as f:
            return json.load(f).get('customTitle') or None
    except (OSError, ValueError, AttributeError):
        return None


def matches(d):
    """Whether a session not yet recorded belongs to the queue folder."""
    # the transcript's project folder is the launch dir with every non-alphanumeric as '-'
    transcript_key = re.sub(r'[^A-Za-z0-9]', '-', MATCH)
    return (MATCH in (d.get('cwd') or '')
        or transcript_key in os.path.dirname(d.get('transcript_path') or ''))


def next_state(event, d, prev):
    if event == 'SessionStart':
        return 'working' if d.get('source') == 'compact' else 'idle'
    if event == 'PostCompact':
        return 'working' if d.get('trigger') == 'auto' else 'idle'
    if event == 'Notification':
        kind = d.get('notification_type') or ''
        if kind in NEEDS_INPUT_NOTIFICATIONS:
            return NEEDS_INPUT_NOTIFICATIONS[kind]
        if kind == 'idle_prompt':
            return 'idle'
        return prev or 'idle'
    return EVENT_STATE.get(event, prev or 'working')


def update(rec, d, host, now):
    """Returns the new record after hook input `d`, given the previous record `rec`."""
    event = d.get('hook_event_name') or '?'
    rec = dict(rec)
    if not rec:
        # the transcript gives the real start of sessions already running when first seen
        start = transcript_start(d.get('transcript_path') or '') or now
        rec = {
            'version': 1,
            'host': host,
            'session_id': d.get('session_id'),
            'started': iso(start),
            'started_epoch': start,
            'events': 0,
        }
    prev = rec.get('state')
    state = next_state(event, d, prev)
    if state != prev:
        rec['state_since'] = iso(now)
        rec['state_since_epoch'] = now

    rec['state'] = state
    rec['last_event'] = event
    rec['last_event_time'] = iso(now)
    rec['last_event_epoch'] = now
    rec['events'] = rec.get('events', 0) + 1
    rec['claude_pid'] = claude_pid() or rec.get('claude_pid')
    for k in ('cwd', 'transcript_path', 'permission_mode'):
        if d.get(k):
            rec[k] = d[k]
    title = custom_title(d)
    if title:
        rec['session_title'] = title
    if d.get('agent_id'):
        rec['last_subagent_event'] = iso(now)

    if event == 'SessionStart':
        rec['start_source'] = d.get('source')
        if d.get('model'):
            rec['model'] = d['model']
        if d.get('source') in ('resume', 'fork'):
            rec['resumed'] = iso(now)
        rec.pop('ended', None)
        rec.pop('end_reason', None)
    elif event == 'SessionEnd':
        rec['ended'] = iso(now)
        rec['end_reason'] = d.get('reason')
    elif event == 'UserPromptSubmit':
        m = TITLE_TICK.fullmatch(d.get('prompt') or '')
        if m:
            rec['label'] = m.group(1)
    elif event in ('PreToolUse', 'PostToolUse', 'PostToolUseFailure', 'PermissionRequest'):
        rec['last_tool'] = d.get('tool_name')

    if event == 'Notification':
        rec['detail'] = short(d.get('message') or d.get('notification_type') or '')
        rec['detail_time'] = iso(now)
    elif event == 'StopFailure':
        err = {k: v for k, v in d.items() if 'error' in k or k in ('reason', 'message')}
        rec['detail'] = short(json.dumps(err) if err else 'API error')
        rec['detail_time'] = iso(now)
    elif event == 'PermissionRequest':
        rec['detail'] = short('permission needed for {}'.format(d.get('tool_name')))
        rec['detail_time'] = iso(now)
    return rec


def write_atomic(path, rec):
    tmp = os.path.join(os.path.dirname(path),
        '.{}.{}.tmp'.format(os.path.basename(path), os.getpid()))
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def prune_cache():
    cutoff = time.time() - CACHE_MAX_AGE_S
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except OSError:
            pass


def main():
    d = json.load(sys.stdin)
    sid = d.get('session_id')
    if not sid or not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', sid):
        return
    host = socket.gethostname().split('.')[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, sid + '.json')

    with open(cache + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(cache) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = {}
        if not rec and not matches(d):
            return
        # async hooks can finish out of order: drop an event older than the last recorded
        if T_EVENT < rec.get('last_event_epoch', 0):
            return

        new = update(rec, d, host, T_EVENT)
        event = new['last_event']
        changed = any(new.get(k) != rec.get(k)
            for k in ('state', 'session_title', 'label', 'detail', 'ended'))
        stale = T_EVENT - rec.get('shared_written_epoch', 0) >= SHARED_REFRESH_S
        if changed or stale or event in LIFECYCLE_EVENTS:
            presence = os.path.join(SHARED_DIR, 'presence')
            os.makedirs(presence, exist_ok=True)
            new['shared_written_epoch'] = T_EVENT
            write_atomic(os.path.join(presence, '{}__{}.json'.format(host, sid)), new)
        write_atomic(cache, new)

    if d.get('hook_event_name') == 'SessionStart':
        prune_cache()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # never disturb the session; KC_PRESENCE_DEBUG=1 shows why a record is missing
        if os.environ.get('KC_PRESENCE_DEBUG'):
            raise
    sys.exit(0)

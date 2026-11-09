#!/usr/bin/env python3
"""Claude Code hook recording what a queue session is doing, for the queue monitor.

Registered (async) for the session lifecycle and tool events in ~/.claude/settings.json
(~/src/dotfiles/claude_settings.json) on Tom's hosts, and in the experiments folder's
.claude/settings.json, which runs the copy in the session's own checkout (so it also works in
~/src/kc_queue on lab members' machines); the two commands differ, so in that folder on Tom's
hosts it runs twice per event, which is harmless. Each call reads the hook
input JSON from stdin and
keeps one file per session in the shared directory, `presence/<host>__<session_id>.json`,
which `server.py` reads. Only sessions whose transcript or cwd is under this folder are
recorded (a session recorded once stays recorded, wherever it `cd`s). The model and effort
level come from the last assistant message in the transcript, since the hook input carries
the model only on some SessionStart events and never the effort.

To keep tool calls cheap and the NAS quiet, the full record lives in a local cache
(~/.cache/claude-queue-presence/) and the shared file is rewritten only when the state or
title changes, on lifecycle events, or when the shared copy is over a minute old. The hook
never prints anything (stdout of some events would reach Claude's context) and always exits
0. It is wrapped in `timeout` in the settings, since the NAS is a hard NFS mount.

Every 15 min (and at the session's start and end) it also appends a usage sample to
`usage/<host>__<session_id>.jsonl` in the shared directory: the session's tokens so far by
model (summed from the `usage` of the assistant messages in its transcript and its subagents'
transcripts, read incrementally), the claude.ai account the machine is logged in with (a hash
of the account ID, its display name and plan, from ~/.claude.json), and the plan's 5-hour and
weekly usage as the status line last received it (`statusline_capture.py`). `chat_watch.sh`
runs `presence_hook.py --sample <session_id>` on each poll, so idle sessions keep sampling,
and that call also marks the session as a queue session (one that watches chat.md).

Tokens are split into queue work and manual work. The transcript tags every prompt a person
typed (`origin.kind` "human" / `promptSource` "typed"; task notifications, scheduled prompts
and peer messages are tagged otherwise), so each assistant message is attributed to the
segment that the last typed prompt before it started: a queue segment if that prompt was a
queue command (the join command, the session-title tick or a bare acknowledgement), a manual
segment otherwise (a request Tom typed into the session). `presence_hook.py --mark queue`
(run by the agent when it has finished such a request; the session ID comes from
$CLAUDE_CODE_SESSION_ID) ends a manual segment early; `--mark manual` starts one; `--at <time>`
(epoch seconds or ISO 8601, local time if no zone) places the boundary in the past, e.g. when a
request was finished before marks existed, and the transcript is re-read at the next sample.
Each sample carries the manual share as `tokens_manual`, the mode in effect and the
queue-session flag.

The shared directory is $KC_SHARED_DIR, else Tom/claude_kc_remapping under the first of
/local/tensor and /mnt/tensor that has it (nothing is written if neither does). Other
environment overrides (used by the tests): KC_PRESENCE_CACHE, KC_PRESENCE_MATCH.
"""
import time

T_EVENT = time.time()

import fcntl
import glob
import hashlib
import json
import os
import re
import socket
import sys
from datetime import datetime

# where the hosts mount tensor (hal: /local/tensor, an automount; atlas: /mnt/tensor)
TENSOR_MOUNTS = ('/local/tensor', '/mnt/tensor')
SHARED_SUBDIR = 'Tom/claude_kc_remapping'
CACHE_DIR = os.environ.get('KC_PRESENCE_CACHE') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
    'claude-queue-presence')
MATCH = os.environ.get('KC_PRESENCE_MATCH', 'claude_pair_remapping_experiments')
# rewrite the shared file at least this often while events keep coming
SHARED_REFRESH_S = 60
CACHE_MAX_AGE_S = 14 * 24 * 3600
# how far back from the end of the transcript to look for the last assistant message
TRANSCRIPT_TAIL_BYTES = (1 << 16, 1 << 20, 1 << 22)

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
# prompts the harness submits itself (the hook event does not say who typed a prompt)
NOTIFICATION_TEXT = re.compile(r'\s*(<task-notification|<system-reminder|<cross-session-message'
    r'|Another Claude session sent a message|\[Cross-session|\[Artifact comment)')
# typed prompts that are queue commands rather than requests of Tom's own (see prompt_mode)
QUEUE_JOIN = re.compile(r'\bjoin\b.*\bqueue\b|\bqueue\b.*\bjoin\b', re.I | re.S)
ACKNOWLEDGEMENTS = {'ok', 'okay', 'continue', 'go on', 'proceed', 'resume', 'carry on', 'yes'}
WORK_MODES = ('queue', 'manual')
# bumped when the transcript scan changes what it records (the work-mode split): a session
# state from an older scan is re-read from the start once (refresh_scan_state)
SCAN_VERSION = 2
BOUNDS_MAX = 500
# a usage sample at most this often per session (and at its start and end)
USAGE_SAMPLE_S = 15 * 60
# transcript usage fields -> sample keys
USAGE_KEYS = (('input_tokens', 'input'), ('output_tokens', 'output'),
    ('cache_read_input_tokens', 'cache_read'), ('cache_creation_input_tokens', 'cache_creation'))


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


def transcript_model(path):
    """Model and effort level of the last main-session assistant message in the transcript.

    Returns:
        (model, effort): the model ID the API served (e.g. 'claude-opus-5-5') and the
        session's effort level (e.g. 'xhigh'; None if the model has none), or (None, None) if
        no assistant message is in the last few MB of the transcript.
    """
    try:
        with open(path, 'rb') as f:
            end = f.seek(0, os.SEEK_END)
            for size in TRANSCRIPT_TAIL_BYTES:
                start = max(0, end - size)
                f.seek(start)
                lines = f.read(end - start).split(b'\n')
                # the first line of a chunk is usually cut off
                for line in reversed(lines[1:] if start else lines):
                    if b'"assistant"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get('type') != 'assistant' or d.get('isSidechain'):
                        continue
                    model = (d.get('message') or {}).get('model') or d.get('requestedModel')
                    # API error messages are written as model '<synthetic>'
                    if not model or model.startswith('<'):
                        continue
                    return model, d.get('effort') or d.get('perTurnEffort')
                if not start:
                    break
    except (OSError, AttributeError, TypeError):
        pass
    return None, None


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
    model, effort = transcript_model(d.get('transcript_path') or '')
    if event == 'SessionStart' and d.get('model'):
        # the model the session starts with (a resumed transcript may end on another one)
        model = d['model']
    if model:
        rec['model'] = model
        if effort:
            rec['effort'] = effort
        else:
            rec.pop('effort', None)
    if d.get('agent_id'):
        rec['last_subagent_event'] = iso(now)

    if event == 'SessionStart':
        rec['start_source'] = d.get('source')
        if d.get('source') in ('resume', 'fork'):
            rec['resumed'] = iso(now)
        rec.pop('ended', None)
        rec.pop('end_reason', None)
    elif event == 'SessionEnd':
        rec['ended'] = iso(now)
        rec['end_reason'] = d.get('reason')
    elif event == 'UserPromptSubmit':
        text = d.get('prompt') or ''
        m = TITLE_TICK.fullmatch(text)
        if m:
            rec['label'] = m.group(1)
        # a hint for the monitor until the next usage sample attributes the tokens exactly
        # (the transcript's prompt tags decide; see scan_usage); a task notification, a peer
        # message or a reminder is not a prompt anyone typed and leaves the mode alone
        if not NOTIFICATION_TEXT.match(text):
            rec['work_mode'] = prompt_mode(text)
            rec['work_mode_since'] = iso(now)
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


def prompt_mode(text):
    """The work mode a typed prompt starts: 'queue' for a queue command (the join command,
    the session-title tick, a bare acknowledgement), 'manual' for anything else Tom typed."""
    t = (text or '').strip()
    if TITLE_TICK.fullmatch(t) or QUEUE_JOIN.search(t):
        return 'queue'
    if t.lower().rstrip('.!') in ACKNOWLEDGEMENTS:
        return 'queue'
    return 'manual'


def line_epoch(d):
    t = d.get('timestamp')
    if not isinstance(t, str):
        return None
    try:
        return datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def typed_prompt(d):
    """(epoch, text) if a transcript line is a prompt a person typed, else None.

    Tool results, meta lines, compaction summaries, task notifications (`origin.kind`
    "task-notification"), scheduled prompts (`turnOrigin` "scheduled") and peer messages are
    not; a typed prompt has `origin.kind` "human" or `promptSource` "typed".
    """
    if d.get('type') != 'user' or 'toolUseResult' in d or d.get('isMeta') or d.get(
            'isCompactSummary'):
        return None
    origin = d.get('origin') if isinstance(d.get('origin'), dict) else {}
    if not (origin.get('kind') == 'human' or d.get('promptSource') == 'typed'):
        return None
    epoch = line_epoch(d)
    if epoch is None:
        return None
    c = (d.get('message') or {}).get('content')
    if isinstance(c, list):
        c = ' '.join(x.get('text') or '' for x in c if isinstance(x, dict)
            and x.get('type') == 'text')
    return epoch, c if isinstance(c, str) else ''


def add_bound(bounds, epoch, mode, source, text=''):
    """Inserts a work-mode boundary [epoch, mode, source, text] in time order (once)."""
    entry = [epoch, mode, source, short(text, 80)]
    for i, b in enumerate(bounds):
        if b[0] == epoch and b[2] == source:
            bounds[i] = entry
            return
        if b[0] > epoch:
            bounds.insert(i, entry)
            break
    else:
        bounds.append(entry)
    del bounds[:-BOUNDS_MAX]


def mode_at(bounds, epoch):
    """The work mode in effect at `epoch`: that of the last boundary at or before it
    ('queue' before any)."""
    mode = 'queue'
    for b in bounds:
        if b[0] > epoch:
            break
        mode = b[1]
    return mode


def _add_usage(totals, model, usage):
    t = totals.setdefault(model or '?', {k: 0 for _, k in USAGE_KEYS})
    for src, dst in USAGE_KEYS:
        v = usage.get(src)
        if isinstance(v, (int, float)):
            t[dst] += int(v)


def scan_usage(path, st, bounds=None, main=False):
    """Adds the token usage of the assistant messages appended to a transcript since the last
    call to its parse state, split by work mode.

    Claude Code writes one transcript line per content block, each repeating its message's
    `usage`, so lines are counted once per message ID: the last line of a message is held as
    `pending` until a line of another message follows. Each message is attributed to the work
    mode in effect at its timestamp (`mode_at`); in the main transcript, the prompts a person
    typed add the boundaries (`typed_prompt`, `prompt_mode`).

    Args:
        path: transcript (.jsonl).
        st: parse state from the previous call ({} the first time): `offset`, `totals`
            ({model: {input, output, cache_read, cache_creation}} of finished messages, all
            modes), `manual` (the same for messages in manual segments), `pending_id` and
            `pending` ([model, usage, mode] of the last message seen).
        bounds: the session's work-mode boundaries (updated in place when `main`).
        main: whether this is the session's own transcript (not a subagent's).

    Returns:
        the updated state.
    """
    bounds = [] if bounds is None else bounds
    try:
        size = os.path.getsize(path)
    except OSError:
        return st
    if size < st.get('offset', 0):
        st = {}
    st.setdefault('offset', 0)
    st.setdefault('totals', {})
    st.setdefault('manual', {})
    if size == st['offset']:
        return st
    with open(path, 'rb') as f:
        f.seek(st['offset'])
        data = f.read(size - st['offset'])
    end = data.rfind(b'\n')
    if end < 0:
        return st
    for line in data[:end].split(b'\n'):
        is_usage = b'"usage"' in line and b'"assistant"' in line
        is_prompt = main and (b'"human"' in line or b'"typed"' in line)
        if not (is_usage or is_prompt):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if is_prompt:
            tp = typed_prompt(d)
            if tp:
                add_bound(bounds, tp[0], prompt_mode(tp[1]), 'prompt', tp[1])
            continue
        if d.get('type') != 'assistant':
            continue
        m = d.get('message') or {}
        u = m.get('usage')
        model = m.get('model')
        # API errors are written as model '<synthetic>'
        if not isinstance(u, dict) or (model or '').startswith('<'):
            continue
        mid = m.get('id') or d.get('requestId') or d.get('uuid')
        if st.get('pending') and mid != st.get('pending_id'):
            _settle(st, st['pending'])
        epoch = line_epoch(d)
        mode = mode_at(bounds, epoch) if epoch is not None else 'queue'
        st['pending_id'] = mid
        st['pending'] = [model, {k: u.get(k) for k, _ in USAGE_KEYS}, mode]
    st['offset'] += end + 1
    return st


def _settle(st, pending):
    """Counts a finished message ([model, usage, mode]) in the parse state's totals."""
    model, usage = pending[0], pending[1]
    mode = pending[2] if len(pending) > 2 else 'queue'
    _add_usage(st['totals'], model, usage)
    if mode == 'manual':
        _add_usage(st.setdefault('manual', {}), model, usage)


def session_tokens(transcript, files, bounds=None):
    """Tokens of a session and its subagents so far, by model, and the manual-work share.

    Args:
        transcript: the session's transcript path.
        files: {path: parse state} from the previous call; updated in place.
        bounds: the session's work-mode boundaries (see `scan_usage`); updated in place from
            the main transcript's typed prompts, and applied to the subagents' messages too.

    Returns:
        ({model: {input, output, cache_read, cache_creation}} for every message, the same for
        the messages in manual segments).
    """
    bounds = [] if bounds is None else bounds
    paths = [transcript]
    if transcript.endswith('.jsonl'):
        paths += sorted(glob.glob(os.path.join(transcript[:-len('.jsonl')], 'subagents',
            '*.jsonl')))
    totals, manual = {}, {}
    for i, path in enumerate(paths):
        st = files[path] = scan_usage(path, files.get(path) or {}, bounds, main=(i == 0))
        for dst_all, key in ((totals, 'totals'), (manual, 'manual')):
            for model, t in st.get(key, {}).items():
                _add_usage(dst_all, model, {src: t[dst] for src, dst in USAGE_KEYS})
        if st.get('pending'):
            _add_usage(totals, *st['pending'][:2])
            if len(st['pending']) > 2 and st['pending'][2] == 'manual':
                _add_usage(manual, *st['pending'][:2])
    return totals, manual


def account_info():
    """The claude.ai account Claude Code is logged in with on this machine.

    Read from the `oauthAccount` entry of ~/.claude.json (or $CLAUDE_CONFIG_DIR/.claude.json),
    where Claude Code keeps the account details; the format is not documented, so anything
    missing gives None. Only a hash of the account ID leaves the machine, with the display
    name and plan.

    Returns:
        {'id', 'name', 'plan'}, or None (no claude.ai login, e.g. an API key).
    """
    paths = [os.path.expanduser('~/.claude.json')]
    if os.environ.get('CLAUDE_CONFIG_DIR'):
        paths.insert(0, os.path.join(os.environ['CLAUDE_CONFIG_DIR'], '.claude.json'))
    for path in paths:
        try:
            with open(path) as f:
                acct = json.load(f).get('oauthAccount') or {}
        except (OSError, ValueError, AttributeError):
            continue
        uuid = acct.get('accountUuid')
        if not uuid:
            return None
        return {'id': hashlib.sha256(uuid.encode()).hexdigest()[:12],
            'name': acct.get('displayName') or None, 'plan': acct.get('planDisplayName') or None}
    return None


def read_json_file(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def plan_usage(capture):
    """The 5-hour and weekly plan usage from a status-line capture, or None."""
    limits = (capture or {}).get('rate_limits')
    if not isinstance(limits, dict):
        return None
    out = {}
    for window in ('five_hour', 'seven_day'):
        w = limits.get(window)
        if isinstance(w, dict) and isinstance(w.get('used_percentage'), (int, float)):
            out[window] = {'used_percentage': w['used_percentage'],
                'resets_at': w.get('resets_at')}
    if not out:
        return None
    out['captured_epoch'] = capture.get('epoch')
    return out


def refresh_scan_state(st):
    """The file states to scan with: the saved ones, or none when the state comes from an
    older scan (`scan_version` below SCAN_VERSION), so that every file is re-read from the
    start once. The work-mode split arrived after sessions had been scanned, and the scan is
    incremental, so prompts typed before a host's hook had it never became boundaries and the
    tokens after them never counted as manual work. Totals are cumulative, so a re-read gives
    the same totals; the `--mark` boundaries are kept and the prompt ones re-derived."""
    if st.get('scan_version', 0) >= SCAN_VERSION:
        return st.get('files') or {}
    st['bounds'] = [b for b in st.get('bounds') or [] if len(b) > 2 and b[2] == 'mark']
    st['scan_version'] = SCAN_VERSION
    return {}


def sample_usage(rec, now, reason, force=False):
    """Appends a usage sample for a session to usage/<host>__<session>.jsonl when one is due.

    Due USAGE_SAMPLE_S after the last sample, when `force`d (session start and end), and at
    once when the status line has captured the session's first plan-usage reading. Must be
    called with the session's lock held.

    Returns:
        the sample written, or None.
    """
    sid, host = rec.get('session_id'), rec.get('host')
    transcript = rec.get('transcript_path') or ''
    if not sid or not host:
        return None
    state_path = os.path.join(CACHE_DIR, sid + '.usage.json')
    st = read_json_file(state_path) or {}
    limits = plan_usage(read_json_file(os.path.join(CACHE_DIR, sid + '.statusline.json')))
    # the first plan-usage reading of a session is sampled at once (its usage when it joined)
    first_limits = bool(limits) and not st.get('limits_epoch')
    if not (force or first_limits or now - st.get('sampled_epoch', 0) >= USAGE_SAMPLE_S):
        return None
    shared = shared_dir()
    if not shared:
        return None
    files = refresh_scan_state(st)
    bounds = st.setdefault('bounds', [])
    tokens, manual = session_tokens(transcript, files, bounds) if transcript else ({}, {})
    mode = mode_at(bounds, now)
    queue_session = bool(st.get('queue_session') or rec.get('queue_session'))
    sample = {
        'version': 2,
        'epoch': now,
        'time': iso(now),
        'host': host,
        'session_id': sid,
        'reason': reason,
        'model': rec.get('model'),
        'effort': rec.get('effort'),
        'account': account_info(),
        'tokens': tokens,
        'tokens_manual': manual,
        'work_mode': mode,
        'queue_session': queue_session,
        'plan_usage': limits,
    }
    udir = os.path.join(shared, 'usage')
    os.makedirs(udir, exist_ok=True)
    with open(os.path.join(udir, '{}__{}.jsonl'.format(host, sid)), 'a') as f:
        f.write(json.dumps(sample, sort_keys=True, separators=(',', ':')) + '\n')
    st.update(files=files, sampled_epoch=now)
    if limits:
        st['limits_epoch'] = limits.get('captured_epoch') or now
    write_atomic(state_path, st)
    total = sum(sum(t.values()) for t in tokens.values())
    rec['tokens'] = {'total': total, 'output': sum(t['output'] for t in tokens.values()),
        'manual': sum(sum(t.values()) for t in manual.values()), 'epoch': now}
    rec['work_mode'] = mode
    if bounds:
        rec['work_mode_since'] = iso(bounds[-1][0])
    if queue_session:
        rec['queue_session'] = True
    return sample


def shared_dir():
    """$KC_SHARED_DIR, else the shared directory under the first tensor mount that has it."""
    if os.environ.get('KC_SHARED_DIR'):
        return os.environ['KC_SHARED_DIR']
    for mount in TENSOR_MOUNTS:
        d = os.path.join(mount, SHARED_SUBDIR)
        if os.path.isdir(d):
            return d
    return None


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


def sample_only(sid):
    """`--sample <session_id>` (run by chat_watch.sh each poll): a usage sample if one is due."""
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', sid or ''):
        return
    cache = os.path.join(CACHE_DIR, sid + '.json')
    if not os.path.exists(cache):
        return
    with open(cache + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rec = read_json_file(cache)
        if not rec or rec.get('state') == 'ended':
            return
        changed = False
        if not rec.get('queue_session'):
            # only queue workers run chat_watch.sh, so this session is one
            rec['queue_session'] = True
            state_path = os.path.join(CACHE_DIR, sid + '.usage.json')
            st = read_json_file(state_path) or {}
            st['queue_session'] = True
            write_atomic(state_path, st)
            changed = True
        if sample_usage(rec, time.time(), 'watch') or changed:
            write_atomic(cache, rec)
            if changed:
                write_shared_presence(rec)


def write_shared_presence(rec):
    shared = shared_dir()
    if not shared or not rec.get('host') or not rec.get('session_id'):
        return
    presence = os.path.join(shared, 'presence')
    os.makedirs(presence, exist_ok=True)
    rec['shared_written_epoch'] = time.time()
    write_atomic(os.path.join(presence, '{}__{}.json'.format(rec['host'], rec['session_id'])),
        rec)


def parse_time(text):
    """A time given to `--at`: epoch seconds, or ISO 8601 ('2026-11-08T02:31:00-08:00';
    '2026-11-08 02:31' is local time)."""
    try:
        return float(text)
    except ValueError:
        pass
    d = datetime.fromisoformat(text.strip().replace('Z', '+00:00'))
    if d.tzinfo is None:
        d = d.astimezone()
    return d.timestamp()


def mark_mode(mode, sid, at=None):
    """`--mark queue|manual [session_id] [--at <time>]`, run by the agent: from now on (or
    from `at`, a past time), the session's tokens count as that kind of work, until the next
    prompt Tom types or the next mark. A boundary in the past (before the last sample) also
    makes the next sample re-read the transcript, so the tokens since it are re-attributed.
    Without a session ID, $CLAUDE_CODE_SESSION_ID (set in the session's shell commands)."""
    if mode not in WORK_MODES:
        raise SystemExit('mark: mode must be one of {}'.format(', '.join(WORK_MODES)))
    sid = sid or os.environ.get('CLAUDE_CODE_SESSION_ID') or ''
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', sid):
        raise SystemExit('mark: no session ID (pass it, or run from a Claude Code session)')
    now = time.time()
    try:
        when = now if at is None else parse_time(at)
    except ValueError:
        raise SystemExit('mark: --at wants epoch seconds or an ISO 8601 time, not {!r}'.format(at))
    if when > now + 60:
        raise SystemExit('mark: --at must not be in the future')
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, sid + '.json')
    with open(cache + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = os.path.join(CACHE_DIR, sid + '.usage.json')
        st = read_json_file(state_path) or {}
        bounds = st.setdefault('bounds', [])
        add_bound(bounds, when, mode, 'mark', 'at' if at is not None else '')
        if when < st.get('sampled_epoch', 0):
            st['files'] = {}   # re-read from the start at the next sample (totals are cumulative)
        write_atomic(state_path, st)
        current = mode_at(bounds, now)
        rec = read_json_file(cache)
        if rec:
            rec['work_mode'] = current
            rec['work_mode_since'] = iso(bounds[-1][0])
            write_atomic(cache, rec)
            write_shared_presence(rec)
    print('{}: {} work from {}{}'.format(sid[:8], mode, iso(when),
        '' if current == mode else ' (now {} work, since {})'.format(current, iso(bounds[-1][0]))))


def main():
    if sys.argv[1:2] == ['--sample']:
        sample_only(sys.argv[2] if len(sys.argv) > 2 else '')
        return
    if sys.argv[1:2] == ['--mark']:
        args, at = sys.argv[2:], None
        if '--at' in args:
            i = args.index('--at')
            at = args[i + 1] if i + 1 < len(args) else ''
            del args[i:i + 2]
        mark_mode(args[0] if args else '', args[1] if len(args) > 1 else '', at)
        return
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
        sample_usage(new, T_EVENT, event, force=event in ('SessionStart', 'SessionEnd'))
        changed = any(new.get(k) != rec.get(k)
            for k in ('state', 'session_title', 'label', 'detail', 'ended', 'model', 'effort'))
        stale = T_EVENT - rec.get('shared_written_epoch', 0) >= SHARED_REFRESH_S
        shared = None
        if changed or stale or event in LIFECYCLE_EVENTS:
            shared = shared_dir()
        if shared:
            presence = os.path.join(shared, 'presence')
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

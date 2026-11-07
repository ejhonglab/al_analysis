#!/usr/bin/env python3
"""LAN dashboard of the KC-remapping work queue: hosts, their Claude sessions, and claims.

Reads only the shared directory (Tom/claude_kc_remapping under /local/tensor or /mnt/tensor):
- presence/<host>__<session>.json, written by presence_hook.py (Claude Code hooks; includes
  each session's model and effort level);
- presence/<host>__<session>.watch, the heartbeat chat_watch.sh writes every poll;
- <host>_status.md, each host's self-reported status (its front matter, if any);
- chat.md entry headers, experiment_priorities.md (experiment index, work claims).

Serves `/` (HTML, reloads itself) and `/api/state` (the same as JSON). Standard library
only, so it runs under any system python3 >= 3.8.
"""
import argparse
import html
import json
import os
import re
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# where the hosts mount tensor (hal: /local/tensor, an automount; atlas: /mnt/tensor)
TENSOR_MOUNTS = ('/local/tensor', '/mnt/tensor')
SHARED_SUBDIR = 'Tom/claude_kc_remapping'
DEFAULT_PORT = 17486

# a session in 'working' state with no hook event for this long is shown as silent (a long
# foreground command, or a session that died mid-turn)
WORKING_SILENT_S = 15 * 60
# a chat_watch.sh heartbeat counts as current if younger than this many poll intervals
WATCH_FRESH_INTERVALS = 2.5
# no hook event and no current heartbeat for this long: the session is probably gone
GONE_S = 6 * 3600
# sessions are listed until this long after they ended / went quiet
HIDE_ENDED_S = 24 * 3600
HIDE_S = 3 * 24 * 3600
CACHE_S = 5
# queue sessions title themselves this way (CLAUDE.md, "Session names")
QUEUE_TITLE_PREFIX = 'al_analysis modeling - '

# how sessions rank when summarizing a host (first = shown)
VIEW_ORDER = ['needs you', 'error', 'working', 'silent', 'listening', 'idle', 'gone', 'ended']
VIEW_TEXT = {
    'needs you': 'waiting for a person (permission prompt or question)',
    'error': 'turn ended on an API error',
    'working': 'working (hook events in the last minutes)',
    'silent': 'last seen working, no hook events since',
    'listening': 'idle, chat_watch.sh heartbeat current',
    'idle': 'idle, not listening to chat.md',
    'gone': 'no events or heartbeat for hours',
    'ended': 'session ended',
}
# values the status front matter is expected to use (CLAUDE.md, "Heartbeat")
DECLARED_STATES = ('working', 'running', 'idle', 'blocked', 'offline')

CHAT_HEADER = re.compile(r'^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (\S+)\s*$')
STATUS_FILE = re.compile(r'^([A-Za-z0-9][A-Za-z0-9-]*)_status\.md$')
PRESENCE_FILE = re.compile(r'^([A-Za-z0-9-]+)__([A-Za-z0-9._-]+)\.(json|watch)$')
# model IDs: 'claude-opus-5-5', 'claude-sonnet-4-5-20250929', 'claude-3-5-haiku-20241022',
# optionally with a provider prefix ('us.anthropic.'), a '-v1:0' suffix or a '[1m]' context tag
MODEL_ID = re.compile(r'^(?:.*\.)?claude-'
    r'(?:(?P<family>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?'
    r'|(?P<major_old>\d+)(?:-(?P<minor_old>\d))?-(?P<family_old>[a-z]+))'
    r'(?:-\d{8})?(?:-v\d+(?::\d+)?)?(?P<context>\[\w+\])?$', re.I)


def default_shared():
    """$KC_SHARED_DIR, else the shared directory under the first tensor mount that has it."""
    if os.environ.get('KC_SHARED_DIR'):
        return os.environ['KC_SHARED_DIR']
    for mount in TENSOR_MOUNTS:
        d = os.path.join(mount, SHARED_SUBDIR)
        if os.path.isdir(d):
            return d
    return os.path.join(TENSOR_MOUNTS[0], SHARED_SUBDIR)


def model_name(model):
    """'Opus 5.5' for 'claude-opus-5-5' (and so on); the ID itself if it is not a Claude one."""
    m = MODEL_ID.match(model or '')
    if not m:
        return model or None
    family = m.group('family') or m.group('family_old')
    major = m.group('major') or m.group('major_old')
    minor = m.group('minor') or m.group('minor_old')
    name = '{} {}{}'.format(family.capitalize(), major, '.' + minor if minor else '')
    if m.group('context'):
        name += ' ({})'.format(m.group('context')[1:-1].upper())
    return name


def model_label(model, effort):
    """'Opus 5.5 · xhigh' (model and effort level), 'Opus 5.5' without an effort, or None."""
    name = model_name(model)
    if name and effort:
        return '{} \u00b7 {}'.format(name, effort)
    return name


def tensor_path(shared):
    """The shared directory relative to the tensor share, or None if it is not on a mount."""
    for mount in TENSOR_MOUNTS:
        if shared.startswith(mount.rstrip('/') + '/'):
            return shared[len(mount.rstrip('/')) + 1:]
    return None


def local_epoch(text):
    """Epoch of 'YYYY-MM-DD HH:MM' (local time) or an ISO 8601 timestamp; None if neither."""
    if not text:
        return None
    text = str(text).strip()
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def read_text(path):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return None


def read_json(path):
    text = read_text(path)
    if text is None:
        return None
    try:
        d = json.loads(text)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def parse_front_matter(text):
    """Splits `key: value` lines between leading `---` lines from the rest of a markdown.

    Returns:
        (fields, body): fields is {} if the text has no front matter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        return {}, text
    for i in range(1, min(len(lines), 40)):
        if lines[i].strip() == '---':
            fields = {}
            for line in lines[1:i]:
                k, sep, v = line.partition(':')
                if sep and k.strip():
                    fields[k.strip().lower()] = v.strip()
            return fields, '\n'.join(lines[i + 1:])
    return {}, text


def status_now_line(body):
    """First line of the 'Now' section of a status file without front matter."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(?:#+\s*now\b|\*\*now:?\*\*:?)\s*(.*)$', line, re.I)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest:
            return rest
        for nxt in lines[i + 1:]:
            if nxt.strip():
                return nxt.strip().lstrip('-* ').strip()
    return None


def md_table_after(text, heading):
    """Rows (dicts keyed by lowercase header) of the first table under a `## heading`."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r'^#+\s*' + re.escape(heading) + r'\b', line, re.I):
            start = i + 1
            break
    if start is None:
        return []
    rows, header = [], None
    for line in lines[start:]:
        if line.startswith('#'):
            break
        if not line.startswith('|'):
            if header is not None:
                break
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if header is None:
            header = [c.lower() for c in cells]
        elif not all(re.fullmatch(r':?-+:?', c) for c in cells if c):
            rows.append(dict(zip(header, cells)))
    return rows


def read_chat(shared):
    """Chat entry headers per host, and the most recent entries."""
    text = read_text(os.path.join(shared, 'chat.md'))
    if text is None:
        return {}, []
    hosts, entries, cur = {}, [], None
    for line in text.splitlines():
        m = CHAT_HEADER.match(line)
        if m:
            when, host = m.group(1), m.group(2)
            t = local_epoch(when)
            h = hosts.setdefault(host, {'first': t, 'last': t, 'posts': 0})
            h['posts'] += 1
            if t is not None:
                h['first'] = t if h['first'] is None else min(h['first'], t)
                h['last'] = t if h['last'] is None else max(h['last'], t)
            cur = {'time': t, 'host': host, 'first_line': ''}
            entries.append(cur)
        elif cur is not None and not cur['first_line'] and line.strip():
            cur['first_line'] = line.strip()[:240]
    return hosts, entries[-12:][::-1]


def read_statuses(shared, names):
    out = {}
    for name in names:
        m = STATUS_FILE.match(name)
        if not m:
            continue
        path = os.path.join(shared, name)
        text = read_text(path)
        if text is None:
            continue
        fields, body = parse_front_matter(text)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        out[m.group(1)] = {
            'fields': fields,
            'structured': bool(fields),
            'state': (fields.get('state') or '').split()[0].lower() if fields.get('state')
                else None,
            'now': fields.get('now') or status_now_line(body),
            'body': body.strip(),
            'mtime': mtime,
        }
    return out


def session_view(rec, watch, now):
    """Adds the derived fields the dashboard shows to a presence record."""
    s = dict(rec)
    last = rec.get('last_event_epoch') or 0
    listening = False
    if watch:
        interval = watch.get('interval_s') or 60
        age = now - (watch.get('epoch') or 0)
        listening = watch.get('state') == 'listening' and age < WATCH_FRESH_INTERVALS * interval
        s['watch'] = {'state': watch.get('state'), 'epoch': watch.get('epoch'),
            'age_s': age, 'current': listening}
    activity = max(last, (watch or {}).get('epoch') or 0) if listening else last
    state = rec.get('state')
    if state == 'ended':
        view = 'ended'
    elif now - activity > GONE_S:
        view = 'gone'
    elif state in ('waiting_permission', 'waiting_input'):
        view = 'needs you'
    elif state == 'error':
        view = 'error'
    elif state in ('working', 'compacting'):
        view = 'silent' if now - last > WORKING_SILENT_S and not listening else 'working'
    else:
        view = 'listening' if listening else 'idle'
    s['view'] = view
    s['listening'] = listening
    s['activity_epoch'] = activity
    s['queue'] = (bool(watch) or bool(rec.get('label'))
        or (rec.get('session_title') or '').startswith(QUEUE_TITLE_PREFIX))
    s['name'] = rec.get('session_title') or rec.get('label') or (rec.get('session_id') or '?')[:8]
    s['model_label'] = model_label(rec.get('model'), rec.get('effort'))
    hide_after = HIDE_ENDED_S if view == 'ended' else HIDE_S
    s['hidden'] = now - activity > hide_after
    return s


def read_presence(shared, now):
    pdir = os.path.join(shared, 'presence')
    try:
        names = os.listdir(pdir)
    except OSError:
        return []
    recs, watches = {}, {}
    for name in names:
        m = PRESENCE_FILE.match(name)
        if not m:
            continue
        d = read_json(os.path.join(pdir, name))
        if d is None:
            continue
        key = (m.group(1), m.group(2))
        (recs if m.group(3) == 'json' else watches)[key] = d
    sessions = []
    for key, rec in recs.items():
        rec.setdefault('host', key[0])
        rec.setdefault('session_id', key[1])
        sessions.append(session_view(rec, watches.get(key), now))
    # a heartbeat without hook records (hooks not registered on that host)
    for key, watch in watches.items():
        if key not in recs:
            rec = {'host': key[0], 'session_id': key[1], 'state': 'idle',
                'last_event_epoch': 0, 'no_hooks': True}
            s = session_view(rec, watch, now)
            if watch.get('state') != 'listening':
                s['view'] = 'ended' if now - (watch.get('epoch') or 0) < GONE_S else 'gone'
            sessions.append(s)
    sessions.sort(key=lambda s: -(s.get('activity_epoch') or 0))
    return sessions


def host_models(sessions):
    """Distinct models (with effort level) of a host's live sessions, queue sessions first.

    Returns:
        list of {'label', 'sessions' (names), 'queue' (any of them on the queue)}.
    """
    groups = {}
    for s in sessions:
        if s['view'] in ('ended', 'gone'):
            continue
        g = groups.setdefault(s.get('model_label') or 'model unknown',
            {'sessions': [], 'queue': False})
        g['sessions'].append(s['name'])
        g['queue'] = g['queue'] or s['queue']
    out = [dict(label=k, **g) for k, g in groups.items()]
    out.sort(key=lambda g: (not g['queue'], g['label'] == 'model unknown', -len(g['sessions'])))
    return out


def collect(shared, now=None):
    """Everything the dashboard shows, as one JSON-serializable dict."""
    now = time.time() if now is None else now
    out = {'shared': shared, 'tensor_path': tensor_path(shared), 'generated': now,
        'server_host': socket.gethostname(), 'errors': []}
    try:
        names = os.listdir(shared)
    except OSError as e:
        out['errors'].append('cannot list the shared directory: {}'.format(e))
        names = []

    statuses = read_statuses(shared, names)
    chat_hosts, recent = read_chat(shared)
    sessions = read_presence(shared, now)
    prio = read_text(os.path.join(shared, 'experiment_priorities.md')) or ''
    if names and not prio:
        out['errors'].append('experiment_priorities.md not readable')
    index = md_table_after(prio, 'Experiment index')
    claims = [r for r in md_table_after(prio, 'Work claims') if any(r.values())]

    hosts = set(statuses) | set(chat_hosts) | {s['host'] for s in sessions}
    host_rows = []
    for host in sorted(hosts):
        hs = [s for s in sessions if s['host'] == host and not s['hidden']]
        views = [s['view'] for s in hs]
        best = min(views, key=VIEW_ORDER.index) if views else None
        st = statuses.get(host)
        chat = chat_hosts.get(host, {})
        starts = [s.get('started_epoch') for s in sessions
            if s['host'] == host and s.get('started_epoch')]
        joined = min([t for t in [chat.get('first')] + starts if t] or [None])
        running = [r for r in index
            if r.get('state', '').lower().startswith('running') and host in r.get('host', '')]
        host_rows.append({
            'host': host,
            'joined': joined,
            'session_state': best,
            'sessions': len(hs),
            'queue_sessions': sum(1 for s in hs if s['queue']),
            'models': host_models(hs),
            'declared': st and {k: v for k, v in st.items() if k != 'body'},
            'status_body': st and st['body'],
            'last_chat': chat.get('last'),
            'chat_posts': chat.get('posts', 0),
            'running': running,
            'claims': [r for r in claims if host in r.get('host', '')],
        })

    counts = {}
    for r in index:
        st = (r.get('state') or '?').split()[0].lower()
        counts[st] = counts.get(st, 0) + 1
    out.update({
        'hosts': host_rows,
        'sessions': [s for s in sessions if not s['hidden']],
        'index_counts': counts,
        'running': [r for r in index if r.get('state', '').lower().startswith('running')],
        'approved': [r for r in index if r.get('state', '').lower().startswith('approved')],
        'proposed': [r for r in index if r.get('state', '').lower().startswith('proposed')],
        'claims': claims,
        'recent_chat': recent,
    })
    return out


# ---------------------------------------------------------------- HTML

CSS = """
:root { --bg: #f7f7f5; --fg: #1d1d1b; --muted: #6b6b66; --card: #ffffff; --line: #e2e1dc;
  --green: #1f7a3a; --blue: #1f5fa8; --amber: #9a6400; --orange: #b4470f; --red: #b3261e;
  --grey: #77766f; --chip: #efeee9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #161615; --fg: #ecebe6; --muted: #9b9a93; --card: #1f1f1d; --line: #34332f;
  --green: #5cc27a; --blue: #7cb0ec; --amber: #e0a63a; --orange: #f08a4b; --red: #f0766c;
  --grey: #8d8c85; --chip: #2a2a27; } }
:root[data-theme="dark"] { --bg: #161615; --fg: #ecebe6; --muted: #9b9a93; --card: #1f1f1d;
  --line: #34332f; --green: #5cc27a; --blue: #7cb0ec; --amber: #e0a63a; --orange: #f08a4b;
  --red: #f0766c; --grey: #8d8c85; --chip: #2a2a27; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 1.35rem; margin: 0 0 2px; }
h2 { font-size: 1.05rem; margin: 28px 0 8px; }
.sub { color: var(--muted); font-size: .88rem; margin-bottom: 14px; }
.err { background: color-mix(in srgb, var(--red) 15%, transparent); color: var(--red);
  padding: 8px 12px; border-radius: 6px; margin: 8px 0; }
.wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
.nw { white-space: nowrap; }
.muted { color: var(--muted); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .8rem;
  font-weight: 600; white-space: nowrap; background: var(--chip); }
.b-working { color: var(--green); } .b-listening { color: var(--blue); }
.b-idle { color: var(--grey); } .b-silent { color: var(--amber); }
.b-needs-you { color: #fff; background: var(--orange); } .b-error { color: #fff; background: var(--red); }
.b-gone, .b-ended, .b-none { color: var(--muted); }
.b-running { color: var(--green); } .b-blocked { color: var(--orange); }
.b-offline, .b-unknown { color: var(--muted); }
details { margin-top: 4px; }
summary { cursor: pointer; color: var(--muted); font-size: .82rem; }
pre { white-space: pre-wrap; word-break: break-word; font-size: .8rem; margin: 6px 0 0;
  max-height: 340px; overflow: auto; background: var(--chip); padding: 8px; border-radius: 6px; }
.legend { color: var(--muted); font-size: .82rem; margin-top: 8px; }
.legend span { margin-right: 14px; }
code { font-size: .85em; }
"""


def esc(x):
    return html.escape('' if x is None else str(x))


def fmt_time(t, now):
    """'15:04 (12 min ago)' today, 'Nov 4 17:42' otherwise."""
    if not t:
        return '<span class="muted">-</span>'
    d = datetime.fromtimestamp(t)
    today = datetime.fromtimestamp(now).date()
    stamp = d.strftime('%H:%M') if d.date() == today else d.strftime('%b %-d %H:%M')
    return '<span class="nw" title="{}">{} <span class="muted">({})</span></span>'.format(
        d.strftime('%Y-%m-%d %H:%M:%S'), stamp, ago(now - t))


def ago(s):
    s = max(0, s)
    if s < 90:
        return '{:.0f} s ago'.format(s)
    if s < 90 * 60:
        return '{:.0f} min ago'.format(s / 60)
    if s < 48 * 3600:
        return '{:.1f} h ago'.format(s / 3600)
    return '{:.0f} d ago'.format(s / 86400)


def badge(view, title=None):
    view = view or 'none'
    cls = 'b-' + view.replace(' ', '-')
    t = ' title="{}"'.format(esc(title)) if title else ''
    return '<span class="badge {}"{}>{}</span>'.format(esc(cls), t, esc(view))


def render(state, refresh):
    now = state['generated']
    out = []
    w = out.append
    w('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width, initial-scale=1">')
    if refresh:
        w('<meta http-equiv="refresh" content="{:d}">'.format(int(refresh)))
    w('<title>KC queue monitor</title><style>{}</style></head><body><main>'.format(CSS))
    w('<h1>KC remapping work queue</h1>')
    w('<div class="sub">served by {} &middot; generated {}{}<br>'.format(
        esc(state['server_host']), datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
        ' &middot; reloads every {:d} s'.format(int(refresh)) if refresh else ''))
    where = ('<code>{}</code> on tensor <span class="muted">(here {})</span>'.format(
        esc(state['tensor_path']), esc(state['shared'])) if state.get('tensor_path')
        else '<code>{}</code>'.format(esc(state['shared'])))
    w('data from {}: <code>presence/</code> (hooks), <code>&lt;host&gt;_status.md</code>, '
        '<code>chat.md</code>, <code>experiment_priorities.md</code></div>'.format(where))
    for e in state['errors']:
        w('<div class="err">{}</div>'.format(esc(e)))

    # hosts
    w('<h2>Hosts</h2><div class="wrap"><table><tr><th>host</th><th>sessions (hooks)</th>'
        '<th>model &middot; effort</th><th>declared status</th><th>doing now</th>'
        '<th>blocked on</th><th>running</th><th>status updated</th><th>last chat post</th>'
        '<th>joined</th></tr>')
    for h in state['hosts']:
        dec = h['declared'] or {}
        fields = dec.get('fields') or {}
        dstate = dec.get('state') or ('unknown' if dec else None)
        dtitle = None if dec.get('structured') else 'status file has no front matter'
        running = ', '.join('{} {}'.format(r.get('id', ''), r.get('name', ''))
            for r in h['running'])
        claims = '; '.join(r.get('item', '') for r in h['claims'])
        sess = badge(h['session_state'], VIEW_TEXT.get(h['session_state']))
        if h['sessions']:
            sess += ' <span class="muted">{} session{}{}</span>'.format(h['sessions'],
                's' if h['sessions'] != 1 else '',
                ', {} on queue'.format(h['queue_sessions']) if h['queue_sessions'] else '')
        now_text = esc(dec.get('now'))
        if fields.get('eta') and fields['eta'] not in ('-', ''):
            now_text += ' <span class="muted">(ETA {})</span>'.format(esc(fields['eta']))
        if fields.get('next') and fields['next'] != '-':
            now_text += '<br><span class="muted">next: {}</span>'.format(esc(fields['next']))
        if h['status_body']:
            now_text += '<details><summary>status file</summary><pre>{}</pre></details>'.format(
                esc(h['status_body']))
        models = '<br>'.join('<span class="nw{}" title="{}">{}{}</span>'.format(
            '' if g['queue'] else ' muted', esc(', '.join(g['sessions'])
                + ('' if g['queue'] else ' (not on queue)')), esc(g['label']),
            ' <span class="muted">&times;{}</span>'.format(len(g['sessions']))
                if len(g['sessions']) > 1 else '') for g in h['models'])
        blocked = fields.get('blocked_on')
        w('<tr><td><b>{}</b></td><td>{}</td><td>{}</td><td>{}{}</td><td>{}</td><td>{}</td>'
            '<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            esc(h['host']), sess, models or '<span class="muted">-</span>',
            badge(dstate, dtitle) if dstate else '<span class="muted">no status file</span>',
            '<br><span class="muted">since {}</span>'.format(esc(fields['since']))
                if fields.get('since') else '',
            now_text,
            esc(blocked) if blocked and blocked != '-' else '<span class="muted">-</span>',
            esc(running + ('; claims: ' + claims if claims else '')) or
                '<span class="muted">-</span>',
            fmt_time(dec.get('mtime'), now), fmt_time(h['last_chat'], now),
            fmt_time(h['joined'], now)))
    w('</table></div>')
    w('<div class="legend">sessions: ' + ''.join('<span>{} {}</span>'.format(badge(v), esc(t))
        for v, t in VIEW_TEXT.items()) + '</div>')

    # sessions
    w('<h2>Claude sessions</h2>')
    if state['sessions']:
        w('<div class="wrap"><table><tr><th>host</th><th>session</th><th>state</th>'
            '<th>model &middot; effort</th><th>last event</th><th>chat heartbeat</th>'
            '<th>started</th><th>detail</th></tr>')
        for s in state['sessions']:
            watch = s.get('watch')
            if watch:
                hb = '{} {}'.format(esc(watch.get('state')), fmt_time(watch.get('epoch'), now))
            else:
                hb = '<span class="muted">none</span>'
            last = '{} {}'.format(esc(s.get('last_event')), fmt_time(s.get('last_event_epoch'),
                now)) if s.get('last_event_epoch') else '<span class="muted">no hook events</span>'
            if s.get('last_tool') and s.get('view') in ('working', 'silent'):
                last += ' <span class="muted">{}</span>'.format(esc(s['last_tool']))
            detail = s.get('detail') if s['view'] in ('needs you', 'error') else (
                'ended ({})'.format(s.get('end_reason')) if s['view'] == 'ended' and
                s.get('end_reason') else '')
            name = esc(s['name'])
            if not s['queue']:
                name += ' <span class="muted">(not on queue)</span>'
            w('<tr><td>{}</td><td title="{}">{}</td><td>{}</td><td class="nw">{}</td><td>{}</td>'
                '<td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                esc(s['host']), esc(s.get('session_id')), name,
                badge(s['view'], VIEW_TEXT.get(s['view'])),
                esc(s['model_label']) or '<span class="muted">-</span>', last, hb,
                fmt_time(s.get('started_epoch'), now), esc(detail)))
        w('</table></div>')
    else:
        w('<p class="muted">No session has reported through the hooks yet.</p>')

    # queue
    counts = state['index_counts']
    w('<h2>Queue</h2><p class="muted">Experiment index: {}</p>'.format(
        esc(', '.join('{} {}'.format(n, k) for k, n in sorted(counts.items()))) or 'empty'))
    for title, rows in (('Running', state['running']), ('Approved, unclaimed',
            state['approved']), ('Proposed (awaiting approval)', state['proposed'])):
        if not rows:
            continue
        w('<h2>{}</h2><div class="wrap"><table><tr><th>ID</th><th>name</th><th>state</th>'
            '<th>host</th><th>started</th></tr>'.format(esc(title)))
        for r in rows:
            w('<tr><td class="nw">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                *(esc(r.get(k)) for k in ('id', 'name', 'state', 'host', 'started'))))
        w('</table></div>')
    if state['claims']:
        w('<h2>Work claims</h2><div class="wrap"><table><tr>' + ''.join(
            '<th>{}</th>'.format(esc(k)) for k in state['claims'][0]) + '</tr>')
        for r in state['claims']:
            w('<tr>' + ''.join('<td>{}</td>'.format(esc(v)) for v in r.values()) + '</tr>')
        w('</table></div>')

    # chat
    w('<h2>Recent chat.md entries</h2><div class="wrap"><table>'
        '<tr><th>posted</th><th>host</th><th>first line</th></tr>')
    for e in state['recent_chat']:
        w('<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            fmt_time(e['time'], now), esc(e['host']), esc(e['first_line'])))
    w('</table></div></main></body></html>')
    return ''.join(out)


# ---------------------------------------------------------------- server

class Monitor:
    """Caches `collect()` for CACHE_S seconds, so several viewers do not multiply NFS reads."""

    def __init__(self, shared):
        self.shared = shared
        self.lock = threading.Lock()
        self.state = None
        self.t = 0.0

    def get(self):
        with self.lock:
            if self.state is None or time.time() - self.t > CACHE_S:
                self.state = collect(self.shared)
                self.t = time.time()
            return self.state


def make_handler(monitor, refresh):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                body = render(monitor.get(), refresh).encode()
                ctype = 'text/html; charset=utf-8'
            elif path == '/api/state':
                body = json.dumps(monitor.get(), indent=1).encode()
                ctype = 'application/json'
            elif path == '/healthz':
                body, ctype = b'ok\n', 'text/plain'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--shared', default=None, help='shared directory (default: $KC_SHARED_DIR, '
        'else Tom/claude_kc_remapping under the first of {} that has it)'.format(
        ' and '.join(TENSOR_MOUNTS)))
    p.add_argument('--port', type=int, default=DEFAULT_PORT, help='default: %(default)s')
    p.add_argument('--bind', default='0.0.0.0', help='address to listen on (default: all)')
    p.add_argument('--refresh', type=int, default=30,
        help='seconds between page reloads, 0 for none (default: %(default)s)')
    args = p.parse_args()
    args.shared = args.shared or default_shared()
    server = ThreadingHTTPServer((args.bind, args.port),
        make_handler(Monitor(args.shared), args.refresh))
    server.daemon_threads = True
    print('serving {} on http://{}:{}/'.format(args.shared, socket.gethostname(), args.port),
        flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

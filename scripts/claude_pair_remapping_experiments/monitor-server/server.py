#!/usr/bin/env python3
"""LAN dashboard of the model-experiments work queue: hosts, claims, token and plan usage,
and the Claude sessions.

Reads only the shared directory (Tom/claude_kc_remapping under /local/tensor or /mnt/tensor):
- presence/<host>__<session>.json, written by presence_hook.py (Claude Code hooks; includes
  each session's model and effort level);
- presence/<host>__<session>.watch, the heartbeat chat_watch.sh writes every poll;
- usage/<host>__<session>.jsonl, the 15-min usage samples presence_hook.py appends (tokens by
  model, the claude.ai account, the plan's 5-hour and weekly usage);
- <host>_status.md, each host's self-reported status (its front matter, if any);
- chat.md entry headers, experiment_priorities.md (experiment index, work claims).

Serves `/` (HTML, reloads itself; query parameters `days`, `tokens` and `group` choose what the
usage plots show) and `/api/state` (the same data as JSON). Standard library only, so it runs
under any system python3 >= 3.8.
"""
import argparse
import html
import json
import math
import os
import re
import socket
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

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
TITLE = 'Model experiments'

# usage plots: time windows offered (days) and the default; token measures
USAGE_DAYS = (1, 3, 7)
USAGE_DEFAULT_DAYS = 3
TOKEN_MEASURES = {
    'all': 'all tokens',
    'nocache': 'all but cache reads',
    'output': 'output tokens',
}
USAGE_FILE = re.compile(r'^([A-Za-z0-9-]+)__([A-Za-z0-9._-]+)\.jsonl$')
# categorical slots (the dataviz reference palette, light / dark), assigned to hosts and
# accounts in a fixed (sorted) order
SERIES_COLORS = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70'),
    ('#eda100', '#c98500'), ('#e87ba4', '#d55181'), ('#008300', '#008300'),
    ('#6250d6', '#9085e9'), ('#e34948', '#e66767')]

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
            cur = {'time': t, 'host': host, 'first_line': '', 'lines': []}
            entries.append(cur)
        elif cur is not None:
            if not cur['first_line'] and line.strip():
                cur['first_line'] = line.strip()[:240]
            cur['lines'].append(line)
    for e in entries:
        e['text'] = '\n'.join(e.pop('lines')).strip()
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
    s['model_name'] = model_name(rec.get('model'))
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
        list of {'label' ('Opus 5.5 · xhigh'), 'model' ('Opus 5.5'), 'effort', 'sessions'
        (names), 'queue' (any of them on the queue)}.
    """
    groups = {}
    for s in sessions:
        if s['view'] in ('ended', 'gone'):
            continue
        name = s.get('model_name') or 'model unknown'
        effort = s.get('effort') if s.get('model_name') else None
        g = groups.setdefault((name, effort), {'sessions': [], 'queue': False})
        g['sessions'].append(s['name'])
        g['queue'] = g['queue'] or s['queue']
    out = [dict(label='{} \u00b7 {}'.format(*k) if k[1] else k[0], model=k[0], effort=k[1], **g)
        for k, g in groups.items()]
    out.sort(key=lambda g: (not g['queue'], g['label'] == 'model unknown', -len(g['sessions'])))
    return out


def sample_tokens(sample):
    """Token counts of a usage sample: {'all', 'nocache', 'output'}, overall ('all'), on Fable
    models ('fable_all'), and the same for the manual-work share ('manual_all',
    'fable_manual_all'; the tokens of the turns a prompt Tom typed started, `tokens_manual`
    in the sample; 0 in samples from before that split)."""
    out = {}
    for field, prefix in (('tokens', ''), ('tokens_manual', 'manual_')):
        for model, t in (sample.get(field) or {}).items():
            if not isinstance(t, dict):
                continue
            vals = {k: t.get(k) if isinstance(t.get(k), (int, float)) else 0
                for k in ('input', 'output', 'cache_read', 'cache_creation')}
            measures = {
                'all': sum(vals.values()),
                'nocache': vals['input'] + vals['output'] + vals['cache_creation'],
                'output': vals['output'],
            }
            fable = 'fable' in (model or '').lower()
            for k, v in measures.items():
                out[prefix + k] = out.get(prefix + k, 0) + v
                out['fable_' + prefix + k] = out.get('fable_' + prefix + k, 0) + (v if fable else 0)
    return out


def token_value(tokens, prefix, measure, work):
    """A sample's tokens for one plot: `prefix` '' or 'fable_', `measure` a TOKEN_MEASURES key,
    `work` 'queue' (the manual-work share taken out) or 'all'."""
    v = tokens.get(prefix + measure, 0)
    if work == 'queue':
        v -= tokens.get(prefix + 'manual_' + measure, 0)
    return max(v, 0)


def account_key(host, account):
    return account['id'] if account and account.get('id') else 'host:' + host


def read_usage(shared, now):
    """The usage samples of the sessions active in the longest plot window.

    Returns:
        {'sessions': [{'host', 'session_id', 'samples': [{'epoch', 'tokens' (sample_tokens),
        'plan_usage', 'model'}]}], 'hosts': {host: account key}, 'accounts': {key: {'label',
        'name', 'plan', 'hosts', 'joined': {host: first plan-usage reading}}}}.
    """
    udir = os.path.join(shared, 'usage')
    try:
        names = os.listdir(udir)
    except OSError:
        names = []
    cutoff = now - (max(USAGE_DAYS) + 1) * 86400
    sessions, latest_account, first_reading = [], {}, {}
    for name in sorted(names):
        m = USAGE_FILE.match(name)
        if not m:
            continue
        path = os.path.join(udir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                continue
        except OSError:
            continue
        host, samples, queue_session, work_mode = m.group(1), [], False, None
        for line in (read_text(path) or '').splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or not isinstance(d.get('epoch'), (int, float)):
                continue
            pu = d.get('plan_usage') if isinstance(d.get('plan_usage'), dict) else None
            samples.append({'epoch': d['epoch'], 'tokens': sample_tokens(d), 'plan_usage': pu,
                'model': d.get('model')})
            queue_session = queue_session or bool(d.get('queue_session'))
            if d.get('work_mode') in ('queue', 'manual'):
                work_mode = d['work_mode']
            acct = d.get('account') if isinstance(d.get('account'), dict) else None
            if d['epoch'] >= latest_account.get(host, (0, None))[0]:
                latest_account[host] = (d['epoch'], acct)
            if pu and (host not in first_reading or d['epoch'] < first_reading[host]['epoch']):
                first_reading[host] = dict(pu, epoch=d['epoch'])
        if samples:
            samples.sort(key=lambda s: s['epoch'])
            sessions.append({'host': host, 'session_id': m.group(2), 'samples': samples,
                'queue_session': queue_session, 'work_mode': work_mode})
    hosts, accounts = {}, {}
    for host in sorted({s['host'] for s in sessions}):
        acct = latest_account.get(host, (0, None))[1]
        key = account_key(host, acct)
        hosts[host] = key
        a = accounts.setdefault(key, {'name': (acct or {}).get('name'),
            'plan': (acct or {}).get('plan'), 'hosts': [], 'joined': {}})
        a['hosts'].append(host)
        if host in first_reading:
            a['joined'][host] = first_reading[host]
    labels = {}
    for key, a in accounts.items():
        if key.startswith('host:'):
            label = 'unknown account ({})'.format(key[5:])
        else:
            label = (a['name'] or 'account ' + key[:6]) + (' ({})'.format(a['plan'])
                if a['plan'] else '')
        labels.setdefault(label, []).append(key)
        a['label'] = label
    for label, keys in labels.items():
        if len(keys) > 1:
            for key in keys:
                accounts[key]['label'] += ' #' + key[:4]
    return {'sessions': sessions, 'hosts': hosts, 'accounts': accounts}


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
        'usage': read_usage(shared, now),
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
:root { color-scheme: light dark; }
:root[data-theme="light"] { color-scheme: light; }
:root[data-theme="dark"] { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 1.35rem; margin: 0 0 2px; }
h2 { font-size: 1.05rem; margin: 28px 0 8px; }
h3 { font-size: .95rem; margin: 16px 0 6px; }
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
.b-manual { color: var(--orange); } .b-queue { color: var(--muted); }
details.chat summary { color: var(--fg); font-size: inherit; list-style: none; }
details.chat summary::-webkit-details-marker { display: none; }
details.chat summary::before { content: "\25B8"; color: var(--muted); margin-right: 6px; }
details.chat[open] summary::before { content: "\25BE"; }
details.chat pre { max-height: none; }
.theme a { color: var(--muted); text-decoration: none; margin: 0 2px; }
.theme a.on { color: var(--fg); font-weight: 600; }
details { margin-top: 4px; }
summary { cursor: pointer; color: var(--muted); font-size: .82rem; }
pre { white-space: pre-wrap; word-break: break-word; font-size: .8rem; margin: 6px 0 0;
  max-height: 340px; overflow: auto; background: var(--chip); padding: 8px; border-radius: 6px; }
.legend { color: var(--muted); font-size: .82rem; margin-top: 8px; }
.legend span { margin-right: 14px; }
code { font-size: .85em; }
.model + .model { margin-top: 6px; }
.controls { font-size: .85rem; color: var(--muted); margin: 0 0 10px; }
.controls a { color: var(--blue); text-decoration: none; }
.controls b { color: var(--fg); }
.controls .sep { margin: 0 8px; }
.plotbox { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 10px 10px; margin-bottom: 12px; }
.plotbox h3 { font-size: .92rem; margin: 2px 0 4px; }
.plot { width: 100%; height: auto; display: block; }
.plot .grid { stroke: var(--line); stroke-width: 1; }
.plot .axis { fill: var(--muted); font-size: 11px; }
.plot .ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; }
.plot .series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.plot .dash { stroke-dasharray: 6 4; }
.plot .pt { stroke: var(--card); stroke-width: 1.5; }
.plot .hit { fill: transparent; stroke: none; }
.keys { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: .84rem; margin: 6px 2px 0; }
.sw { display: inline-block; width: 18px; height: 0; border-top: 2px solid; vertical-align: middle;
  margin-right: 6px; }
.sw.dash { border-top-style: dashed; }
.note { color: var(--muted); font-size: .82rem; margin: 6px 2px 0; }
"""
CSS += ':root { ' + ' '.join('--s{}: {};'.format(i + 1, c[0])
    for i, c in enumerate(SERIES_COLORS)) + ' }\n'
CSS += ('@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ' + ' '.join(
    '--s{}: {};'.format(i + 1, c[1]) for i, c in enumerate(SERIES_COLORS)) + ' } }\n')
CSS += ':root[data-theme="dark"] { ' + ' '.join('--s{}: {};'.format(i + 1, c[1])
    for i, c in enumerate(SERIES_COLORS)) + ' }\n'


# applies the reader's theme choice (localStorage 'theme': light | dark; absent = follow the
# system or browser setting) before the page paints, and wires the theme links
THEME_SCRIPT = (
    "(function(){var k='theme',r=document.documentElement;function g(){try{return localStorage."
    "getItem(k)}catch(e){return null}}function ap(t){if(t==='light'||t==='dark')r.setAttribute("
    "'data-theme',t);else r.removeAttribute('data-theme');var l=document.querySelectorAll("
    "'.theme a');for(var i=0;i<l.length;i++)l[i].className=(l[i].getAttribute('data-theme')==="
    "(t||'auto'))?'on':''}ap(g());document.addEventListener('DOMContentLoaded',function(){ap(g());"
    "var l=document.querySelectorAll('.theme a');for(var i=0;i<l.length;i++)l[i].addEventListener("
    "'click',function(e){e.preventDefault();var t=this.getAttribute('data-theme');try{if(t==="
    "'auto')localStorage.removeItem(k);else localStorage.setItem(k,t)}catch(x){}ap(t)})});"
    # every option link reloads the page and so does the refresh every 30 s: keep the reader's
    # place (the usage controls sit far down the page), unless the URL names an anchor
    "var sk='scroll:'+location.pathname;function sv(){try{sessionStorage.setItem(sk,String("
    "window.pageYOffset||0))}catch(e){}}window.addEventListener('pagehide',sv);document."
    "addEventListener('DOMContentLoaded',function(){if(location.hash)return;try{var y="
    "sessionStorage.getItem(sk);if(y!==null)window.scrollTo(0,parseInt(y,10)||0)}catch(e){}})"
    "})();")


def esc(x):
    return html.escape('' if x is None else str(x))


def fmt_time(t, now, split=False):
    """'15:04 (12 min ago)' today, 'Nov 4 17:42 (2.5 d ago)' otherwise.

    Args:
        split: put how long ago on a second line, without the parentheses.
    """
    if not t:
        return '<span class="muted">-</span>'
    d = datetime.fromtimestamp(t)
    today = datetime.fromtimestamp(now).date()
    stamp = d.strftime('%H:%M') if d.date() == today else d.strftime('%b %-d %H:%M')
    return '<span class="nw" title="{}">{}{}<span class="muted">{}</span></span>'.format(
        d.strftime('%Y-%m-%d %H:%M:%S'), stamp, '<br>' if split else ' ',
        ago(now - t) if split else '({})'.format(ago(now - t)))


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

# ---------------------------------------------------------------- usage plots

def parse_opts(query):
    """The usage plots' options from a query string: days, tokens (measure), group."""
    q = parse_qs(query or '')
    def one(k):
        v = q.get(k)
        return v[-1] if v else None
    try:
        days = int(one('days') or USAGE_DEFAULT_DAYS)
    except ValueError:
        days = USAGE_DEFAULT_DAYS
    scope = one('scope')
    if scope not in {k for k, _, _ in SCOPES}:
        # links from before the three-way choice carried work= and sessions=
        scope = ('all' if one('sessions') == 'all' else 'queue_sessions') if one('work') == 'all' \
            else 'queue_work'
    return {
        'days': days if days in USAGE_DAYS else USAGE_DEFAULT_DAYS,
        'tokens': one('tokens') if one('tokens') in TOKEN_MEASURES else 'all',
        'group': 'account' if one('group') == 'account' else 'host',
        'scope': scope,
    }


OPT_KEYS = ('days', 'tokens', 'group', 'scope')
# what the usage plots count: key, label, explanation
SCOPES = (
    ('queue_work', 'queue work only', 'the queue sessions (those that watch chat.md), without the '
        "turns that prompts typed into a session started (Tom's requests), until the agent "
        'marks the session back as queue work'),
    ('queue_sessions', 'queue sessions only', 'everything the queue sessions did, manual work '
        'included; other sessions started in the experiments folder are left out'),
    ('all', 'all work', 'every session started in the experiments folder, all of its work'),
)


def opts_href(opts, **change):
    o = dict(opts, **change)
    return '?' + urlencode([(k, o[k]) for k in OPT_KEYS])


def nice_max(v):
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= v * (1 - 1e-9):
            return m * e
    return 10 * e


def fmt_tokens(v):
    for div, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'k')):
        if abs(v) >= div:
            x = v / div
            return ('{:.0f}' if x >= 100 else '{:.1f}').format(x).replace('.0', '') + suffix
    return '{:.0f}'.format(v)


def fmt_pct(v):
    return '{:.0f}%'.format(v)


def time_ticks(t0, t1):
    """Tick times and labels for a time axis from t0 to t1 (at most about 8, on local hours)."""
    span_h = (t1 - t0) / 3600
    step = next((s for s in (1, 2, 3, 6, 12, 24, 48) if span_h / s <= 8), 48)
    d = datetime.fromtimestamp(t0).replace(minute=0, second=0, microsecond=0)
    ticks = []
    while d.timestamp() <= t1:
        ts = d.timestamp()
        if ts >= t0:
            if step < 24 and d.hour % step == 0:
                ticks.append((ts, d.strftime('%b %-d') if d.hour == 0 else d.strftime('%H:%M')))
            elif step >= 24 and d.hour == 0 and d.toordinal() % (step // 24) == 0:
                ticks.append((ts, d.strftime('%b %-d')))
        d += timedelta(hours=1)
    return ticks


def step_series(sessions, t0, t1):
    """Cumulative total over sessions within [t0, t1], as step points relative to t0.

    Args:
        sessions: per session, its (epoch, cumulative value) samples, sorted.

    Returns:
        [(t, value - value at t0)], from t0 to t1.
    """
    events, cur = [], []
    for i, ss in enumerate(sessions):
        v0 = 0
        for t, v in ss:
            if t <= t0:
                v0 = v
            elif t <= t1:
                events.append((t, i, v))
        cur.append(v0)
    base = sum(cur)
    pts = [(t0, 0)]
    for t, i, v in sorted(events):
        cur[i] = v
        pts.append((t, sum(cur) - base))
    pts.append((t1, sum(cur) - base))
    return pts


def svg_plot(series, t0, t1, y_fmt, y_max=None, ref=None, label='', height=230):
    """An inline SVG line plot over time.

    Args:
        series: [{'label', 'color' (a CSS color), 'points' [(t, y)], 'step' (draw as a step
            function), 'dash'}].
        y_fmt: formats a y value for the axis and the hover text.
        y_max: top of the y axis (default: a round number above the data).
        ref: y value of a dotted reference line (e.g. 100 %).
    """
    W, H, ml, mr, mt, mb = 900, height, 58, 14, 8, 24
    ys = [y for s in series for _, y in s['points']]
    top = y_max or nice_max(max(ys) if ys else 1)
    def X(tv):
        return ml + (tv - t0) / max(t1 - t0, 1) * (W - ml - mr)
    def Y(v):
        return mt + (1 - min(max(v, 0), top) / top) * (H - mt - mb)
    o = ['<svg class="plot" viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(
        W, H, esc(label))]
    for i in range(5):
        v = top * i / 4
        o.append('<line class="grid" x1="{0}" x2="{1}" y1="{2:.1f}" y2="{2:.1f}"/>'
            '<text class="axis" x="{3}" y="{4:.1f}" text-anchor="end">{5}</text>'.format(
            ml, W - mr, Y(v), ml - 6, Y(v) + 4, esc(y_fmt(v))))
    for tv, text in time_ticks(t0, t1):
        o.append('<text class="axis" x="{:.1f}" y="{}" text-anchor="middle">{}</text>'.format(
            X(tv), H - 6, esc(text)))
    if ref is not None and ref <= top:
        o.append('<line class="ref" x1="{0}" x2="{1}" y1="{2:.1f}" y2="{2:.1f}"/>'.format(
            ml, W - mr, Y(ref)))
    for s in series:
        pts = s['points']
        if not pts:
            continue
        xy = []
        for j, (tv, v) in enumerate(pts):
            if s.get('step') and j:
                xy.append((X(tv), Y(pts[j - 1][1])))
            xy.append((X(tv), Y(v)))
        o.append('<polyline class="series{}" style="stroke: {}" points="{}"><title>{}</title>'
            '</polyline>'.format(' dash' if s.get('dash') else '', s['color'],
            ' '.join('{:.1f},{:.1f}'.format(*q) for q in xy), esc(s['label'])))
        marks = pts[1:-1] if s.get('step') else pts
        if len(marks) > 150:
            marks = marks[::math.ceil(len(marks) / 150)] + [marks[-1]]
        for tv, v in marks:
            tip = '{}: {} at {}'.format(s['label'], y_fmt(v),
                datetime.fromtimestamp(tv).strftime('%b %-d %H:%M'))
            o.append('<g><circle class="hit" cx="{0:.1f}" cy="{1:.1f}" r="8"/><circle class="pt" '
                'cx="{0:.1f}" cy="{1:.1f}" r="3.5" style="fill: {2}"/><title>{3}</title></g>'.format(
                X(tv), Y(v), s['color'], esc(tip)))
    o.append('</svg>')
    keys = ''.join('<span><span class="sw{}" style="border-color: {}"></span>{}</span>'.format(
        ' dash' if s.get('dash') else '', s['color'], esc(s['label'])) for s in series)
    return ''.join(o) + '<div class="keys">{}</div>'.format(keys)


def series_color(i):
    return 'var(--s{})'.format(i + 1) if i < len(SERIES_COLORS) else 'var(--grey)'


def options_table(sessions, t0, t1, opts):
    """Per host, what the two usage options leave out over the plot window: the manual-work
    share ("queue work only") and the tokens of sessions that are not queue sessions ("queue
    sessions only"), beside the total and what the plots show under the current options. The
    manual share is usually a few per cent of a host's tokens, so without this table toggling
    either option moves nothing visible on the plots."""
    def total(ss, work):
        return sum(step_series([[(x['epoch'], token_value(x['tokens'], '', opts['tokens'], work))
            for x in s['samples']]], t0, t1)[-1][1] for s in ss)

    def share(v, whole):
        if not whole:
            return fmt_tokens(v)
        return '{} <span class="muted">({:.1f} %)</span>'.format(fmt_tokens(v), 100.0 * v / whole)
    queue_work, queue_only = opts['scope'] == 'queue_work', opts['scope'] != 'all'
    rows = []
    for h in sorted({s['host'] for s in sessions}):
        ss = [s for s in sessions if s['host'] == h]
        sq = [s for s in ss if s.get('queue_session')]
        so = [s for s in ss if not s.get('queue_session')]
        everything = total(ss, 'all')
        manual = everything - total(ss, 'queue')
        shown = total(sq if queue_only else ss, 'queue' if queue_work else 'all')
        rows.append('<tr><td>{}</td><td class="nw">{} / {}</td><td class="nw">{}</td>'
            '<td class="nw">{}</td><td class="nw">{}</td><td class="nw">{}</td></tr>'.format(
            esc(h), len(sq), len(so), fmt_tokens(everything), share(manual, everything),
            share(total(so, 'all'), everything), fmt_tokens(shown)))
    return ('<div class="wrap"><table><tr><th>host</th><th>sessions<br>queue / other</th>'
        '<th>tokens, {}<br>last {} d, everything</th><th>manual work<br>(left out by "queue work '
        'only")</th><th>in other sessions<br>(left out unless "all work")</th>'
        '<th>shown in the plots</th></tr>{}</table></div>'.format(
        esc(TOKEN_MEASURES[opts['tokens']]), opts['days'], ''.join(rows)))


def usage_html(state, now, opts):
    """The usage section: what the options leave out per host, tokens per host (or account),
    tokens on Fable models, plan usage."""
    u = state.get('usage') or {}
    sessions, hosts, accounts = u.get('sessions') or [], u.get('hosts') or {}, u.get(
        'accounts') or {}
    o = []
    w = o.append
    links = []
    for d in USAGE_DAYS:
        links.append('<b>{} d</b>'.format(d) if d == opts['days'] else
            '<a href="{}">{} d</a>'.format(esc(opts_href(opts, days=d)), d))
    tok = [('<b>{}</b>' if k == opts['tokens'] else '<a href="{1}">{0}</a>').format(
        esc(v), esc(opts_href(opts, tokens=k))) for k, v in TOKEN_MEASURES.items()]
    grouped = opts['group'] == 'account'
    def checkbox(on, href, text, title):
        return '<a href="{}" title="{}">{} {}</a>'.format(esc(href), esc(title),
            '&#9745;' if on else '&#9744;', text)
    box = checkbox(grouped, opts_href(opts, group='host' if grouped else 'account'),
        'group hosts by Claude account', 'sum the hosts that use one claude.ai account')
    queue_work = opts['scope'] == 'queue_work'
    queue_only = opts['scope'] != 'all'
    work = ' | '.join(('<b title="{2}">{0}</b>' if k == opts['scope']
        else '<a href="{1}" title="{2}">{0}</a>').format(esc(label), esc(opts_href(opts, scope=k)),
        esc(title)) for k, label, title in SCOPES)
    w('<div class="controls">last {}<span class="sep">&middot;</span>tokens: {}'
        '<span class="sep">&middot;</span>work: {}<span class="sep">&middot;</span>{}</div>'.format(
        ' | '.join(links), ' | '.join(tok), work, box))
    if not sessions:
        w('<p class="muted">No usage samples yet (<code>usage/</code> in the shared directory, '
            'written by presence_hook.py every 15 min).</p>')
        return ''.join(o)
    t1, t0 = now, now - opts['days'] * 86400
    w(options_table(sessions, t0, t1, opts))
    if queue_only:
        sessions = [s for s in sessions if s.get('queue_session')]
        hosts = {h: k for h, k in hosts.items() if any(s['host'] == h for s in sessions)}
        accounts = {k: dict(a, hosts=[h for h in a['hosts'] if h in hosts])
            for k, a in accounts.items() if k in set(hosts.values())}
    if not sessions:
        w('<p class="muted">No queue sessions in this window; uncheck "queue sessions only" to '
            'see the other sessions.</p>')
        return ''.join(o)
    host_order = sorted(hosts)
    acct_order = sorted(accounts, key=lambda k: accounts[k]['label'].lower())
    def acct_label(key):
        a = accounts[key]
        return '{}: {}'.format(a['label'], ', '.join(a['hosts']))
    if grouped:
        entities = [(acct_label(k), series_color(i), [s for s in sessions
            if hosts.get(s['host']) == k]) for i, k in enumerate(acct_order)]
    else:
        entities = [(h, series_color(i), [s for s in sessions if s['host'] == h])
            for i, h in enumerate(host_order)]

    def token_plot(prefix, title, label):
        series = []
        for name, color, ss in entities:
            pts = step_series([[(x['epoch'], token_value(x['tokens'], prefix, opts['tokens'],
                'queue' if queue_work else 'all')) for x in s['samples']] for s in ss], t0, t1)
            if any(v for _, v in pts):
                series.append({'label': '{} ({})'.format(name, fmt_tokens(pts[-1][1])),
                    'color': color, 'points': pts, 'step': True})
        w('<div class="plotbox"><h3>{}</h3>'.format(esc(title)))
        if series:
            w(svg_plot(series, t0, t1, fmt_tokens, label=label))
        else:
            w('<p class="muted">none in this window</p>')
        w('</div>')
        return series

    what = TOKEN_MEASURES[opts['tokens']] + {'queue_work': ', queue work',
        'queue_sessions': ', queue sessions', 'all': ', all sessions'}[opts['scope']]
    w('<p class="note">Queue and manual work are told apart by time alone: every message in a '
        'session\'s transcript, and in its subagents\' transcripts, counts for the mode in effect '
        'when it was written, from the typed prompt that started a segment to the agent\'s '
        '<code>--mark queue</code> or the next typed queue command. So a subagent started for queue '
        'work that runs on during one of Tom\'s requests counts as manual work for that time, and '
        'the reverse; and a request whose end was never marked counts as manual until the next '
        'typed queue command. Ends set after the fact (<code>--mark queue --at</code>) are as exact '
        'as the record they were taken from.</p>')
    token_plot('', 'Tokens per {} ({}, cumulative over the window)'.format(
        'Claude account' if grouped else 'host', what), 'tokens over time')
    token_plot('fable_', 'Tokens on Fable models per {} ({}, cumulative)'.format(
        'Claude account' if grouped else 'host', what), 'Fable tokens over time')

    # plan usage, per account
    series, rows = [], []
    for i, key in enumerate(acct_order):
        a = accounts[key]
        readings = {}
        for s in sessions:
            if hosts.get(s['host']) != key:
                continue
            for x in s['samples']:
                pu = x.get('plan_usage')
                if pu:
                    readings[pu.get('captured_epoch') or x['epoch']] = pu
        pts = sorted(readings.items())
        color = series_color(i)
        for window, name, dash in (('seven_day', 'weekly', False), ('five_hour', '5-hour', True)):
            wp = [(tv, r[window]['used_percentage']) for tv, r in pts
                if t0 <= tv <= t1 and isinstance(r.get(window), dict)]
            if wp:
                series.append({'label': '{}, {}'.format(a['label'], name), 'color': color,
                    'points': wp, 'dash': dash})
        last = pts[-1] if pts else None
        def cell(window):
            if not last or not isinstance(last[1].get(window), dict):
                return '<span class="muted">-</span>'
            r = last[1][window]
            reset = r.get('resets_at')
            return '{}{}'.format(fmt_pct(r['used_percentage']),
                ' <span class="muted">(resets {})</span>'.format(esc(datetime.fromtimestamp(
                reset).strftime('%a %H:%M'))) if isinstance(reset, (int, float)) else '')
        joined = '<br>'.join('{} {}{} <span class="muted">({})</span>'.format(esc(h),
            fmt_pct(j['seven_day']['used_percentage']) + ' weekly' if isinstance(
            j.get('seven_day'), dict) else '', ', ' + fmt_pct(j['five_hour']['used_percentage'])
            + ' 5-hour' if isinstance(j.get('five_hour'), dict) else '',
            esc(datetime.fromtimestamp(j['epoch']).strftime('%b %-d %H:%M')))
            for h, j in sorted(a['joined'].items()))
        tokens = sum(step_series([[(x['epoch'], token_value(x['tokens'], '', opts['tokens'],
            'queue' if queue_work else 'all')) for x in s['samples']]], t0, t1)[-1][1]
            for s in sessions if hosts.get(s['host']) == key)
        rows.append('<tr><td><span class="sw" style="border-color: {}"></span>{}</td><td>{}</td>'
            '<td class="nw">{}</td><td class="nw">{}</td><td>{}</td><td class="nw">{}</td>'
            '<td>{}</td></tr>'.format(color, esc(a['label']), esc(', '.join(a['hosts'])),
            cell('five_hour'), cell('seven_day'), fmt_time(last[0], now) if last else
            '<span class="muted">-</span>', fmt_tokens(tokens),
            joined or '<span class="muted">-</span>'))
    w('<div class="plotbox"><h3>Plan usage per Claude account (% of the 5-hour and weekly '
        'limits)</h3>')
    if series:
        w(svg_plot(series, t0, t1, fmt_pct, y_max=100, ref=100, label='plan usage over time'))
    else:
        w('<p class="muted">no plan-usage readings in this window</p>')
    w('<div class="wrap" style="margin-top: 10px"><table><tr><th>account</th><th>hosts</th>'
        '<th>5-hour</th><th>weekly</th><th>read</th><th>tokens ({}, last {} d)</th>'
        '<th>when each host joined</th></tr>{}</table></div>'.format(esc(what), opts['days'],
        ''.join(rows)))
    w('<p class="note">Tokens are summed from the session transcripts (subagents included) every '
        '15 min. Queue work: the transcript tags every prompt a person typed, so the turns from '
        'such a prompt to the next one count as manual work unless the prompt was a queue command '
        '(the join command, the session-title tick, an acknowledgement) or the agent has since '
        'run <code>presence_hook.py --mark queue</code>; task notifications, scheduled prompts and '
        'peer messages keep the mode in effect. Queue sessions are those that have run '
        '<code>chat_watch.sh</code>. Plan usage is what Claude Code&#39;s status line last received (its '
        '<code>rate_limits</code>: the 5-hour and weekly limits of claude.ai Pro, Max and Ultra '
        'plans); Claude Code reports no separate Fable or Opus limit, so the Fable plot shows '
        'tokens only. "When each host joined" is the first reading of each host&#39;s '
        'sessions. A session&#39;s tokens from before its first sample (it was running when '
        'sampling began) appear as one step at that sample.</p></div>')
    return ''.join(o)


def render(state, refresh, opts=None):
    """The dashboard page.

    Args:
        opts: the usage plots' options (parse_opts); defaults if None.
    """
    opts = opts or parse_opts('')
    now = state['generated']
    out = []
    w = out.append
    w('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width, initial-scale=1">')
    w('<meta name="color-scheme" content="light dark">')
    if refresh:
        w('<meta http-equiv="refresh" content="{:d}">'.format(int(refresh)))
    # the theme follows the system or browser setting (prefers-color-scheme) unless the reader
    # picked one here; the choice is kept in the browser and applied before the first paint
    w('<script>' + THEME_SCRIPT + '</script>')
    w('<title>{}</title><style>{}</style></head><body><main>'.format(esc(TITLE), CSS))
    w('<h1>{}</h1>'.format(esc(TITLE)))
    w('<div class="sub">served by {} &middot; generated {}{} &middot; <span class="theme">theme: '
        '<a href="#" data-theme="auto">auto</a> | <a href="#" data-theme="light">light</a> | '
        '<a href="#" data-theme="dark">dark</a></span><br>'.format(
        esc(state['server_host']), datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
        ' &middot; reloads every {:d} s'.format(int(refresh)) if refresh else ''))
    where = ('<code>{}</code> on tensor <span class="muted">(here {})</span>'.format(
        esc(state['tensor_path']), esc(state['shared'])) if state.get('tensor_path')
        else '<code>{}</code>'.format(esc(state['shared'])))
    w('data from {}: <code>presence/</code> and <code>usage/</code> (hooks), '
        '<code>&lt;host&gt;_status.md</code>, <code>chat.md</code>, '
        '<code>experiment_priorities.md</code></div>'.format(where))
    for e in state['errors']:
        w('<div class="err">{}</div>'.format(esc(e)))

    # hosts
    w('<h2>Hosts</h2><div class="wrap"><table><tr><th>host</th><th>sessions (hooks)</th>'
        '<th>model<br>effort</th><th>declared status</th><th>doing now</th>'
        '<th>blocked on</th><th>running</th><th>status updated</th><th>last chat</th>'
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
        models = ''.join('<div class="model nw{}" title="{}">{}{}{}</div>'.format(
            '' if g['queue'] else ' muted', esc(', '.join(g['sessions'])
                + ('' if g['queue'] else ' (not on queue)')), esc(g['model']),
            ' <span class="muted">&times;{}</span>'.format(len(g['sessions']))
                if len(g['sessions']) > 1 else '',
            '<br>' + esc(g['effort']) if g['effort'] else '') for g in h['models'])
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
            fmt_time(dec.get('mtime'), now, True), fmt_time(h['last_chat'], now, True),
            fmt_time(h['joined'], now, True)))
    w('</table></div>')
    w('<div class="legend">sessions: ' + ''.join('<span>{} {}</span>'.format(badge(v), esc(t))
        for v, t in VIEW_TEXT.items()) + '</div>')

    # running experiments
    w('<h2>Running</h2>')
    if state['running']:
        w('<div class="wrap"><table><tr><th>ID</th><th>name</th><th>state</th><th>host</th>'
            '<th>started</th></tr>')
        for r in state['running']:
            w('<tr><td class="nw">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                *(esc(r.get(k)) for k in ('id', 'name', 'state', 'host', 'started'))))
        w('</table></div>')
    else:
        w('<p class="muted">Nothing running.</p>')

    # queue
    counts = state['index_counts']
    w('<h2>Queue</h2><p class="muted">Experiment index: {}</p>'.format(
        esc(', '.join('{} {}'.format(n, k) for k, n in sorted(counts.items()))) or 'empty'))
    for title, rows in (('Approved, unclaimed', state['approved']),
            ('Proposed (awaiting approval)', state['proposed'])):
        if not rows:
            continue
        w('<h3>{}</h3><div class="wrap"><table><tr><th>ID</th><th>name</th><th>state</th>'
            '<th>host</th><th>started</th></tr>'.format(esc(title)))
        for r in rows:
            w('<tr><td class="nw">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                *(esc(r.get(k)) for k in ('id', 'name', 'state', 'host', 'started'))))
        w('</table></div>')
    if state['claims']:
        w('<h3>Work claims</h3><div class="wrap"><table><tr>' + ''.join(
            '<th>{}</th>'.format(esc(k)) for k in state['claims'][0]) + '</tr>')
        for r in state['claims']:
            w('<tr>' + ''.join('<td>{}</td>'.format(esc(v)) for v in r.values()) + '</tr>')
        w('</table></div>')

    # usage
    w('<h2>Usage</h2>')
    w(usage_html(state, now, opts))

    # chat
    w('<h2>Recent chat.md entries</h2><p class="sub">click an entry for the whole message, '
        'again to fold it</p><div class="wrap"><table>'
        '<tr><th>posted</th><th>host</th><th>message</th></tr>')
    for e in state['recent_chat']:
        text = e.get('text') or e['first_line']
        if text and text != e['first_line']:
            body = '<details class="chat"><summary>{}</summary><pre>{}</pre></details>'.format(
                esc(e['first_line']), esc(text))
        else:
            body = esc(e['first_line'])
        w('<tr><td class="nw">{}</td><td>{}</td><td>{}</td></tr>'.format(
            fmt_time(e['time'], now), esc(e['host']), body))
    w('</table></div>')

    # sessions
    w('<h2>Claude sessions</h2>')
    if state['sessions']:
        w('<div class="wrap"><table><tr><th>host</th><th>session</th><th>state</th>'
            '<th>model &middot; effort</th><th>tokens</th><th>work</th><th>last event</th>'
            '<th>chat heartbeat</th><th>started</th><th>detail</th></tr>')
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
            tok = s.get('tokens') if isinstance(s.get('tokens'), dict) else {}
            tokens = '<span title="output: {}; manual work: {}">{}</span>'.format(
                fmt_tokens(tok.get('output') or 0), fmt_tokens(tok.get('manual') or 0),
                fmt_tokens(tok['total'])) if tok.get('total') else '<span class="muted">-</span>'
            if s.get('work_mode') == 'manual':
                work = '<span class="badge b-manual" title="a prompt typed into the session '
                'started this; its turns count as manual work until the agent runs '
                'presence_hook.py --mark queue">manual</span>'
                if s.get('work_mode_since'):
                    work += ' <span class="muted">since {}</span>'.format(
                        fmt_time(local_epoch(s['work_mode_since'][:16].replace('T', ' ')), now))
            elif s.get('work_mode') == 'queue':
                work = '<span class="badge b-queue">queue</span>'
            else:
                work = '<span class="muted">-</span>'
            w('<tr><td>{}</td><td title="{}">{}</td><td>{}</td><td class="nw">{}</td>'
                '<td class="nw">{}</td><td class="nw">{}</td><td>{}</td><td>{}</td><td>{}</td>'
                '<td>{}</td></tr>'.format(
                esc(s['host']), esc(s.get('session_id')), name,
                badge(s['view'], VIEW_TEXT.get(s['view'])),
                esc(s['model_label']) or '<span class="muted">-</span>', tokens, work, last, hb,
                fmt_time(s.get('started_epoch'), now), esc(detail)))
        w('</table></div>')
    else:
        w('<p class="muted">No session has reported through the hooks yet.</p>')

    w('</main></body></html>')
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
            url = urlparse(self.path)
            path = url.path
            if path in ('/', '/index.html'):
                body = render(monitor.get(), refresh, parse_opts(url.query)).encode()
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

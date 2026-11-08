#!/usr/bin/env python3
"""Claude Code status-line command for queue sessions: keeps the plan usage it receives.

Registered as `statusLine` in the experiments folder's .claude/settings.json. Claude Code runs
it at session start, after each assistant message and every `refreshInterval` seconds, with a
JSON object on stdin whose `rate_limits` (claude.ai Pro / Max / Ultra subscriptions, after the
first API response) give the 5-hour and weekly usage of the plan
(https://code.claude.com/docs/en/statusline). This script saves that object, with the time, to
the local cache presence_hook.py uses (`<cache>/<session_id>.statusline.json`), from which the
hook's usage samples take the plan usage; it never touches the NAS, so the status line cannot
hang on it.

What it prints is the status line: the user's own `statusLine` command from their user
settings, if they have one (a project's `statusLine` replaces the user's, so this one runs it
in its place), else the plan usage, e.g. "5h 23% (to 21:00) · week 41% (to Mon 09:00)".
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

CACHE_DIR = os.environ.get('KC_PRESENCE_CACHE') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
    'claude-queue-presence')


def save(d):
    sid = d.get('session_id') or ''
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', sid):
        return
    rec = {
        'epoch': time.time(),
        'rate_limits': d.get('rate_limits'),
        'model': (d.get('model') or {}).get('id'),
        'cost_usd': (d.get('cost') or {}).get('total_cost_usd'),
        'version': d.get('version'),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, sid + '.statusline.json')
    tmp = '{}.{}.tmp'.format(path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, path)


def user_command():
    """The `statusLine` command of the user's own settings, unless it is this script."""
    cfg = os.environ.get('CLAUDE_CONFIG_DIR') or os.path.expanduser('~/.claude')
    try:
        with open(os.path.join(cfg, 'settings.json')) as f:
            sl = json.load(f).get('statusLine') or {}
    except (OSError, ValueError, AttributeError):
        return None
    cmd = sl.get('command') if isinstance(sl, dict) else None
    if not cmd or 'statusline_capture.py' in cmd:
        return None
    return cmd


def when(epoch):
    try:
        t = datetime.fromtimestamp(float(epoch))
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return t.strftime('%H:%M') if t.date() == datetime.now().date() else t.strftime('%a %H:%M')


def usage_line(d):
    parts = []
    for key, label in (('five_hour', '5h'), ('seven_day', 'week')):
        w = (d.get('rate_limits') or {}).get(key) or {}
        pct = w.get('used_percentage')
        if isinstance(pct, (int, float)):
            reset = when(w.get('resets_at'))
            parts.append('{} {:.0f}%{}'.format(label, pct, ' (to {})'.format(reset) if reset
                else ''))
    return ' · '.join(parts)


def main():
    raw = sys.stdin.read()
    try:
        d = json.loads(raw)
    except ValueError:
        return
    if not isinstance(d, dict):
        return
    try:
        save(d)
    except OSError:
        pass
    cmd = user_command()
    if cmd:
        try:
            r = subprocess.run(['bash', '-c', cmd], input=raw, capture_output=True, text=True,
                timeout=4)
            sys.stdout.write(r.stdout)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    line = usage_line(d)
    if line:
        print(line)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if os.environ.get('KC_PRESENCE_DEBUG'):
            raise
    sys.exit(0)

"""Tests for the queue monitor: presence_hook.py, the chat_watch.sh heartbeat and server.py.

Run with `./venv/bin/python -m pytest scripts/claude_pair_remapping_experiments/monitor-server`
from the al_analysis root (standard library only, besides pytest).
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server  # noqa: E402

HOST = socket.gethostname().split('.')[0]
QUEUE_DIR = '/home/x/src/al_analysis/scripts/claude_pair_remapping_experiments'
TRANSCRIPT = ('/home/x/.claude/projects/-home-x-src-al-analysis-scripts-claude-pair-'
    'remapping-experiments/{}.jsonl')


@pytest.fixture
def dirs(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    env = dict(os.environ, KC_SHARED_DIR=str(shared), KC_PRESENCE_CACHE=str(tmp_path / 'cache'),
        KC_PRESENCE_DEBUG='1', CLAUDE_PID='4242')
    return shared, env


def hook(env, event, sid='s1', cwd=QUEUE_DIR, **fields):
    d = dict(session_id=sid, hook_event_name=event, cwd=cwd,
        transcript_path=TRANSCRIPT.format(sid))
    d.update(fields)
    r = subprocess.run([sys.executable, '-I', str(HERE / 'presence_hook.py')],
        input=json.dumps(d), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ''
    time.sleep(0.01)


def record(shared, sid='s1'):
    p = shared / 'presence' / '{}__{}.json'.format(HOST, sid)
    return json.loads(p.read_text()) if p.exists() else None


def test_hook_lifecycle(dirs):
    shared, env = dirs
    hook(env, 'SessionStart', source='startup', model='claude-opus-5-5')
    r = record(shared)
    assert r['state'] == 'idle' and r['model'] == 'claude-opus-5-5'
    assert r['host'] == HOST and r['claude_pid'] == 4242
    started = r['started']

    hook(env, 'UserPromptSubmit', prompt='__set_session_title__ hal-2')
    assert record(shared)['label'] == 'hal-2'
    hook(env, 'PreToolUse', tool_name='Bash')
    assert record(shared)['state'] == 'working'
    # unchanged state: only the local cache gets the event (see the throttle test)
    cache = json.loads((Path(env['KC_PRESENCE_CACHE']) / 's1.json').read_text())
    assert cache['last_tool'] == 'Bash'
    hook(env, 'Notification', notification_type='permission_prompt',
        message='Claude needs your permission to use Bash')
    r = record(shared)
    assert r['state'] == 'waiting_permission' and 'permission' in r['detail']
    hook(env, 'PostToolUse', tool_name='Bash')
    assert record(shared)['state'] == 'working'
    hook(env, 'Stop')
    assert record(shared)['state'] == 'idle'
    hook(env, 'StopFailure', error='rate_limit')
    r = record(shared)
    assert r['state'] == 'error' and 'rate_limit' in r['detail']
    hook(env, 'SessionEnd', reason='prompt_input_exit')
    r = record(shared)
    assert r['state'] == 'ended' and r['end_reason'] == 'prompt_input_exit'
    assert r['started'] == started


def assistant_line(model, effort=None, sidechain=False):
    d = dict(type='assistant', isSidechain=sidechain, message=dict(model=model, role='assistant'))
    if effort:
        d['effort'] = effort
    return json.dumps(d, separators=(',', ':')) + '\n'


def test_hook_records_model_and_effort(dirs, tmp_path):
    shared, env = dirs
    transcript = tmp_path / 'proj-claude-pair-remapping-experiments' / 's4.jsonl'
    transcript.parent.mkdir()
    transcript.write_text('{"type":"user","timestamp":"2026-11-07T10:00:00Z"}\n')
    kw = dict(sid='s4', transcript_path=str(transcript))
    # before the first reply, only the SessionStart input has the model
    hook(env, 'SessionStart', source='startup', model='claude-fable-5-1', **kw)
    r = record(shared, 's4')
    assert r['model'] == 'claude-fable-5-1' and 'effort' not in r

    with open(transcript, 'a') as f:
        f.write(assistant_line('claude-fable-5-1', 'max'))
        # a large tool result after the last assistant message
        f.write(json.dumps(dict(type='user', content='x' * 200000)) + '\n')
    hook(env, 'PreToolUse', tool_name='Bash', **kw)
    r = record(shared, 's4')
    assert r['model'] == 'claude-fable-5-1' and r['effort'] == 'max'

    # switching model / effort mid-session (the Fable usage cap) reaches the shared file at
    # once; subagent and synthetic (API error) messages do not count
    with open(transcript, 'a') as f:
        f.write(assistant_line('claude-opus-5-5', 'xhigh'))
        f.write(assistant_line('claude-haiku-5-5', 'low', sidechain=True))
        f.write(assistant_line('<synthetic>'))
        f.write('{"type":"assistant","message":{"mod')
    hook(env, 'PostToolUse', tool_name='Bash', **kw)
    r = record(shared, 's4')
    assert r['model'] == 'claude-opus-5-5' and r['effort'] == 'xhigh'

    # a model without effort levels drops the old one
    with open(transcript, 'a') as f:
        f.write('\n' + assistant_line('claude-haiku-4-5'))
    hook(env, 'Stop', **kw)
    r = record(shared, 's4')
    assert r['model'] == 'claude-haiku-4-5' and 'effort' not in r


def test_hook_ignores_other_folders_and_follows_cd(dirs):
    shared, env = dirs
    other = dict(cwd='/home/x/src/olfsysm', sid='s2')
    hook(env, 'SessionStart', source='startup', transcript_path='/x/-home-x-src-olfsysm/s2.jsonl',
        **other)
    assert record(shared, 's2') is None
    # a recorded session stays recorded after it cds elsewhere
    hook(env, 'SessionStart', source='startup')
    hook(env, 'PreToolUse', tool_name='Read', cwd='/home/x/src/olfsysm')
    assert record(shared)['state'] == 'working'


def test_hook_throttles_shared_writes(dirs):
    shared, env = dirs
    hook(env, 'PreToolUse', tool_name='Bash')
    p = shared / 'presence' / '{}__s1.json'.format(HOST)
    m0 = p.stat().st_mtime_ns
    hook(env, 'PostToolUse', tool_name='Bash')
    hook(env, 'PreToolUse', tool_name='Read')
    assert p.stat().st_mtime_ns == m0
    cache = json.loads((Path(env['KC_PRESENCE_CACHE']) / 's1.json').read_text())
    assert cache['events'] == 3 and cache['last_tool'] == 'Read'


def test_chat_watch_heartbeat(dirs):
    shared, env = dirs
    chat = shared / 'chat.md'
    chat.write_text('## 2026-11-06 10:00 {}\nhello\n'.format(HOST))
    env = dict(env, CLAUDE_CODE_SESSION_ID='s9')
    proc = subprocess.Popen(['bash', str(HERE.parent / 'chat_watch.sh'), str(chat), '1'],
        env=env, stdout=subprocess.PIPE, text=True)
    watch = shared / 'presence' / '{}__s9.watch'.format(HOST)
    for _ in range(50):
        if watch.exists():
            break
        time.sleep(0.1)
    w = json.loads(watch.read_text())
    assert w['state'] == 'listening' and w['claude_pid'] == 4242 and w['interval_s'] == 1
    with open(chat, 'a') as f:
        f.write('## 2026-11-06 10:05 otherhost\nhi\n')
    out, _ = proc.communicate(timeout=10)
    assert 'otherhost' in out and proc.returncode == 0
    assert json.loads(watch.read_text())['state'] == 'exited'


STATUS = """---
state: blocked
since: 2026-11-06 14:09
eta: -
blocked_on: Tom (OK on the footnote)
now: idle; summary published
next: the footnote
---
# hal status
free text
"""

PRIORITIES = """# Experiment priorities

## Experiment index

| ID | name | state | host | proposed | approved | started | completed |
|---|---|---|---|---|---|---|---|
| E1 | the baseline | completed | hal | report | report | 11-04 | 11-04 (#1) |
| E37 | something new | running | atlas | 11-06, atlas | 11-06, Tom | 11-06 15:00 | |
| E38 | next thing | approved | | 11-06, hal | 11-06, Tom | | |

## Work claims

| item | host | started | ETA | scope / stub |
|---|---|---|---|---|
| a data pull | atlas | 15:00 | 16:00 | data/x |

## Priority list
"""


def make_shared(shared, now):
    (shared / 'hal_status.md').write_text(STATUS)
    (shared / 'atlas_status.md').write_text('# atlas status\n\n## Now\n- Idle (nothing).\n')
    (shared / 'experiment_priorities.md').write_text(PRIORITIES)
    (shared / 'chat.md').write_text('## 2026-11-04 17:42 atlas\nfirst\n\n'
        '## 2026-11-04 17:45 hal\nsecond\n\n## 2026-11-06 15:52 atlas\nthird\n')
    pdir = shared / 'presence'
    pdir.mkdir(exist_ok=True)

    def rec(host, sid, state, age, **kw):
        d = dict(host=host, session_id=sid, state=state, last_event='x',
            last_event_epoch=now - age, started_epoch=now - 3600, **kw)
        (pdir / '{}__{}.json'.format(host, sid)).write_text(json.dumps(d))

    def watch(host, sid, state, age):
        (pdir / '{}__{}.watch'.format(host, sid)).write_text(json.dumps(dict(
            host=host, session_id=sid, state=state, epoch=now - age, interval_s=60)))

    rec('hal', 'a', 'idle', 600, model='claude-fable-5-1', effort='xhigh',
        session_title='al_analysis modeling - hal')
    watch('hal', 'a', 'listening', 30)
    rec('hal', 'b', 'waiting_permission', 60, detail='needs permission',
        model='claude-opus-5-5', effort='high')
    rec('atlas', 'c', 'working', 3600, model='claude-opus-5-5', effort='xhigh')
    rec('atlas', 'd', 'ended', 100, model='claude-sonnet-4-5-20250929')
    rec('atlas', 'e', 'idle', 30 * 24 * 3600)
    watch('atlas', 'f', 'listening', 10)

    # usage samples: hal and atlas on one account, viki on another without plan usage
    udir = shared / 'usage'
    udir.mkdir(exist_ok=True)

    def usage(host, sid, account, points):
        with open(udir / '{}__{}.jsonl'.format(host, sid), 'w') as f:
            for age, fable, opus, pu in points:
                tokens = {}
                if fable:
                    tokens['claude-fable-5-1'] = dict(input=10, output=fable, cache_read=10 * fable,
                        cache_creation=0)
                if opus:
                    tokens['claude-opus-5-5'] = dict(input=10, output=opus, cache_read=10 * opus,
                        cache_creation=0)
                f.write(json.dumps(dict(epoch=now - age, host=host, session_id=sid,
                    account=account, tokens=tokens, plan_usage=pu)) + '\n')

    tom = dict(id='aaaa1111bbbb', name='Tom', plan='Max')
    def pu(age, five, week):
        return dict(five_hour=dict(used_percentage=five, resets_at=now + 3600),
            seven_day=dict(used_percentage=week, resets_at=now + 86400), captured_epoch=now - age)
    usage('hal', 'a', tom, [(7200, 1000, 0, pu(7200, 10, 40)), (3600, 3000, 500, pu(3600, 25, 42)),
        (60, 5000, 2000, pu(60, 30, 45))])
    usage('atlas', 'c', tom, [(5000, 0, 1000, None), (100, 0, 4000, pu(100, 31, 45))])
    usage('viki', 'v', dict(id='cccc2222dddd', name='Yang', plan='Pro'),
        [(3000, 200, 0, None), (200, 900, 0, None)])


def test_collect(tmp_path):
    now = time.time()
    make_shared(tmp_path, now)
    st = server.collect(str(tmp_path), now)
    assert not st['errors']
    views = {s['session_id']: s['view'] for s in st['sessions']}
    assert views == {'a': 'listening', 'b': 'needs you', 'c': 'silent', 'd': 'ended',
        'f': 'listening'}
    hosts = {h['host']: h for h in st['hosts']}
    assert hosts['hal']['session_state'] == 'needs you'
    assert hosts['hal']['declared']['state'] == 'blocked'
    assert hosts['hal']['declared']['fields']['blocked_on'] == 'Tom (OK on the footnote)'
    assert hosts['atlas']['declared']['state'] is None
    assert hosts['atlas']['declared']['now'] == 'Idle (nothing).'
    assert [r['id'] for r in hosts['atlas']['running']] == ['E37']
    assert hosts['atlas']['claims'][0]['item'] == 'a data pull'
    assert hosts['atlas']['chat_posts'] == 2
    assert [r['id'] for r in st['approved']] == ['E38']
    assert st['index_counts'] == {'completed': 1, 'running': 1, 'approved': 1}
    assert st['recent_chat'][0]['first_line'] == 'third'
    # queue sessions first; ended sessions and heartbeats without hook records do not count
    assert [(g['label'], g['queue']) for g in hosts['hal']['models']] == [
        ('Fable 5.1 \u00b7 xhigh', True), ('Opus 5.5 \u00b7 high', False)]
    assert [(g['label'], g['sessions']) for g in hosts['atlas']['models']] == [
        ('model unknown', ['f']), ('Opus 5.5 \u00b7 xhigh', ['c'])]
    assert hosts['hal']['models'][0]['model'] == 'Fable 5.1'
    assert hosts['hal']['models'][0]['effort'] == 'xhigh'
    html = server.render(st, 30)
    assert 'needs you' in html and 'E37' in html and '<script' not in html
    # Hosts table: model and effort on two lines, as are the times and their recency
    hosts_table = html[html.index('<h2>Hosts'):html.index('<h2>Running')]
    assert 'Fable 5.1<br>xhigh' in hosts_table and '<th>last chat</th>' in hosts_table
    assert ' ago</span>' in hosts_table and ' ago)' not in hosts_table
    assert 'Fable 5.1 \u00b7 xhigh' in html
    assert 'data from <code>{}</code>'.format(tmp_path) in html


def test_usage_collect_and_render(tmp_path):
    now = time.time()
    make_shared(tmp_path, now)
    st = server.collect(str(tmp_path), now)
    u = st['usage']
    assert u['hosts'] == {'atlas': 'aaaa1111bbbb', 'hal': 'aaaa1111bbbb', 'viki': 'cccc2222dddd'}
    tom = u['accounts']['aaaa1111bbbb']
    assert tom['label'] == 'Tom (Max)' and tom['hosts'] == ['atlas', 'hal']
    assert tom['joined']['hal']['seven_day']['used_percentage'] == 40
    assert tom['joined']['atlas']['five_hour']['used_percentage'] == 31
    hal = next(s for s in u['sessions'] if s['host'] == 'hal')
    assert hal['samples'][-1]['tokens']['output'] == 7000
    assert hal['samples'][-1]['tokens']['fable_output'] == 5000
    assert hal['samples'][-1]['tokens']['all'] == 5000 * 11 + 2000 * 11 + 20
    # cumulative steps within the window, relative to its start
    pts = server.step_series([[(now - 7200, 10), (now - 60, 30)], [(now - 100, 5)]],
        now - 3600, now)
    assert pts == [(now - 3600, 0), (now - 100, 5), (now - 60, 25), (now, 25)]
    html = server.render(st, 30, server.parse_opts('days=1&tokens=output'))
    order = [html.index('<h2>' + h) for h in ('Hosts', 'Running', 'Queue', 'Usage',
        'Recent chat.md entries', 'Claude sessions')]
    assert order == sorted(order)
    usage_html = html[html.index('<h2>Usage'):html.index('<h2>Recent chat')]
    assert usage_html.count('<svg class="plot"') == 3 and '<script' not in html
    # by host: hal's output over the last day (it started 2 h ago, so all of it) is 7k
    assert '>hal (7k)<' in usage_html and '>viki (900)<' in usage_html
    assert '&#9744; group hosts by Claude account' in usage_html
    grouped = server.render(st, 30, server.parse_opts('group=account&tokens=output'))
    grouped = grouped[grouped.index('<h2>Usage'):grouped.index('<h2>Recent chat')]
    assert '&#9745; group hosts by Claude account' in grouped
    assert 'Tom (Max): atlas, hal (11k)' in grouped
    assert 'Tom (Max), weekly' in grouped and 'Tom (Max), 5-hour' in grouped
    assert '45%' in grouped and 'Yang (Pro)' in grouped
    # the sessions table shows each session's tokens from its presence record
    assert server.parse_opts('days=99&tokens=bogus') == dict(days=3, tokens='all', group='host')


def test_model_names():
    names = {
        'claude-opus-5-5': 'Opus 5.5',
        'claude-fable-5-1': 'Fable 5.1',
        'claude-sonnet-4-5-20250929': 'Sonnet 4.5',
        'claude-opus-4-20250514': 'Opus 4',
        'claude-3-5-haiku-20241022': 'Haiku 3.5',
        'claude-opus-4-6[1m]': 'Opus 4.6 (1M)',
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0': 'Sonnet 4.5',
        'some-other-model': 'some-other-model',
        None: None,
    }
    assert {m: server.model_name(m) for m in names} == names
    assert server.model_label('claude-opus-5-5', 'xhigh') == 'Opus 5.5 \u00b7 xhigh'
    assert server.model_label('claude-haiku-4-5', None) == 'Haiku 4.5'
    assert server.tensor_path('/local/tensor/Tom/claude_kc_remapping') == 'Tom/claude_kc_remapping'
    assert server.tensor_path('/tmp/Tom/claude_kc_remapping') is None


def test_missing_shared_dir(tmp_path):
    st = server.collect(str(tmp_path / 'missing'))
    assert st['errors'] and st['hosts'] == []
    server.render(st, 0)


def test_http(tmp_path):
    make_shared(tmp_path, time.time())
    httpd = ThreadingHTTPServer(('127.0.0.1', 0),
        server.make_handler(server.Monitor(str(tmp_path)), 30))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = 'http://127.0.0.1:{}'.format(httpd.server_address[1])
    try:
        page = urllib.request.urlopen(base + '/').read().decode()
        assert '<title>Model experiments</title>' in page
        assert '<svg class="plot"' in urllib.request.urlopen(
            base + '/?days=7&group=account&tokens=output').read().decode()
        state = json.loads(urllib.request.urlopen(base + '/api/state').read())
        assert {h['host'] for h in state['hosts']} == {'hal', 'atlas'}
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + '/../etc/passwd')
    finally:
        httpd.shutdown()


def usage_line(mid, model, out, sidechain=False):
    return json.dumps(dict(type='assistant', isSidechain=sidechain, message=dict(id=mid,
        model=model, usage=dict(input_tokens=1, output_tokens=out, cache_read_input_tokens=100,
        cache_creation_input_tokens=10)))) + '\n'


def test_session_tokens(tmp_path):
    import presence_hook
    transcript = tmp_path / 's.jsonl'
    # one message written as three content-block lines, and a tool result between messages
    transcript.write_text(usage_line('m1', 'claude-fable-5-1', 5) + usage_line('m1',
        'claude-fable-5-1', 7) + '{"type":"user","message":{"content":"x"}}\n' + usage_line(
        'm2', 'claude-fable-5-1', 3) + usage_line('m2', 'claude-fable-5-1', 3) + usage_line(
        'e', '<synthetic>', 99) + '{"type":"assistant","message":{"id":"m3","us')
    sub = tmp_path / 's' / 'subagents'
    sub.mkdir(parents=True)
    (sub / 'agent-1.jsonl').write_text(usage_line('x1', 'claude-opus-5-5', 20))
    files = {}
    tok = presence_hook.session_tokens(str(transcript), files)
    assert tok == {'claude-fable-5-1': dict(input=2, output=10, cache_read=200, cache_creation=20),
        'claude-opus-5-5': dict(input=1, output=20, cache_read=100, cache_creation=10)}
    # incremental: the cut-off line completes, m3 counts once, m2 stays counted once
    with open(transcript, 'a') as f:
        f.write('age":{"output_tokens":4}}}\n' + usage_line('m3', 'claude-fable-5-1', 4))
    files = json.loads(json.dumps(files))
    tok = presence_hook.session_tokens(str(transcript), files)
    assert tok['claude-fable-5-1']['output'] == 14


def test_hook_writes_usage_samples(dirs, tmp_path):
    shared, env = dirs
    home = tmp_path / 'home'
    home.mkdir()
    (home / '.claude.json').write_text(json.dumps(dict(oauthAccount=dict(accountUuid='u-1',
        displayName='Tom', planDisplayName='Max', emailAddress='x@y'))))
    env = dict(env, HOME=str(home))
    transcript = tmp_path / 'proj-claude-pair-remapping-experiments' / 's5.jsonl'
    transcript.parent.mkdir()
    transcript.write_text(usage_line('m1', 'claude-fable-5-1', 50))
    kw = dict(sid='s5', transcript_path=str(transcript))
    path = shared / 'usage' / '{}__s5.jsonl'.format(HOST)
    def samples():
        return [json.loads(x) for x in path.read_text().splitlines()]

    hook(env, 'SessionStart', source='startup', **kw)
    s = samples()
    assert len(s) == 1 and s[0]['reason'] == 'SessionStart' and s[0]['plan_usage'] is None
    assert s[0]['tokens'] == {'claude-fable-5-1': dict(input=1, output=50, cache_read=100,
        cache_creation=10)}
    assert s[0]['account']['name'] == 'Tom' and s[0]['account']['plan'] == 'Max'
    assert len(s[0]['account']['id']) == 12 and 'x@y' not in path.read_text()
    hook(env, 'PreToolUse', tool_name='Bash', **kw)
    assert len(samples()) == 1
    # the status line's first plan-usage reading is sampled at the next event
    r = subprocess.run([sys.executable, '-I', str(HERE / 'statusline_capture.py')],
        input=json.dumps(dict(session_id='s5', rate_limits=dict(five_hour=dict(
        used_percentage=23.5, resets_at=time.time() + 3600), seven_day=dict(used_percentage=41.2,
        resets_at=time.time() + 86400)))), env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith('5h 24% (to ') and 'week 41%' in r.stdout
    hook(env, 'PostToolUse', tool_name='Bash', **kw)
    s = samples()
    assert len(s) == 2 and s[1]['plan_usage']['seven_day']['used_percentage'] == 41.2
    rec = json.loads((Path(env['KC_PRESENCE_CACHE']) / 's5.json').read_text())
    assert rec['tokens']['total'] == 161 and rec['tokens']['output'] == 50
    # chat_watch.sh's --sample: nothing until 15 min have passed
    run = [sys.executable, '-I', str(HERE / 'presence_hook.py'), '--sample', 's5']
    subprocess.run(run, env=env, check=True)
    assert len(samples()) == 2
    state = Path(env['KC_PRESENCE_CACHE']) / 's5.usage.json'
    st = json.loads(state.read_text())
    st['sampled_epoch'] -= 16 * 60
    state.write_text(json.dumps(st))
    subprocess.run(run, env=env, check=True)
    s = samples()
    assert len(s) == 3 and s[2]['reason'] == 'watch'
    hook(env, 'SessionEnd', reason='other', **kw)
    assert samples()[-1]['reason'] == 'SessionEnd'


def test_statusline_runs_users_own_command(tmp_path):
    cfg = tmp_path / 'cfg'
    cfg.mkdir()
    (cfg / 'settings.json').write_text(json.dumps(dict(statusLine=dict(type='command',
        command='jq -r .session_id 2>/dev/null || python3 -c "print(\'mine\')"'))))
    env = dict(os.environ, CLAUDE_CONFIG_DIR=str(cfg), KC_PRESENCE_CACHE=str(tmp_path / 'c'))
    r = subprocess.run([sys.executable, '-I', str(HERE / 'statusline_capture.py')],
        input=json.dumps(dict(session_id='s6')), env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() in ('s6', 'mine')
    assert json.loads((tmp_path / 'c' / 's6.statusline.json').read_text())['rate_limits'] is None


def test_hook_reads_custom_title(dirs, tmp_path):
    shared, env = dirs
    transcript = tmp_path / 'proj-claude-pair-remapping-experiments' / 's3.jsonl'
    (tmp_path / 'proj-claude-pair-remapping-experiments' / 's3').mkdir(parents=True)
    transcript.write_text('{"type": "mode"}\n{"type": "user", '
        '"timestamp": "2026-11-05T00:31:50.890Z"}\n')
    (transcript.parent / 's3' / 'custom-title.json').write_text(
        '{"customTitle": "al_analysis modeling - hal"}')
    hook(env, 'Stop', sid='s3', transcript_path=str(transcript))
    r = record(shared, 's3')
    assert r['session_title'] == 'al_analysis modeling - hal'
    assert r['started_epoch'] == pytest.approx(1793838710.89)


def test_shared_dir_on_either_mount(tmp_path, monkeypatch):
    import presence_hook
    monkeypatch.delenv('KC_SHARED_DIR', raising=False)
    local, mnt = tmp_path / 'local', tmp_path / 'mnt'
    shared = mnt / 'Tom' / 'claude_kc_remapping'
    shared.mkdir(parents=True)
    monkeypatch.setattr(presence_hook, 'TENSOR_MOUNTS', (str(local), str(mnt)))
    monkeypatch.setattr(server, 'TENSOR_MOUNTS', (str(local), str(mnt)))
    assert presence_hook.shared_dir() == str(shared)
    assert server.default_shared() == str(shared)
    monkeypatch.setattr(presence_hook, 'TENSOR_MOUNTS', (str(local),))
    assert presence_hook.shared_dir() is None

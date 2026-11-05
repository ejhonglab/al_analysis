"""Tests of the model-provenance functions in `al_analysis.mb_model` (which olfsysm build
and which commits of our code produced an output). None of these compare version
information against reference outputs: the repro tests are meant to work across versions.
"""
import glob
import hashlib
import json
from pathlib import Path
import re

import pytest

from al_analysis import mb_model
from al_analysis.mb_model import (olfsysm_build_info, olfsysm_checkout_info, olfsysm_is_clean,
    olfsysm_clean_problems, require_clean_olfsysm, repo_info, model_provenance,
    write_model_provenance, OlfsysmBuildError, ALLOW_DIRTY_OLFSYSM_ENV_VAR,
    model_provenance_name
)


def test_olfsysm_build_info_keys():
    info = olfsysm_build_info()
    for k in ('version', 'git_hash', 'git_dirty', 'git_untracked', 'git_diff', 'source_sha256',
            'build_time', 'build_flags', 'compiler', 'build_host', 'module_file',
            'legacy_build'):
        assert k in info, k
    assert info['git_dirty'] in (-1, 0, 1)
    if not info['legacy_build']:
        assert re.fullmatch(r'[0-9a-f]{40}', info['git_hash']) or info['git_hash'] == ''
        assert re.fullmatch(r'[0-9a-f]{64}', info['source_sha256'])
        # a clean build has no diff; a dirty one must carry it
        assert (info['git_dirty'] == 1) == (len(info['git_diff']) > 0)


def test_source_sha_matches_checkout_when_clean():
    """The sha baked into the extension is over the checkout's sources and headers."""
    info = olfsysm_build_info()
    checkout = olfsysm_checkout_info()
    if info['legacy_build'] or not olfsysm_is_clean():
        pytest.skip('needs a clean olfsysm build matching its checkout')
    root = Path(checkout['path'])
    sources = ['libolfsysm/src/olfsysm.cpp', 'bindings/python/pyolfsysm.cpp']
    headers = sorted(str(Path(p).relative_to(root)) for pat in
        ('libolfsysm/api/**/*', 'libolfsysm/include/**/*')
        for p in glob.glob(str(root / pat), recursive=True) if Path(p).is_file()
    )
    h = hashlib.sha256()
    for rel in sources + headers:
        h.update(rel.encode()); h.update(b'\0')
        h.update((root / rel).read_bytes()); h.update(b'\0')
    assert h.hexdigest() == info['source_sha256']


def test_checkout_info_keys():
    c = olfsysm_checkout_info()
    for k in ('path', 'editable', 'git_hash', 'dirty', 'untracked', 'remote'):
        assert k in c, k
    if c['git_hash'] is not None:
        assert re.fullmatch(r'[0-9a-f]{40}', c['git_hash'])


def test_is_clean_consistent_with_problems():
    assert olfsysm_is_clean() == (len(olfsysm_clean_problems()) == 0)


def test_require_clean_error_and_override(monkeypatch):
    monkeypatch.setattr(mb_model, 'olfsysm_clean_problems', lambda: ['pretend it is dirty'])
    monkeypatch.setattr(mb_model, 'STRICT_CLEAN_OLFSYSM', False)
    monkeypatch.delenv(ALLOW_DIRTY_OLFSYSM_ENV_VAR, raising=False)
    with pytest.raises(OlfsysmBuildError) as e:
        require_clean_olfsysm()
    msg = str(e.value)
    assert 'pretend it is dirty' in msg
    # the message must say how to run anyway during development
    assert f'export {ALLOW_DIRTY_OLFSYSM_ENV_VAR}=1' in msg
    # the override turns it into a warning
    monkeypatch.setenv(ALLOW_DIRTY_OLFSYSM_ENV_VAR, '1')
    with pytest.warns(UserWarning):
        require_clean_olfsysm()
    # but not where a clean build is always required
    with pytest.raises(OlfsysmBuildError) as e:
        require_clean_olfsysm(allow_env_override=False, context='this test')
    assert 'this test' in str(e.value)
    monkeypatch.setattr(mb_model, 'STRICT_CLEAN_OLFSYSM', True)
    with pytest.raises(OlfsysmBuildError):
        require_clean_olfsysm()


def test_require_clean_passes_when_clean(monkeypatch):
    monkeypatch.setattr(mb_model, 'olfsysm_clean_problems', lambda: [])
    require_clean_olfsysm()
    require_clean_olfsysm(allow_env_override=False)


def test_repo_info_this_repo():
    info = repo_info(mb_model)
    assert info['git_hash'] is not None and re.fullmatch(r'[0-9a-f]{40}', info['git_hash'])
    assert isinstance(info['dirty'], bool)
    assert (info['diff'] is not None) == info['dirty']
    assert repo_info('/')['git_hash'] is None


def test_model_provenance_and_write(tmp_path):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=['a', 'b'], columns=list('xyz'))
    prov = model_provenance(orn_deltas=df, scale=1.5, nothing=None)
    for k in ('timestamp', 'host', 'platform', 'cpu', 'python', 'packages', 'olfsysm',
            'al_analysis', 'hong2p', 'inputs'):
        assert k in prov, k
    assert prov['inputs']['orn_deltas']['sha256'] == model_provenance(orn_deltas=df
        )['inputs']['orn_deltas']['sha256']
    assert prov['inputs']['scale'] == 1.5 and prov['inputs']['nothing'] is None
    assert set(prov['olfsysm']) >= {'build', 'checkout', 'clean', 'problems'}
    assert prov['olfsysm']['clean'] == olfsysm_is_clean()
    path = write_model_provenance(tmp_path, prov)
    assert path.name == model_provenance_name
    assert json.loads(path.read_text())['python'] == prov['python']

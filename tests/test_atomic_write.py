import json
import os
import stat
import sys

import pytest

from astro_notebooks import image_selector
from astro_notebooks.image_selector import _atomic_write_json

posix_only = pytest.mark.skipif(
    sys.platform == 'win32',
    reason='POSIX permission bits are not meaningful on Windows')


@pytest.fixture
def umask_022():
    """Run the test under umask 022 and restore the previous umask after."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _mode(path):
    """Return only the permission bits of ``path``."""
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_write_round_trip_leaves_no_tmp_file(tmp_path):
    """The contents are written as JSON and no temporary file is left over."""
    target = tmp_path / 'image_selection.json'
    contents = {'a.fit': True, 'b.fit': False}

    _atomic_write_json(target, contents)

    assert json.loads(target.read_text()) == contents
    assert [p.name for p in tmp_path.iterdir()] == ['image_selection.json']


@posix_only
def test_atomic_write_new_file_mode_follows_umask(tmp_path, umask_022):
    """A new file gets the umask-derived mode, not mkstemp's owner-only 0600."""
    target = tmp_path / 'image_selection.json'

    _atomic_write_json(target, {'a.fit': True})

    assert _mode(target) == 0o644


@posix_only
def test_atomic_write_rewrite_mode_follows_umask(tmp_path, umask_022):
    """Rewriting a file this function created keeps the umask-derived mode."""
    target = tmp_path / 'image_selection.json'

    _atomic_write_json(target, {'a.fit': True})
    _atomic_write_json(target, {'a.fit': False})

    assert _mode(target) == 0o644
    assert json.loads(target.read_text()) == {'a.fit': False}


@posix_only
def test_atomic_write_preserves_existing_mode(tmp_path, umask_022):
    """Rewriting a file keeps the permissions that file already had."""
    target = tmp_path / 'image_selection.json'
    target.write_text('{}')
    target.chmod(0o664)

    _atomic_write_json(target, {'a.fit': True})

    assert _mode(target) == 0o664
    assert json.loads(target.read_text()) == {'a.fit': True}


def test_atomic_write_failure_removes_tmp_and_keeps_old_file(tmp_path,
                                                            monkeypatch):
    """If json.dump raises, the temp file is removed and the old file kept."""
    target = tmp_path / 'image_selection.json'
    target.write_text('{"a.fit": false}')

    def failing_dump(*args, **kwargs):
        """Stand in for json.dump and fail the way a full disk would."""
        raise OSError('No space left on device')

    monkeypatch.setattr(image_selector.json, 'dump', failing_dump)

    with pytest.raises(OSError, match='No space left'):
        _atomic_write_json(target, {'a.fit': True})

    assert [p.name for p in tmp_path.iterdir()] == ['image_selection.json']
    assert target.read_text() == '{"a.fit": false}'


def test_atomic_write_syncs_before_replace(tmp_path, monkeypatch):
    """The data is fsynced to disk before the rename makes it visible."""
    target = tmp_path / 'image_selection.json'
    calls = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        """Record the fsync call, then do the real thing."""
        calls.append('fsync')
        real_fsync(fd)

    def recording_replace(src, dst):
        """Record the replace call, then do the real thing."""
        calls.append('replace')
        real_replace(src, dst)

    monkeypatch.setattr(image_selector.os, 'fsync', recording_fsync)
    monkeypatch.setattr(image_selector.os, 'replace', recording_replace)

    _atomic_write_json(target, {'a.fit': True})

    assert calls == ['fsync', 'replace']

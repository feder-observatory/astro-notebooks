"""What ImageSelect does when its selection file cannot be read or written.

The point of all of these is that a selection is never lost *silently*:
either the file on disk is left exactly as it was, or the user is told in
the widget itself that a choice was not saved.
"""
import builtins
import errno
import json
import os

import pytest

from astro_notebooks import image_selector
from astro_notebooks.image_selector import SELECTION_FILE_NAME, ImageSelect

from .conftest import N_IMAGES

ALL_NAMES = [f"image-{i:03d}.fit" for i in range(N_IMAGES)]


@pytest.fixture
def saved_selection(fits_dir):
    """A data directory whose saved selection rejects the first image.

    Returns the directory and the exact bytes of the selection file, so a
    test can check that the file was not touched.
    """
    w = ImageSelect(directory=fits_dir)
    w._selectors[0]._selector.value = False
    path = fits_dir / SELECTION_FILE_NAME
    assert json.loads(path.read_text())[ALL_NAMES[0]] is False
    return fits_dir, path.read_bytes()


def test_failed_save_in_observer_is_shown_not_raised(fits_dir, mocker):
    """A save that fails when a box is clicked is reported in the widget.

    An exception raised in the checkbox observer would only reach the
    kernel log, so it must be caught and shown in ``message`` instead. The
    widget keeps trying: the next change that saves successfully writes
    the whole selection, including the choice that failed, and clears the
    message.
    """
    w = ImageSelect(directory=fits_dir)
    assert w.message == ''
    real_write = image_selector._atomic_write_json
    mocker.patch.object(image_selector, '_atomic_write_json',
                        side_effect=OSError(errno.ENOSPC, 'disk full'))

    # must not raise
    w._selectors[0]._selector.value = False

    assert 'could NOT be saved' in w.message
    assert w._message.layout.display != 'none'
    assert w._can_save
    on_disk = json.loads((fits_dir / SELECTION_FILE_NAME).read_text())
    assert on_disk[ALL_NAMES[0]] is True

    mocker.patch.object(image_selector, '_atomic_write_json',
                        side_effect=real_write)
    w._selectors[1]._selector.value = False

    assert w.message == ''
    assert w._message.layout.display == 'none'
    on_disk = json.loads((fits_dir / SELECTION_FILE_NAME).read_text())
    assert on_disk[ALL_NAMES[0]] is False
    assert on_disk[ALL_NAMES[1]] is False


def test_read_error_leaves_selection_file_alone(saved_selection,
                                                monkeypatch):
    """An I/O error reading the selection file never overwrites the file.

    The file may hold a perfectly good selection (a stale handle on a
    network disk, say), so the widget opens with everything included,
    warns, says in its message that nothing is being saved, and neither
    construction nor later clicks nor an explicit ``save_selection`` write
    to the file.
    """
    fits_dir, original = saved_selection
    path = fits_dir / SELECTION_FILE_NAME

    def flaky_open(file, *args, **kwargs):
        """Fail for the selection file only; open anything else."""
        if os.fspath(file) == os.fspath(path):
            raise OSError(errno.ESTALE, 'Stale file handle')
        return builtins.open(file, *args, **kwargs)

    # The fault is transient: it is gone again once the widget is made.
    with monkeypatch.context() as patch:
        patch.setattr(image_selector, 'open', flaky_open, raising=False)
        with pytest.warns(UserWarning, match='will not be saved'):
            w = ImageSelect(directory=fits_dir)

    assert w._can_save is False
    assert w.selected_files == ALL_NAMES
    assert 'NOT being saved' in w.message
    assert path.read_bytes() == original

    w._selectors[1]._selector.value = False
    w.save_selection()
    assert path.read_bytes() == original
    assert 'NOT being saved' in w.message
    assert not (fits_dir / (SELECTION_FILE_NAME + '.bak')).exists()


def test_malformed_selection_file_is_kept_as_backup(fits_dir):
    """A selection file that is not a mapping is moved aside, not erased.

    The original bytes end up in ``image_selection.json.bak``, a fresh
    all-included file is written, and saving stays switched on.
    """
    path = fits_dir / SELECTION_FILE_NAME
    bad = b'["image-000.fit", "image-001.fit"]'
    path.write_bytes(bad)

    with pytest.warns(UserWarning, match=r'moved to .*\.bak'):
        w = ImageSelect(directory=fits_dir)

    assert (fits_dir / (SELECTION_FILE_NAME + '.bak')).read_bytes() == bad
    assert json.loads(path.read_text()) == {name: True for name in ALL_NAMES}
    assert w._can_save
    assert w.message == ''


def test_empty_selection_file_is_kept_as_backup(fits_dir):
    """A zero-length selection file, as a crash can leave, is set aside.

    It is treated like any other file that is not JSON, and the widget
    goes on saving normally afterwards.
    """
    path = fits_dir / SELECTION_FILE_NAME
    path.write_bytes(b'')

    with pytest.warns(UserWarning, match=r'moved to .*\.bak'):
        w = ImageSelect(directory=fits_dir)

    assert (fits_dir / (SELECTION_FILE_NAME + '.bak')).read_bytes() == b''
    w._selectors[0]._selector.value = False
    assert json.loads(path.read_text())[ALL_NAMES[0]] is False


def test_backup_failure_switches_saving_off(fits_dir, mocker):
    """If a malformed file cannot be moved aside it is not overwritten.

    Without a backup the only safe thing to do is to leave the file as it
    is, exactly as for a file that could not be read.
    """
    path = fits_dir / SELECTION_FILE_NAME
    path.write_bytes(b'{not json')
    mocker.patch.object(image_selector.os, 'replace',
                        side_effect=OSError(errno.EACCES, 'denied'))

    with pytest.warns(UserWarning, match='will not be saved'):
        w = ImageSelect(directory=fits_dir)

    assert w._can_save is False
    assert path.read_bytes() == b'{not json'


@pytest.fixture
def read_only_dir(saved_selection):
    """The ``saved_selection`` directory, made read-only for the test.

    Thumbnails and a selection file already exist, so the only thing the
    widget still wants to write is the selection. The mode is restored
    afterwards so that pytest can clean up ``tmp_path``.
    """
    fits_dir, original = saved_selection
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        pytest.skip('root can write to a read-only directory')
    fits_dir.chmod(0o555)
    yield fits_dir, original
    fits_dir.chmod(0o755)


def test_read_only_directory_still_opens(read_only_dir):
    """The widget can be used to look at images it has no right to change.

    The saved selection is restored, a warning and the in-widget message
    say that nothing will be saved, and a click neither raises nor changes
    the file.
    """
    fits_dir, original = read_only_dir

    with pytest.warns(UserWarning, match='cannot be saved'):
        w = ImageSelect(directory=fits_dir)

    assert w._can_save is False
    assert w.selected_files == ALL_NAMES[1:]
    assert 'NOT being saved' in w.message
    assert w._message.layout.display != 'none'

    w._selectors[2]._selector.value = False
    assert (fits_dir / SELECTION_FILE_NAME).read_bytes() == original
    assert not list(fits_dir.glob('*.tmp'))

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_notebooks.image_selector import ImageSelect, SelectedCombiner

# Names of the frames made by ``mixed_dirs``, in collection order.
DARK = "dark-0.fit"
LIGHTS = ["light-0.fit", "light-1.fit"]


@pytest.fixture
def mixed_dirs(tmp_path, monkeypatch):
    """Data directory of two light frames and a dark, and an empty destination.

    The lights hold 1 and 2 and the dark holds 100, all in filter V, so the
    mean of the combined image is 1.5 if only the lights were combined and
    34.33 if the dark was too. The destination directory exists and is
    empty, as it is in the notebook.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = tmp_path / "combined"
    destination.mkdir()
    frames = [(DARK, "DARK", 100.0),
              (LIGHTS[0], "LIGHT", 1.0),
              (LIGHTS[1], "LIGHT", 2.0)]
    for name, imagetyp, value in frames:
        hdu = fits.PrimaryHDU(np.full((16, 16), value, dtype="float32"))
        hdu.header["IMAGETYP"] = imagetyp
        hdu.header["FILTER"] = "V"
        hdu.header["BUNIT"] = "adu"
        hdu.writeto(data_dir / name)
    monkeypatch.chdir(tmp_path)
    return data_dir, destination


def _make_combiner(isel, destination):
    """Make a ``SelectedCombiner`` for light frames, the way the notebook does."""
    return SelectedCombiner(image_select=isel,
                            run_label="run",
                            description="Combine light images",
                            toggle_type="button",
                            group_by="filter",
                            apply_to={"imagetyp": "light"},
                            destination=str(destination))


def _press_go(combiner):
    """Choose to combine, then press the go button as a browser click does.

    Going through the button's handler, rather than calling ``action()``,
    also runs the part of reducer's ``ToggleGo`` that shows "Unlock
    settings" again, which only happens if ``action()`` returns.
    """
    combiner.toggle.value = True
    combiner._combine_method.toggle.value = True
    combiner._go_button.click()


def _assert_unlockable(combiner):
    """Check that reducer got as far as showing the "Unlock settings" button."""
    assert combiner._change_settings.layout.display == ""
    assert not combiner._change_settings.disabled


def test_manifest_lists_only_frames_matching_apply_to(mixed_dirs):
    """A checked dark is not recorded as part of a combination of lights.

    ``ImageSelect`` shows every FITS file in the directory, checked by
    default, but the combiner only combines frames that match
    ``apply_to``. The manifest is the permanent record of the run, so
    ``included`` must hold the lights that were combined (the mean of the
    result, 1.5, shows the dark was not) and the dark belongs in
    ``excluded``. The message reports the number actually combined.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)

    _press_go(combiner)

    combined = destination / "run_filter_V.fit"
    assert fits.getdata(combined).mean() == pytest.approx(1.5)
    manifest = json.loads(combiner.manifest_path.read_text())
    assert manifest["included"] == LIGHTS
    assert manifest["excluded"] == [DARK]
    assert "2 of 3 images were combined" in combiner.message
    assert combiner.last_error is None


def test_nothing_combined_when_no_checked_frame_matches_apply_to(mixed_dirs):
    """With only a dark checked, nothing is written and the widget says so.

    Without this check reducer finds no group to combine and returns
    quietly, and the widget used to report "Done" and write a manifest for
    a combination that never happened. There must be a visible message, no
    manifest, no image, and the widget must be left unlockable.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    for fname, selector in zip(isel._im_file_names, isel._selectors):
        selector._selector.value = fname == DARK

    _press_go(combiner)

    assert list(destination.iterdir()) == []
    assert combiner.manifest_path is None
    assert "None of the checked images match" in combiner.message
    assert "Done" not in combiner.message
    assert combiner._message.layout.display != "none"
    _assert_unlockable(combiner)


def test_failed_combine_does_not_escape_and_leaves_widget_unlockable(
        mixed_dirs, mocker):
    """A combination that raises is shown, remembered and can be retried.

    reducer disables the go button before calling ``action()`` and only
    shows "Unlock settings" after it returns, so an exception that escaped
    would leave the widget locked for good. Pressing the button must not
    raise, the message must name the error, and the exception and its
    traceback must be kept for the user to look at.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    mocker.patch("reducer.astro_gui.Combiner.action",
                 side_effect=RuntimeError("disk full"))

    _press_go(combiner)

    assert isinstance(combiner.last_error, RuntimeError)
    assert "RuntimeError" in combiner.last_traceback
    assert "disk full" in combiner.message
    assert "no manifest was written" in combiner.message
    assert combiner._message.layout.display != "none"
    assert combiner.manifest_path is None
    assert list(destination.iterdir()) == []
    _assert_unlockable(combiner)


def test_manifest_failure_says_images_were_combined(mixed_dirs, mocker):
    """If only the manifest fails, the message says the images exist.

    The combined image is on disk at this point, so reporting nothing, or
    reporting that the combination failed, would both be wrong. The
    message must say the images were combined and that the manifest was
    not written, and the widget must be left unlockable.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    mocker.patch("astro_notebooks.image_selector.write_selection_manifest",
                 side_effect=OSError("quota exceeded"))

    _press_go(combiner)

    assert (destination / "run_filter_V.fit").exists()
    assert combiner.manifest_path is None
    assert "WERE combined" in combiner.message
    assert "manifest" in combiner.message
    assert "quota exceeded" in combiner.message
    assert isinstance(combiner.last_error, OSError)
    _assert_unlockable(combiner)


def test_stale_selector_is_refused(mixed_dirs):
    """A combiner whose selector has been replaced refuses to combine.

    Re-running the notebook cell that makes the selector leaves the
    combiner holding the old, discarded selector while the user reviews
    frames in the new one. The new selector saves the selection, so the
    file on disk no longer matches the old selector's checkboxes; the
    combiner must notice, combine nothing and tell the user to re-run its
    cell.
    """
    data_dir, destination = mixed_dirs
    old_isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(old_isel, destination)

    new_isel = ImageSelect(directory=data_dir)
    new_isel._selectors[1]._selector.value = False

    _press_go(combiner)

    assert list(destination.iterdir()) == []
    assert combiner.manifest_path is None
    assert "does not match" in combiner.message
    assert "re-run this cell" in combiner.message
    _assert_unlockable(combiner)


def test_stale_check_skipped_when_selector_cannot_save(mixed_dirs):
    """A selector that cannot save is not compared with the file on disk.

    A selector on a read-only directory, or one whose selection file could
    not be read, marks itself with ``_can_save = False`` and does not
    write the file, so the file is not expected to match its checkboxes.
    The combination must go ahead from the checkboxes.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    isel.selection_path.write_text(json.dumps({LIGHTS[0]: False}))
    isel._can_save = False

    _press_go(combiner)

    assert (destination / "run_filter_V.fit").exists()
    manifest = json.loads(combiner.manifest_path.read_text())
    assert manifest["included"] == LIGHTS


def test_stale_check_skipped_after_failed_save(mixed_dirs, mocker):
    """A failed save of the latest click does not block the combination.

    When the write fails after a click the file on disk no longer matches
    the checkboxes, but the selector has already said so in its own
    message and the checkboxes are what the user sees. The combiner must
    use them rather than refuse with a message about a re-run cell.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    mocker.patch("astro_notebooks.image_selector._atomic_write_json",
                 side_effect=OSError("disk full"))
    isel._selectors[isel._im_file_names.index(LIGHTS[0])]._selector.value = False
    assert "could NOT be saved" in isel.message
    mocker.stopall()

    _press_go(combiner)

    assert (destination / "run_filter_V.fit").exists()
    manifest = json.loads(combiner.manifest_path.read_text())
    assert manifest["included"] == [LIGHTS[1]]
    _assert_unlockable(combiner)


def test_stale_check_skipped_when_selection_file_unreadable(mixed_dirs):
    """A missing or unparseable selection file does not block the combine.

    The check exists to catch a newer selector having written the file.
    If the file cannot be read there is nothing to compare with, and
    refusing to combine would leave the user stuck.
    """
    data_dir, destination = mixed_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_combiner(isel, destination)
    isel.selection_path.write_text("{ not json")

    _press_go(combiner)

    assert (destination / "run_filter_V.fit").exists()
    assert combiner.last_error is None

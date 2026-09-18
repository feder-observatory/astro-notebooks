import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import ipywidgets as ipw
import numpy as np
import pytest
from astropy.io import fits
from ccdproc import ImageFileCollection
from PIL import Image

from astro_notebooks.image_selector import (
    SELECTION_FILE_NAME,
    ImageSelect,
    ImageWithSelector,
    SelectedCombiner,
    _make_one_thumbnail,
    _scale_and_downsample,
    _thumbnail_data,
    write_selection_manifest,
)

from .conftest import IMAGE_SHAPE, N_IMAGES


def _walk_widgets(widget):
    """Yield widget and all its descendants."""
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


def test_scale_and_downsample_output_range_and_nans():
    """Scaled thumbnail data is always displayable.

    An image containing a NaN and a pixel above the 1e5 clamp must come
    out with no NaNs and every value between 0 and 1, since the result is
    turned straight into an 8-bit PNG.
    """
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    out = _scale_and_downsample(data, downsample=8)
    assert not np.isnan(out).any()
    assert out.min() >= 0
    assert out.max() <= 1


def test_scale_and_downsample_shape_reduced():
    """The ``downsample`` factor sets the output size.

    A 64x64 image becomes 8x8 with ``downsample=8``, and ``downsample=1``
    leaves the shape alone instead of failing or block-reducing by one.
    """
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    out_downsampled = _scale_and_downsample(data, downsample=8)
    assert out_downsampled.shape == (8, 8)
    out_no_downsample = _scale_and_downsample(data, downsample=1)
    assert out_no_downsample.shape == (64, 64)


def test_scale_and_downsample_does_not_mutate_input():
    """Making a thumbnail never changes the caller's array.

    The clamp and NaN replacement happen on a copy, so the NaN and the
    too-bright pixel in the input are still there afterwards.
    """
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    original = data.copy()
    _scale_and_downsample(data, downsample=8)
    np.testing.assert_array_equal(data, original)


def test_make_one_thumbnail(fits_dir, tmp_path):
    """One FITS file becomes one grayscale PNG of the expected size.

    The PNG must be single-channel ("L") mode, which is what keeps the
    thumbnails small, be 1/8 of the image size on each side, and contain
    image structure rather than a constant.
    """
    src = sorted(fits_dir.glob("*.fit"))[0]
    dest = tmp_path / "thumb.png"
    _make_one_thumbnail(src, dest, 8)
    assert dest.exists()
    img = Image.open(dest)
    assert img.mode == "L"
    assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)
    assert np.asarray(img).std() > 0


def test_thumbnails_one_per_fits_grayscale(fits_dir):
    """Building the widget writes exactly one thumbnail per FITS file.

    The thumbnails are named after the full image file names, extension
    included, so that ``x.fit`` and ``x.fits`` do not share one. Each is a
    grayscale PNG at 1/8 of the image size, the default downsampling.
    """
    w = ImageSelect(directory=fits_dir)
    thumbs_dir = w.thumbs
    png_names = {p.name for p in thumbs_dir.glob("*.png")}
    expected_names = {f"image-{i:03d}.fit.png" for i in range(N_IMAGES)}
    assert png_names == expected_names
    for p in thumbs_dir.glob("*.png"):
        img = Image.open(p)
        assert img.mode == "L"
        assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)


def test_existing_thumbnails_not_regenerated(fits_dir):
    """Thumbnails that already exist are reused, not rewritten.

    Making the widget a second time for the same directory must leave
    every PNG's modification time unchanged; making thumbnails is the
    slow part of opening a directory of images.
    """
    thumbs = fits_dir / "thumbs"
    ImageSelect(directory=fits_dir)
    mtimes = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("*.png")}
    ImageSelect(directory=fits_dir)
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("*.png")}
    assert mtimes == mtimes2


def test_stale_thumbnails_cleaned_up(fits_dir):
    """A thumbnail whose FITS file has gone is deleted.

    After an image is removed from the directory, rebuilding the widget
    removes that image's thumbnail and keeps all of the others.
    """
    ImageSelect(directory=fits_dir)
    (fits_dir / "image-000.fit").unlink()
    ImageSelect(directory=fits_dir)
    assert not (fits_dir / "thumbs" / "image-000.fit.png").exists()
    for i in range(1, N_IMAGES):
        assert (fits_dir / "thumbs" / f"image-{i:03d}.fit.png").exists()


def test_progress_ui_shown_and_hidden(fits_dir, mocker):
    """A progress bar is shown while thumbnails are made, then hidden.

    Exactly one progress widget is displayed, its bar has counted up to
    the number of images, and it is hidden once the work is done so it
    does not linger above the selector.
    """
    displayed = []
    mocker.patch("astro_notebooks.image_selector.display", side_effect=lambda *a, **k: displayed.extend(a))
    ImageSelect(directory=fits_dir)
    assert len(displayed) == 1
    progress_widgets = [w for w in _walk_widgets(displayed[0]) if isinstance(w, ipw.IntProgress)]
    assert len(progress_widgets) == 1
    assert progress_widgets[0].max == N_IMAGES
    assert progress_widgets[0].value == N_IMAGES
    assert displayed[0].layout.display == "none"


def test_no_progress_display_when_cached(fits_dir, mocker):
    """Nothing is displayed when there are no thumbnails to make.

    With every thumbnail already cached, a second widget must not flash
    an empty progress bar.
    """
    ImageSelect(directory=fits_dir)
    displayed = []
    mocker.patch("astro_notebooks.image_selector.display", side_effect=lambda *a, **k: displayed.extend(a))
    ImageSelect(directory=fits_dir)
    assert displayed == []


def test_image_select_structure(fits_dir):
    """The widget is a hidden message above one grid of selectors.

    Also checks that no button is left (nothing is moved any more, so
    "Move rejects" is gone), that the thumbnail names and file names are
    both the full file names in collection order, and that each selector
    is showing real PNG bytes.
    """
    w = ImageSelect(directory=fits_dir)
    assert len(w.children) == 2
    assert w.children[0].layout.display == 'none' and w.message == ''
    assert isinstance(w.children[1], ipw.GridspecLayout)
    assert not [c for c in _walk_widgets(w) if isinstance(c, ipw.Button)]
    assert w._im_base_names == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    assert w._im_file_names == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    assert len(w._selectors) == N_IMAGES
    for sel in w._selectors:
        assert isinstance(sel, ImageWithSelector)
    for sel in w._selectors:
        # ipywidgets may hand the value back as a memoryview
        png_bytes = bytes(sel.image_display.value)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b'\x89PNG')


def test_downsample_kwarg_flows_through(fits_dir):
    """``ImageSelect(downsample=...)`` reaches the thumbnail writer.

    With ``downsample=4`` the thumbnails are 1/4 of the image size, not
    the default 1/8.
    """
    w = ImageSelect(directory=fits_dir, downsample=4)
    for p in w.thumbs.glob("*.png"):
        img = Image.open(p)
        assert img.size == (IMAGE_SHAPE[1] // 4, IMAGE_SHAPE[0] // 4)


def test_thumb_cache_lives_in_data_dir(fits_dir, tmp_path):
    """Thumbnails are cached in ``<data_dir>/thumbs``, not in the cwd.

    A cache in the working directory would be reused for a different
    directory of images. The ``fits_dir`` fixture makes ``tmp_path`` the
    cwd, so nothing should be written there: it should hold only the data
    directory itself.
    """
    w = ImageSelect(directory=fits_dir)
    assert w.thumbs == fits_dir / "thumbs"
    assert w.thumbs.is_dir()
    assert len(list(w.thumbs.glob("*.png"))) == N_IMAGES
    assert not (tmp_path / "thumbs").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"data"}


def test_default_worker_cap_is_four(fits_dir, mocker):
    """By default the thumbnail thread pool has four workers.

    The cap, rather than one thread per CPU, is what keeps peak memory
    down on a JupyterHub with a per-user limit, so a change to the
    default should be deliberate.
    """
    spy = mocker.patch(
        "astro_notebooks.image_selector.ThreadPoolExecutor",
        side_effect=ThreadPoolExecutor,
    )
    ImageSelect(directory=fits_dir)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["max_workers"] == 4


def test_max_workers_kwarg_flows_through(fits_dir, mocker):
    """``ImageSelect(max_workers=...)`` sets the thread pool size."""
    spy = mocker.patch(
        "astro_notebooks.image_selector.ThreadPoolExecutor",
        side_effect=ThreadPoolExecutor,
    )
    ImageSelect(directory=fits_dir, max_workers=2)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["max_workers"] == 2


def test_thumbnail_data_matches_whole_frame(tmp_path):
    """A banded read gives the same thumbnail as a whole-frame call.

    The banded read comes from reducer, which tests the band arithmetic
    itself; what is checked here is that the clamp is handed to it as a
    preprocess and the normalization is applied after it, so that a
    banded read and a whole-frame call still agree. The shape is not a
    multiple of the band height (64) or of downsample (8), and there is
    both a clamped pixel and a NaN.
    """
    rng = np.random.default_rng(1234)
    data = rng.uniform(100.0, 1000.0, size=(300, 130))
    data[0:2, 0:2] = np.nan
    data[7, 7] = 2e5
    path = tmp_path / "odd-shape.fit"
    fits.PrimaryHDU(data).writeto(path)

    banded = _thumbnail_data(path, downsample=8, band_rows=64)
    whole = _scale_and_downsample(fits.getdata(path), downsample=8)
    assert banded.shape == (300 // 8, 130 // 8)
    assert np.array_equal(banded, whole)


def test_banded_read_uses_first_hdu_with_data(tmp_path):
    """The thumbnail comes from the first HDU that has image data.

    For a file with an empty primary HDU and the image in the first
    extension, the banded read must skip to the extension and match a
    whole-frame call on that extension's data.
    """
    rng = np.random.default_rng(5)
    data = rng.uniform(100.0, 1000.0, size=(64, 64))
    path = tmp_path / "empty-primary.fit"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data)]
    ).writeto(path)

    banded = _thumbnail_data(path, downsample=8)
    whole = _scale_and_downsample(data, downsample=8)
    assert np.array_equal(banded, whole)


def _selection_json(fits_dir):
    """Contents of the saved selection file in ``fits_dir``."""
    return json.loads((fits_dir / SELECTION_FILE_NAME).read_text())


def test_selection_file_written_at_construction(fits_dir):
    """The selection file exists as soon as the widget is made.

    It is written beside the data with every image included, and
    ``selected_files`` / ``selected_paths`` list every image, in
    collection order, as names and as full paths.
    """
    w = ImageSelect(directory=fits_dir)
    assert w.selection_path == fits_dir / SELECTION_FILE_NAME
    saved = _selection_json(fits_dir)
    assert saved == {f"image-{i:03d}.fit": True for i in range(N_IMAGES)}
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    assert w.selected_paths == [fits_dir / f for f in w.selected_files]


def test_selection_round_trip(fits_dir):
    """Unticking images is saved at once and restored in a new widget.

    The JSON file is updated by the checkbox change itself, with no save
    step, and a second widget for the same directory comes up with the
    same two images unticked. This is what lets a user close the notebook
    and come back without redoing the selection.
    """
    w = ImageSelect(directory=fits_dir)
    w._selectors[1]._selector.value = False
    w._selectors[3]._selector.value = False

    expected = [f"image-{i:03d}.fit" for i in (0, 2, 4)]
    assert w.selected_files == expected
    saved = _selection_json(fits_dir)
    assert saved["image-001.fit"] is False
    assert saved["image-003.fit"] is False

    w2 = ImageSelect(directory=fits_dir)
    assert w2.selected_files == expected
    for i, sel in enumerate(w2._selectors):
        assert sel._selector.value == (i not in (1, 3))


def test_new_file_defaults_to_included(fits_dir):
    """A frame added after the selection was saved starts out included.

    The earlier choice to exclude another frame is kept, and the new
    frame gets its own entry in the rewritten selection file.
    """
    w = ImageSelect(directory=fits_dir)
    w._selectors[0]._selector.value = False

    # a frame that arrived after the selection was saved
    rng = np.random.default_rng(3)
    hdu = fits.PrimaryHDU(rng.uniform(100.0, 1000.0, size=IMAGE_SHAPE))
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.writeto(fits_dir / "image-099.fit")

    w2 = ImageSelect(directory=fits_dir)
    new_index = w2._im_file_names.index("image-099.fit")
    assert w2._selectors[new_index]._selector.value
    assert "image-099.fit" in w2.selected_files
    assert "image-000.fit" not in w2.selected_files
    assert _selection_json(fits_dir)["image-099.fit"] is True


def test_entry_for_deleted_file_dropped(fits_dir):
    """A deleted image's entry is removed from the selection file.

    Rebuilding the widget rewrites the file with exactly the images that
    are still in the directory, so it cannot accumulate stale names.
    """
    ImageSelect(directory=fits_dir)
    assert "image-000.fit" in _selection_json(fits_dir)

    (fits_dir / "image-000.fit").unlink()
    ImageSelect(directory=fits_dir)
    saved = _selection_json(fits_dir)
    assert "image-000.fit" not in saved
    assert set(saved) == {f"image-{i:03d}.fit" for i in range(1, N_IMAGES)}


def test_corrupt_selection_file_ignored(fits_dir):
    """A selection file that is not JSON is set aside with a warning.

    The widget must still open, with every image included. The bad file
    is kept as ``image_selection.json.bak`` rather than destroyed, and is
    replaced by a valid one so the warning does not come back on every
    later run.
    """
    (fits_dir / SELECTION_FILE_NAME).write_text("{not json at all")
    with pytest.warns(UserWarning, match="image selection file"):
        w = ImageSelect(directory=fits_dir)
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    backup = fits_dir / (SELECTION_FILE_NAME + ".bak")
    assert backup.read_text() == "{not json at all"
    # the bad file has been replaced by a good one
    assert _selection_json(fits_dir) == {
        f"image-{i:03d}.fit": True for i in range(N_IMAGES)
    }


def test_selection_file_of_wrong_type_ignored(fits_dir):
    """Valid JSON that is not a name-to-bool mapping is ignored.

    A list parses without error but cannot be a saved selection, so it
    gets the same warning and all-included fallback as a corrupt file.
    """
    (fits_dir / SELECTION_FILE_NAME).write_text('["image-000.fit"]')
    with pytest.warns(UserWarning, match="image selection file"):
        w = ImageSelect(directory=fits_dir)
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]


def test_selection_entry_with_non_boolean_value_ignored(fits_dir):
    """One wrongly typed value is dropped without losing the other choices.

    A hand-edited ``"false"`` is a string, and ``bool("false")`` is True,
    so without a check a rejected frame would quietly come back. Only that
    entry is ignored, with a warning that names it: the frame is included,
    the valid entries are still restored, and the file is rewritten with
    real booleans. Rejecting the whole file instead would wipe every other
    choice, because the selection is saved again right after it is
    restored.
    """
    path = fits_dir / SELECTION_FILE_NAME
    path.write_text(json.dumps({
        "image-000.fit": "false",
        "image-001.fit": False,
        "image-002.fit": True,
        "image-003.fit": False,
    }))
    with pytest.warns(UserWarning, match="image-000.fit"):
        w = ImageSelect(directory=fits_dir)

    assert w.selected_files == [f"image-{i:03d}.fit" for i in (0, 2, 4)]
    assert _selection_json(fits_dir) == {
        "image-000.fit": True,
        "image-001.fit": False,
        "image-002.fit": True,
        "image-003.fit": False,
        "image-004.fit": True,
    }


def test_collection_from_selected_files(fits_dir):
    """``selected_files`` can be used directly to make a collection.

    This is how ``SelectedCombiner`` combines only the chosen frames
    without moving files. The chosen names must be the only files in the
    collection, and must still filter by ``imagetyp``, both before and
    after a ``refresh()``, because reducer's Combiner refreshes the
    collection before using it.
    """
    w = ImageSelect(directory=fits_dir)
    w._selectors[2]._selector.value = False
    expected = [f"image-{i:03d}.fit" for i in (0, 1, 3, 4)]
    assert w.selected_files == expected

    images = ImageFileCollection(location=fits_dir, filenames=w.selected_files)
    assert list(images.files) == expected
    assert list(images.files_filtered(imagetyp="light")) == expected
    # reducer's Combiner refreshes the collection before using it
    images.refresh()
    assert list(images.files) == expected
    assert list(images.files_filtered(imagetyp="light")) == expected


def test_write_selection_manifest(fits_dir, tmp_path):
    """The manifest records which frames went into a combination.

    It is written to ``<destination>/<run_label>_manifest.json``,
    creating the directory if needed, and holds the run label, the data
    directory, the included and excluded file names in collection order,
    and a timestamp that parses as ISO 8601.
    """
    w = ImageSelect(directory=fits_dir)
    w._selectors[0]._selector.value = False
    w._selectors[4]._selector.value = False

    destination = tmp_path / "combined"
    manifest_path = write_selection_manifest(w, destination, "run_one")
    assert manifest_path == destination / "run_one_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {
        "run_label", "timestamp", "data_dir", "included", "excluded"
    }
    assert manifest["run_label"] == "run_one"
    assert Path(manifest["data_dir"]) == fits_dir
    assert manifest["included"] == [f"image-{i:03d}.fit" for i in (1, 2, 3)]
    assert manifest["excluded"] == [f"image-{i:03d}.fit" for i in (0, 4)]
    # a plain ISO timestamp
    datetime.fromisoformat(manifest["timestamp"])


def test_write_selection_manifest_records_given_frames(fits_dir, tmp_path):
    """``included`` makes the manifest record a snapshot, not the checkboxes.

    ``SelectedCombiner`` writes the manifest after the combination, from
    the list of frames it actually combined. A checkbox changed in the
    meantime must not alter what the manifest says went into the result.
    """
    w = ImageSelect(directory=fits_dir)
    snapshot = [f"image-{i:03d}.fit" for i in (0, 1, 2)]
    # the live selection differs from the snapshot in both directions
    w._selectors[0]._selector.value = False

    manifest_path = write_selection_manifest(w, tmp_path / "combined",
                                             "run_two", included=snapshot)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["included"] == snapshot
    assert manifest["excluded"] == [f"image-{i:03d}.fit" for i in (3, 4)]


# Pixel value of each frame made by combine_dirs. The frames are constant,
# so the mean of their average says which of them were combined.
COMBINE_VALUES = [1.0, 2.0, 100.0]


@pytest.fixture
def combine_dirs(tmp_path, monkeypatch):
    """Data directory of three constant light frames, and an empty destination.

    The frames hold 1, 2 and 100, all in filter V, so the average of the
    first two is 1.5 and the average of all three is 34.33: the combined
    image itself shows whether the bright frame was used. The destination
    directory exists and is empty, as it is in the notebook.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = tmp_path / "combined"
    destination.mkdir()
    for i, value in enumerate(COMBINE_VALUES):
        hdu = fits.PrimaryHDU(np.full((16, 16), value, dtype="float32"))
        hdu.header["IMAGETYP"] = "LIGHT"
        hdu.header["FILTER"] = "V"
        hdu.header["BUNIT"] = "adu"
        hdu.writeto(data_dir / f"frame-{i}.fit")
    monkeypatch.chdir(tmp_path)
    return data_dir, destination


def _make_selected_combiner(isel, destination):
    """Make a ``SelectedCombiner`` the way the notebook does."""
    return SelectedCombiner(image_select=isel,
                            run_label="run",
                            description="Combine light images",
                            toggle_type="button",
                            group_by="filter",
                            apply_to={"imagetyp": "light"},
                            destination=str(destination))


def _press_go(combiner):
    """Choose to combine, then press the combiner's go button.

    Clicking the button, rather than calling ``action()``, runs the same
    handler that a click in the browser does.
    """
    combiner.toggle.value = True
    combiner._combine_method.toggle.value = True
    combiner._go_button.click()


def test_selected_combiner_uses_frames_checked_when_go_is_pressed(
        combine_dirs):
    """A frame unticked after the combiner was made is left out.

    This is the bug the class exists to fix. With a plain ``Combiner``
    given ``ImageFileCollection(filenames=isel.selected_files)``, the list
    of frames was fixed when that notebook cell ran, so under "Run All",
    or if the cell was run before the review was finished, unticking a
    frame had no effect on the result. The combiner here is made while
    every frame is checked, the bright frame is unticked afterwards, and
    the mean of the output (1.5, not 34.33) shows it was not combined.
    Grouping by filter is on, so this also checks that the group-by
    widget was given the new collection.
    """
    data_dir, destination = combine_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_selected_combiner(isel, destination)

    isel._selectors[2]._selector.value = False
    _press_go(combiner)

    combined = destination / "run_filter_V.fit"
    assert combined.exists()
    assert fits.getdata(combined).mean() == pytest.approx(1.5)
    assert list(combiner.image_source.files) == ["frame-0.fit", "frame-1.fit"]


def test_selected_combiner_writes_manifest_of_combined_frames(combine_dirs):
    """A finished combination leaves a manifest of exactly what it used.

    The manifest is written by the combiner itself once the combination
    has succeeded, from the same snapshot of the selection, so it cannot
    be written before the combination or disagree with it. A frame ticked
    again after the combination must not appear in it. The path is kept
    on the combiner and shown in the widget.
    """
    data_dir, destination = combine_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_selected_combiner(isel, destination)
    assert combiner.manifest_path is None

    isel._selectors[2]._selector.value = False
    _press_go(combiner)
    isel._selectors[2]._selector.value = True

    assert combiner.manifest_path == destination / "run_manifest.json"
    manifest = json.loads(combiner.manifest_path.read_text())
    assert manifest["run_label"] == "run"
    assert manifest["included"] == ["frame-0.fit", "frame-1.fit"]
    assert manifest["excluded"] == ["frame-2.fit"]
    assert str(combiner.manifest_path) in combiner.message
    assert combiner._message.layout.display != "none"


def test_selected_combiner_refuses_empty_selection(combine_dirs):
    """With nothing checked, nothing is combined and the widget says so.

    An ``ImageFileCollection`` given an empty list of file names uses
    every file in the directory, so without this check unticking every
    frame would combine all of them. Nothing may be written to the
    destination, neither an image nor a manifest, and because an
    exception raised in a button callback is easy to miss, the refusal
    has to be a visible message in the widget.
    """
    data_dir, destination = combine_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_selected_combiner(isel, destination)

    for selector in isel._selectors:
        selector._selector.value = False
    _press_go(combiner)

    assert list(destination.iterdir()) == []
    assert combiner.manifest_path is None
    assert "No images are checked" in combiner.message
    assert combiner._message.layout.display != "none"
    assert combiner._message in combiner.children


def test_selected_combiner_no_manifest_when_combine_fails(combine_dirs,
                                                          mocker):
    """A failed combination writes no manifest and shows the failure.

    A manifest next to a missing or half-made result would claim a
    combination that did not happen, and the manifest left by an earlier
    successful run is no longer reported as this run's. The error is not
    raised, because reducer only shows "Unlock settings" once ``action``
    has returned and an exception would leave the widget locked; it is
    kept on the combiner instead, so the traceback is not lost.
    """
    data_dir, destination = combine_dirs
    isel = ImageSelect(directory=data_dir)
    combiner = _make_selected_combiner(isel, destination)
    mocker.patch("reducer.astro_gui.Combiner.action",
                 side_effect=RuntimeError("disk full"))

    combiner.action()

    assert isinstance(combiner.last_error, RuntimeError)
    assert "disk full" in combiner.last_traceback
    assert list(destination.iterdir()) == []
    assert combiner.manifest_path is None
    assert "disk full" in combiner.message


def test_selected_combiner_rejects_image_source_argument(combine_dirs):
    """``image_source`` and ``file_name_base`` cannot be passed in.

    Both are set from the selector and the run label. Accepting a
    collection here would bring back the fixed list of frames that this
    class replaces, so it is an error rather than being silently ignored.
    """
    data_dir, destination = combine_dirs
    isel = ImageSelect(directory=data_dir)
    images = ImageFileCollection(location=data_dir)

    with pytest.raises(TypeError, match="image_source"):
        SelectedCombiner(image_select=isel, run_label="run",
                         image_source=images, destination=str(destination))
    with pytest.raises(TypeError, match="file_name_base"):
        SelectedCombiner(image_select=isel, run_label="run",
                         file_name_base="other", destination=str(destination))

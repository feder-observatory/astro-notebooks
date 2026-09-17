import json
import shutil
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


def _thumb_arrays(thumb_dir):
    """Map png name -> pixel array for every thumbnail."""
    return {
        p.name: np.asarray(Image.open(p))
        for p in sorted(Path(thumb_dir).glob("*.png"))
    }


def test_scale_and_downsample_output_range_and_nans():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    out = _scale_and_downsample(data, downsample=8)
    assert not np.isnan(out).any()
    assert out.min() >= 0
    assert out.max() <= 1


def test_scale_and_downsample_shape_reduced():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    out_downsampled = _scale_and_downsample(data, downsample=8)
    assert out_downsampled.shape == (8, 8)
    out_no_downsample = _scale_and_downsample(data, downsample=1)
    assert out_no_downsample.shape == (64, 64)


def test_scale_and_downsample_does_not_mutate_input():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    original = data.copy()
    _scale_and_downsample(data, downsample=8)
    np.testing.assert_array_equal(data, original)


def test_make_one_thumbnail(fits_dir, tmp_path):
    src = sorted(fits_dir.glob("*.fit"))[0]
    dest = tmp_path / "thumb.png"
    _make_one_thumbnail(src, dest, 8)
    assert dest.exists()
    img = Image.open(dest)
    assert img.mode == "L"
    assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)
    assert np.asarray(img).std() > 0


def test_thumbnails_one_per_fits_grayscale(fits_dir):
    w = ImageSelect(directory=fits_dir)
    thumbs_dir = w.thumbs
    png_names = {p.name for p in thumbs_dir.glob("*.png")}
    expected_names = {f"image-{i:03d}.png" for i in range(N_IMAGES)}
    assert png_names == expected_names
    for p in thumbs_dir.glob("*.png"):
        img = Image.open(p)
        assert img.mode == "L"
        assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)


def test_existing_thumbnails_not_regenerated(fits_dir):
    thumbs = fits_dir / "thumbs"
    ImageSelect(directory=fits_dir)
    mtimes = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("*.png")}
    ImageSelect(directory=fits_dir)
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("*.png")}
    assert mtimes == mtimes2


def test_stale_thumbnails_cleaned_up(fits_dir):
    ImageSelect(directory=fits_dir)
    (fits_dir / "image-000.fit").unlink()
    ImageSelect(directory=fits_dir)
    assert not (fits_dir / "thumbs" / "image-000.png").exists()
    for i in range(1, N_IMAGES):
        assert (fits_dir / "thumbs" / f"image-{i:03d}.png").exists()


def test_parallel_matches_serial(fits_dir):
    thumbs = fits_dir / "thumbs"
    ImageSelect(directory=fits_dir, max_workers=1)
    serial = _thumb_arrays(thumbs)
    shutil.rmtree(thumbs)
    ImageSelect(directory=fits_dir, max_workers=2)
    parallel = _thumb_arrays(thumbs)
    assert serial.keys() == parallel.keys()
    for k in serial:
        assert serial[k].size > 0
        assert np.array_equal(serial[k], parallel[k])


def test_progress_ui_shown_and_hidden(fits_dir, mocker):
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
    ImageSelect(directory=fits_dir)
    displayed = []
    mocker.patch("astro_notebooks.image_selector.display", side_effect=lambda *a, **k: displayed.extend(a))
    ImageSelect(directory=fits_dir)
    assert displayed == []


def test_image_select_structure(fits_dir):
    w = ImageSelect(directory=fits_dir)
    assert len(w.children) == 1
    assert isinstance(w.children[0], ipw.GridspecLayout)
    assert not [c for c in _walk_widgets(w) if isinstance(c, ipw.Button)]
    assert w._im_base_names == [f"image-{i:03d}" for i in range(N_IMAGES)]
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
    w = ImageSelect(directory=fits_dir, downsample=4)
    for p in w.thumbs.glob("*.png"):
        img = Image.open(p)
        assert img.size == (IMAGE_SHAPE[1] // 4, IMAGE_SHAPE[0] // 4)


def test_thumb_cache_lives_in_data_dir(fits_dir, tmp_path):
    w = ImageSelect(directory=fits_dir)
    assert w.thumbs == fits_dir / "thumbs"
    assert w.thumbs.is_dir()
    assert len(list(w.thumbs.glob("*.png"))) == N_IMAGES
    # cwd is tmp_path (see the fits_dir fixture); nothing should be written
    # there, only the data directory itself should exist.
    assert not (tmp_path / "thumbs").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"data"}


def test_collection_ignores_thumbs_dir(fits_dir):
    expected = set(ImageFileCollection(fits_dir).files)
    assert expected == {f"image-{i:03d}.fit" for i in range(N_IMAGES)}
    w = ImageSelect(directory=fits_dir)
    assert (fits_dir / "thumbs").is_dir()
    w._collection.refresh()
    assert set(w._collection.files) == expected
    # and a freshly built collection does not see the thumbnails either
    assert set(ImageFileCollection(fits_dir).files) == expected


def test_default_worker_cap_is_four(fits_dir, mocker):
    spy = mocker.patch(
        "astro_notebooks.image_selector.ThreadPoolExecutor",
        side_effect=ThreadPoolExecutor,
    )
    ImageSelect(directory=fits_dir)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["max_workers"] == 4


def test_max_workers_kwarg_flows_through(fits_dir, mocker):
    spy = mocker.patch(
        "astro_notebooks.image_selector.ThreadPoolExecutor",
        side_effect=ThreadPoolExecutor,
    )
    ImageSelect(directory=fits_dir, max_workers=2)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["max_workers"] == 2


def test_banded_read_matches_whole_frame(tmp_path):
    # height is not a multiple of the band height (64) or of downsample (8),
    # and the width is not a multiple of downsample either
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


def test_banded_read_matches_whole_frame_no_downsample(tmp_path):
    rng = np.random.default_rng(99)
    data = rng.uniform(100.0, 1000.0, size=(70, 33)).astype(np.float32)
    data[3, 3] = 5e5
    path = tmp_path / "no-downsample.fit"
    fits.PrimaryHDU(data).writeto(path)

    banded = _thumbnail_data(path, downsample=1, band_rows=32)
    whole = _scale_and_downsample(fits.getdata(path), downsample=1)
    assert banded.shape == (70, 33)
    assert np.array_equal(banded, whole)


def test_banded_read_uses_first_hdu_with_data(tmp_path):
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
    return json.loads((fits_dir / SELECTION_FILE_NAME).read_text())


def test_selection_file_written_at_construction(fits_dir):
    w = ImageSelect(directory=fits_dir)
    assert w.selection_path == fits_dir / SELECTION_FILE_NAME
    saved = _selection_json(fits_dir)
    assert saved == {f"image-{i:03d}.fit": True for i in range(N_IMAGES)}
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    assert w.selected_paths == [fits_dir / f for f in w.selected_files]


def test_selection_round_trip(fits_dir):
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
    ImageSelect(directory=fits_dir)
    assert "image-000.fit" in _selection_json(fits_dir)

    (fits_dir / "image-000.fit").unlink()
    ImageSelect(directory=fits_dir)
    saved = _selection_json(fits_dir)
    assert "image-000.fit" not in saved
    assert set(saved) == {f"image-{i:03d}.fit" for i in range(1, N_IMAGES)}


def test_corrupt_selection_file_ignored(fits_dir):
    (fits_dir / SELECTION_FILE_NAME).write_text("{not json at all")
    with pytest.warns(UserWarning, match="image selection file"):
        w = ImageSelect(directory=fits_dir)
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    # the bad file has been replaced by a good one
    assert _selection_json(fits_dir) == {
        f"image-{i:03d}.fit": True for i in range(N_IMAGES)
    }


def test_selection_file_of_wrong_type_ignored(fits_dir):
    (fits_dir / SELECTION_FILE_NAME).write_text('["image-000.fit"]')
    with pytest.warns(UserWarning, match="image selection file"):
        w = ImageSelect(directory=fits_dir)
    assert w.selected_files == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]


def test_collection_from_selected_files(fits_dir):
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

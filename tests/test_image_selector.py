from concurrent.futures import ThreadPoolExecutor

import ipywidgets as ipw
import numpy as np
from astropy.io import fits
from ccdproc import ImageFileCollection
from PIL import Image

from astro_notebooks.image_selector import (
    ImageSelect,
    ImageWithSelector,
    _make_one_thumbnail,
    _scale_and_downsample,
    _thumbnail_data,
)

from .conftest import IMAGE_SHAPE, N_IMAGES


def _walk_widgets(widget):
    """Yield widget and all its descendants."""
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


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
    assert len(w.children) == 2
    assert isinstance(w.children[0], ipw.GridspecLayout)
    assert isinstance(w.children[1], ipw.Button)
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


def test_move_rejects(fits_dir):
    w = ImageSelect(directory=fits_dir)
    w._selectors[0]._selector.value = False
    w._move_rejects_clicked(None)
    assert (fits_dir / "rejects" / "image-000.fit").exists()
    assert not (fits_dir / "image-000.fit").exists()
    assert len(w._selectors) == N_IMAGES - 1
    assert "image-000" not in w._im_base_names
    assert "image-000.fit" not in w._im_file_names
    assert len(w.children) == 2
    assert isinstance(w.children[0], ipw.GridspecLayout)


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


def test_thumbnail_data_matches_whole_frame(tmp_path):
    # The banded read comes from reducer, which tests the band arithmetic
    # itself; what is checked here is that the clamp is handed to it as a
    # preprocess and the normalization is applied after it, so that a
    # banded read and a whole-frame call still agree. The shape is not a
    # multiple of the band height (64) or of downsample (8), and there is
    # both a clamped pixel and a NaN.
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
    rng = np.random.default_rng(5)
    data = rng.uniform(100.0, 1000.0, size=(64, 64))
    path = tmp_path / "empty-primary.fit"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data)]
    ).writeto(path)

    banded = _thumbnail_data(path, downsample=8)
    whole = _scale_and_downsample(data, downsample=8)
    assert np.array_equal(banded, whole)

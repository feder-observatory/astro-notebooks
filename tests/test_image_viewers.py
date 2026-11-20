import numpy as np
import pytest
from astropy.io import fits
from astropy.visualization import LinearStretch, LogStretch, ManualInterval, SqrtStretch
from reducer.image_browser import ImageBrowser

from astro_notebooks.image_viewers import (
    DEFAULT_CUT_PERCENTILES,
    SLIDER_PERCENTILES,
    FolderViewer,
    SingleImageViewer,
    level_range,
    read_image,
)

from .conftest import IMAGE_SHAPE, N_IMAGES


def file_names(viewer):
    """Names of the files a folder viewer's browser lists."""
    return sorted(str(name) for name in viewer.browser._tree_widget._tree.table["file"])


def choose_folder(viewer, folder):
    """Pick a folder the way clicking through the chooser does."""
    chooser = viewer.chooser
    chooser._select.click()
    chooser._set_form_values(str(folder), "")
    chooser._select.click()


@pytest.fixture
def other_dir(tmp_path):
    """A second folder of images, whose headers have an OBJECT."""
    folder = tmp_path / "other"
    folder.mkdir()
    for name in ("a.fit", "b.fit"):
        hdu = fits.PrimaryHDU(np.ones((8, 8)))
        hdu.header["IMAGETYP"] = "LIGHT"
        hdu.header["OBJECT"] = "m13"
        hdu.writeto(folder / name)
    return folder


def test_read_image_without_a_unit_is_adu(fits_dir):
    image = read_image(fits_dir / "image-000.fit")
    assert image.unit == "adu"
    assert image.shape == IMAGE_SHAPE


def test_read_image_keeps_the_unit_in_the_header(tmp_path):
    hdu = fits.PrimaryHDU(np.ones((8, 8)))
    hdu.header["BUNIT"] = "electron"
    hdu.writeto(tmp_path / "electrons.fit")
    assert read_image(tmp_path / "electrons.fit").unit == "electron"


def test_read_image_raises_for_a_file_that_is_not_fits(tmp_path):
    path = tmp_path / "not.fit"
    path.write_text("not a FITS file")
    with pytest.raises(Exception):
        read_image(path)


def test_level_range_is_percentiles_of_the_finite_pixels():
    rng = np.random.default_rng(0)
    data = rng.normal(500, 10, (200, 200))
    data[0, 0] = np.nan
    data[5, 5] = 2e5
    finite = data[np.isfinite(data)]

    limits, cuts = level_range(data)

    assert limits == pytest.approx(np.percentile(finite, SLIDER_PERCENTILES))
    assert cuts == pytest.approx(np.percentile(finite, DEFAULT_CUT_PERCENTILES))
    # The one very bright pixel does not set the top of the slider.
    assert limits[1] < 600
    assert limits[0] <= cuts[0] < cuts[1] <= limits[1]


def test_level_range_of_a_big_image_is_close_to_that_of_all_its_pixels():
    rng = np.random.default_rng(1)
    data = rng.normal(500, 10, (2500, 2500)).astype(np.float32)
    limits, cuts = level_range(data)
    assert limits == pytest.approx(np.percentile(data, SLIDER_PERCENTILES), abs=1)
    assert cuts == pytest.approx(np.percentile(data, DEFAULT_CUT_PERCENTILES), abs=1)


@pytest.mark.parametrize(
    "data",
    [np.full((10, 10), 7.0), np.full((10, 10), np.nan)],
    ids=["one value", "all NaN"],
)
def test_level_range_can_always_be_the_ends_of_a_slider(data):
    limits, cuts = level_range(data)
    assert limits[1] > limits[0]
    assert limits[0] <= cuts[0] <= cuts[1] <= limits[1]


def test_folder_viewer_shows_the_starting_folder(fits_dir):
    viewer = FolderViewer(fits_dir)
    assert isinstance(viewer.browser, ImageBrowser)
    assert file_names(viewer) == [f"image-{i:03d}.fit" for i in range(N_IMAGES)]
    assert viewer.note.value == ""
    assert viewer.chooser.selected_path == str(fits_dir)


def test_folder_viewer_starts_in_the_current_directory(fits_dir, monkeypatch):
    monkeypatch.chdir(fits_dir)
    viewer = FolderViewer()
    assert viewer.folder == str(fits_dir)
    assert len(file_names(viewer)) == N_IMAGES


def test_choosing_a_folder_shows_its_images(fits_dir, other_dir):
    viewer = FolderViewer(fits_dir)
    first_browser = viewer.browser

    choose_folder(viewer, other_dir)

    assert viewer.folder == str(other_dir)
    assert viewer.browser is not first_browser
    assert file_names(viewer) == ["a.fit", "b.fit"]
    # Only the new browser is on screen.
    assert list(viewer._browser_box.children) == [viewer.browser]


def test_folder_without_images_says_so(fits_dir, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    viewer = FolderViewer(fits_dir)

    viewer.show_folder(empty)

    assert viewer.browser is None
    assert list(viewer._browser_box.children) == []
    assert "no FITS images" in viewer.note.value

    # And picking a folder that has some shows them again.
    viewer.show_folder(fits_dir)
    assert len(file_names(viewer)) == N_IMAGES
    assert viewer.note.value == ""


def test_folder_whose_images_lack_every_key_is_still_shown(tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    fits.PrimaryHDU(np.ones((8, 8))).writeto(folder / "bare.fit")

    viewer = FolderViewer(folder)

    assert viewer.note.value == ""
    assert file_names(viewer) == ["bare.fit"]


def test_folder_that_cannot_be_read_says_so(fits_dir, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk went away")

    viewer = FolderViewer(fits_dir)
    monkeypatch.setattr("astro_notebooks.image_viewers.ImageFileCollection", broken)

    viewer.show_folder(fits_dir)

    assert viewer.browser is None
    assert "disk went away" in viewer.note.value


def test_single_viewer_has_nothing_to_set_before_an_image_is_picked():
    viewer = SingleImageViewer()
    assert viewer.level_slider.disabled
    # Changing the stretch with no image loaded is not an error.
    viewer.stretch_chooser.value = "log"
    assert list(viewer.stretch_chooser.options) == ["linear", "log", "sqrt"]


def test_loading_an_image_sets_the_slider_from_it(fits_dir):
    path = fits_dir / "image-000.fit"
    viewer = SingleImageViewer(path)
    limits, cuts = level_range(read_image(path).data)

    slider = viewer.level_slider
    assert not slider.disabled
    assert (slider.min, slider.max) == pytest.approx(limits)
    assert slider.value == pytest.approx(cuts)
    assert slider.step == pytest.approx((limits[1] - limits[0]) / 1000)

    shown = viewer.image_widget.get_cuts()
    assert isinstance(shown, ManualInterval)
    assert (shown.vmin, shown.vmax) == pytest.approx(cuts)
    assert isinstance(viewer.image_widget.get_stretch(), LinearStretch)


def test_slider_and_stretch_change_the_viewer(fits_dir):
    viewer = SingleImageViewer(fits_dir / "image-000.fit")

    viewer.level_slider.value = (300, 700)
    shown = viewer.image_widget.get_cuts()
    assert (shown.vmin, shown.vmax) == pytest.approx((300, 700), abs=1)

    viewer.stretch_chooser.value = "log"
    assert isinstance(viewer.image_widget.get_stretch(), LogStretch)
    viewer.stretch_chooser.value = "sqrt"
    assert isinstance(viewer.image_widget.get_stretch(), SqrtStretch)


def test_choosing_another_image_resets_the_levels_and_keeps_the_stretch(fits_dir, tmp_path):
    # Far brighter than the first image, so the old slider range is no use.
    bright = tmp_path / "bright.fit"
    rng = np.random.default_rng(5)
    fits.PrimaryHDU(rng.normal(20000, 100, (64, 64))).writeto(bright)

    viewer = SingleImageViewer(fits_dir / "image-000.fit")
    viewer.stretch_chooser.value = "log"
    viewer.level_slider.value = (300, 700)

    chooser = viewer.chooser
    chooser._select.click()
    chooser._set_form_values(str(tmp_path), "bright.fit")
    chooser._select.click()

    limits, cuts = level_range(read_image(bright).data)
    assert (viewer.level_slider.min, viewer.level_slider.max) == pytest.approx(limits)
    assert viewer.level_slider.value == pytest.approx(cuts)
    shown = viewer.image_widget.get_cuts()
    assert (shown.vmin, shown.vmax) == pytest.approx(cuts)
    assert isinstance(viewer.image_widget.get_stretch(), LogStretch)
    assert viewer.note.value == ""


def test_image_that_cannot_be_read_says_so_and_keeps_the_last_one(fits_dir, tmp_path):
    bad = tmp_path / "bad.fit"
    bad.write_text("not a FITS file")
    viewer = SingleImageViewer(fits_dir / "image-000.fit")
    before = viewer.level_slider.value

    viewer.load(bad)

    assert "bad.fit" in viewer.note.value
    assert viewer.level_slider.value == before

    viewer.load(fits_dir / "image-001.fit")
    assert viewer.note.value == ""


@pytest.mark.parametrize("make", [FolderViewer, SingleImageViewer])
def test_viewers_can_be_sent_to_the_page(fits_dir, make):
    # An attribute that shadows one of the widget's own (``keys`` did)
    # passes every other test and fails when the notebook displays it.
    viewer = make()
    assert viewer.get_state()["children"]

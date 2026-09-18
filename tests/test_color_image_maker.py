import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mimg
import numpy as np
import pytest
from astropy.io import fits
from astropy.visualization import (
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)

from astro_notebooks.color_image_maker import ColorImageMaker

# Must be a multiple of the 8x thumbnail reduction, and large enough for the
# 512 pixel background boxes used on the full-size frames.
IMAGE_SHAPE = (1024, 1024)
REDUCED_SHAPE = (128, 128)
FILTERS = ["rp", "V", "B"]


def _write_combined_images(directory, object_name, seed=42):
    directory.mkdir()
    rng = np.random.default_rng(seed)
    # A gradient across the frame gives the background subtraction
    # something to remove.
    gradient = np.linspace(0.0, 300.0, IMAGE_SHAPE[1])[np.newaxis, :]
    for filt in FILTERS:
        data = rng.uniform(100.0, 1000.0, size=IMAGE_SHAPE) + gradient
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header["BUNIT"] = "adu"
        hdu.header["OBJECT"] = object_name
        hdu.writeto(directory / f"combined_light_filter_{filt}.fit")
    return directory


@pytest.fixture
def combined_dir(tmp_path):
    """Directory holding one small synthetic combined image per filter."""
    return _write_combined_images(tmp_path / "combined", "m 101")


@pytest.fixture
def maker(combined_dir):
    return ColorImageMaker(str(combined_dir))


def test_construction_loads_images(maker):
    assert maker.object_name == "m 101"
    for color in ["red", "green", "blue"]:
        viewer = maker.image_widgets[color]
        # The cuts come from the level slider, not the viewer's default.
        assert isinstance(viewer.get_cuts(), ManualInterval)
        assert isinstance(viewer.get_stretch(), LinearStretch)
        assert maker.sc_raw[color].shape == REDUCED_SHAPE
        assert maker.sc_raw_f[color].shape == IMAGE_SHAPE


def test_level_slider_sets_cuts(maker):
    green_before = maker.sc_raw["green"].copy()
    red_before = maker.sc_raw["red"].copy()

    maker.level_sliders["red"].value = (200.0, 900.0)

    cuts = maker.image_widgets["red"].get_cuts()
    assert (cuts.vmin, cuts.vmax) == (200.0, 900.0)
    assert not np.allclose(red_before, maker.sc_raw["red"])
    assert maker.sc_raw["red"].shape == REDUCED_SHAPE
    assert maker.sc_raw_f["red"].shape == IMAGE_SHAPE
    # Only the red channel should have been touched.
    np.testing.assert_array_equal(green_before, maker.sc_raw["green"])


@pytest.mark.parametrize(
    "name,stretch_class", [("log", LogStretch), ("sqrt", SqrtStretch)]
)
def test_stretch_chooser_sets_stretch(maker, name, stretch_class):
    before = {c: maker.sc_raw[c].copy() for c in maker.image_widgets}
    before_full = maker.sc_raw_f["blue"].copy()

    maker.stretch_chooser.value = name

    for color, viewer in maker.image_widgets.items():
        assert isinstance(viewer.get_stretch(), stretch_class)
        assert not np.allclose(before[color], maker.sc_raw[color])
    assert not np.allclose(before_full, maker.sc_raw_f["blue"])


def test_background_subtraction_keeps_cuts_and_stretch(maker):
    maker.level_sliders["blue"].value = (200.0, 900.0)
    maker.stretch_chooser.value = "sqrt"
    data_before = maker.data_sm["blue"].copy()

    maker.subtract_bkgd_checkbox.value = True

    assert not np.allclose(data_before, maker.data_sm["blue"])
    # The images are reloaded into the viewers; the settings must survive.
    cuts = maker.image_widgets["blue"].get_cuts()
    assert (cuts.vmin, cuts.vmax) == (200.0, 900.0)
    for viewer in maker.image_widgets.values():
        assert isinstance(viewer.get_stretch(), SqrtStretch)

    maker.subtract_bkgd_checkbox.value = False
    np.testing.assert_array_equal(data_before, maker.data_sm["blue"])


def test_save_tab_renders_and_saves_full_resolution(maker, tmp_path, monkeypatch):
    # The image is saved relative to the cwd.
    monkeypatch.chdir(tmp_path)
    maker.r_slider.value = 0.7

    # Selecting the save tab generates the full resolution image.
    maker.widget.selected_index = 2

    filename_input, button_row = maker.widget.children[2].children[:2]
    filename_input.value = "test"
    button_row.children[0].click()

    saved = tmp_path / "m 101-test-color.png"
    assert saved.exists()
    rgb = mimg.imread(saved)
    assert rgb.shape[:2] == IMAGE_SHAPE


def test_setting_image_directory_reloads(maker, tmp_path):
    red_before = maker.data_sm["red"].copy()
    other_dir = _write_combined_images(tmp_path / "other", "ngc 7331", seed=7)

    maker.image_directory = str(other_dir)

    assert maker.object_name == "ngc 7331"
    assert not np.allclose(red_before, maker.data_sm["red"])
    for viewer in maker.image_widgets.values():
        assert isinstance(viewer.get_cuts(), ManualInterval)

import io
import tracemalloc

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mimg
import numpy as np
import pytest
from PIL import Image
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.visualization import (
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)

from astro_notebooks.color_image_maker import (
    BAND_ROWS,
    COLORS,
    REDUCE,
    SAVE_REDUCE,
    ColorImageMaker,
    blank_missing_pixels,
    block_mean_and_noise,
    block_mean_band,
    iter_bands,
    png_bytes,
    preview_plane,
    read_frame,
    reduced_rgb_uint8,
    rgb_uint8,
    scaled_band,
    subtract_background_band,
)


def _repeated_background(background_sm, band_shape, start=0):
    """Scale a reduced background up to a band of a frame by repeating it.

    This is the way the background used to come off a band, and what the
    broadcast the code does now has to agree with, pixel for pixel.
    """
    rows = background_sm[start // REDUCE:-(-(start + band_shape[0]) // REDUCE)]
    full = np.repeat(np.repeat(rows, REDUCE, axis=0), REDUCE, axis=1)
    return full[:band_shape[0], :band_shape[1]]

# Must be a multiple of the 8x thumbnail reduction, and large enough for the
# 512 pixel background boxes used on the full-size frames.
IMAGE_SHAPE = (1024, 1024)
REDUCED_SHAPE = (128, 128)
# Several bands of rows, for the tests that measure memory.
BIG_IMAGE_SHAPE = (2048, 2048)
FILTERS = ["rp", "V", "B"]


def _write_combined_images(directory, object_name, seed=42):
    """Write one synthetic combined image per filter into a new directory.

    The files have the names and the ``OBJECT`` header that
    ``ColorImageMaker`` expects. ``seed`` makes the pixel values differ
    between directories, and the directory is returned.
    """
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
def big_maker(tmp_path):
    """A ``ColorImageMaker`` built from frames of a more realistic size.

    Big enough that one band of rows is a small fraction of a frame,
    which is what makes a measurement of the memory a slider move costs
    mean anything.
    """
    directory = tmp_path / "big"
    directory.mkdir()
    rng = np.random.default_rng(11)
    for filt in FILTERS:
        data = rng.uniform(100.0, 1000.0, size=BIG_IMAGE_SHAPE).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["BUNIT"] = "adu"
        hdu.header["OBJECT"] = "ngc 7331"
        hdu.writeto(directory / f"combined_light_filter_{filt}.fit")
    return ColorImageMaker(str(directory))


@pytest.fixture
def maker(combined_dir):
    """A ``ColorImageMaker`` built from the synthetic combined images."""
    return ColorImageMaker(str(combined_dir))


def test_construction_loads_images(maker):
    """Making the widget loads all three images into their viewers.

    This is the call that failed with ``'ImageWidget' object has no
    attribute 'load_array'`` after astrowidgets 0.6 changed its viewer
    API. Each viewer must end up with the level slider's cuts rather than
    the viewer's default, a linear stretch, and the reduced image to show;
    the frames stay at full size and the object name comes from the
    header.
    """
    assert maker.object_name == "m 101"
    for color in COLORS:
        viewer = maker.image_widgets[color]
        # The cuts come from the level slider, not the viewer's default.
        assert isinstance(viewer.get_cuts(), ManualInterval)
        assert isinstance(viewer.get_stretch(), LinearStretch)
        assert maker._preview_plane(color).shape == REDUCED_SHAPE
        assert maker.data[color].shape == IMAGE_SHAPE


def test_level_slider_sets_cuts(maker):
    """Moving one channel's level slider sets that viewer's cuts.

    That channel of the preview is thrown away, to be made again with
    the new cuts when the preview is next looked at, and the other two
    are left as they were rather than being recomputed, which would mean
    two more passes over a frame for nothing.
    """
    green_before = maker._preview_plane("green")
    red_before = maker._preview_plane("red").copy()

    maker.level_sliders["red"].value = (200.0, 900.0)

    cuts = maker.image_widgets["red"].get_cuts()
    assert (cuts.vmin, cuts.vmax) == (200.0, 900.0)
    # The preview is not on screen, so the red plane is only dropped.
    assert "red" not in maker.preview_planes

    maker.widget.selected_index = 1

    assert not np.allclose(red_before, maker.preview_planes["red"])
    assert maker.preview_planes["red"].shape == REDUCED_SHAPE
    # Only the red channel should have been touched.
    assert maker.preview_planes["green"] is green_before


@pytest.mark.parametrize(
    "name,stretch_class", [("log", LogStretch), ("sqrt", SqrtStretch)]
)
def test_stretch_chooser_sets_stretch(maker, name, stretch_class):
    """The stretch dropdown sets an astropy stretch on every viewer.

    The dropdown holds names, but astrowidgets 0.6 only accepts stretch
    objects, so each name must map to the right class. The stretch is
    part of every colour of the preview, so all three are out of date
    and come out different when they are next made.
    """
    before = {c: maker._preview_plane(c).copy() for c in COLORS}

    maker.stretch_chooser.value = name

    assert maker.preview_planes == {}
    for color, viewer in maker.image_widgets.items():
        assert isinstance(viewer.get_stretch(), stretch_class)
        assert not np.allclose(before[color], maker._preview_plane(color))


def test_background_subtraction_keeps_cuts_and_stretch(maker):
    """Toggling background subtraction keeps the cuts and the stretch.

    Subtracting the background reloads the images into the viewers, and
    a reload must not throw away the settings the user has already
    chosen. Unticking the box gives back exactly the original data.
    """
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
    """The save tab shows a small PNG and writes the full size one.

    Selecting the tab builds the full resolution colour image and shows a
    reduced PNG of it, rather than drawing the whole thing as a figure.
    The save button writes it to the cwd under a name made from the
    object name and the text the user typed, at the full image size.
    """
    # The image is saved relative to the cwd.
    monkeypatch.chdir(tmp_path)
    maker.mix_sliders["red"].value = 0.7

    # Selecting the save tab generates the full resolution image.
    maker.widget.selected_index = 2

    filename_input, button_row = maker.widget.children[2].children[:2]
    shown = Image.open(io.BytesIO(maker.widget.children[2].children[3].value))
    assert shown.format == "PNG"
    assert shown.size == (IMAGE_SHAPE[1] // 4, IMAGE_SHAPE[0] // 4)

    filename_input.value = "test"
    button_row.children[0].click()

    saved = tmp_path / "m 101-test-color.png"
    assert saved.exists()
    rgb = mimg.imread(saved)
    assert rgb.shape[:2] == IMAGE_SHAPE


def test_save_tab_shows_the_image_it_would_save(maker, tmp_path, monkeypatch):
    """The picture on the save tab is the file that would be written.

    The tab used to build the whole full size image and hand it to Pillow
    to be reduced. It is now built at the size it is shown, band by band,
    and the order of operations is what makes that the same picture: each
    band is scaled to the bytes that would be saved and only then
    averaged. Averaging the frames first, as the notebook once did to the
    preview, would lift the sky instead.
    """
    monkeypatch.chdir(tmp_path)
    maker.level_sliders["red"].value = (100.0, 900.0)
    maker.mix_sliders["blue"].value = 0.3

    maker.widget.selected_index = 2

    shown = np.asarray(
        Image.open(io.BytesIO(maker.widget.children[2].children[3].value))
    )
    saved = _full_res_rgb(maker)
    assert shown.shape == (IMAGE_SHAPE[0] // 4, IMAGE_SHAPE[1] // 4, 3)
    for plane in range(3):
        np.testing.assert_allclose(
            shown[:, :, plane], _block_means(saved[:, :, plane], factor=4), atol=1
        )


def test_opening_the_save_tab_makes_nothing_the_size_of_the_saved_image(big_maker):
    """Opening the save tab does not build the image that would be saved.

    It used to build all of it, 50 MB of bytes at 4096 by 4096, and hand
    it to Pillow, which has no way of encoding an RGB array without
    copying it, for the sake of a picture a quarter of the size on a
    side: about 100 MB against the gigabyte a student has on the hub, and
    the largest transient left on the tab that was killing kernels. The
    picture is now made band by band at the size it is shown, so opening
    the tab costs a band and the picture rather than the image itself.
    """
    saved_bytes = 3 * big_maker.data["red"].size
    band_bytes = BAND_ROWS * BIG_IMAGE_SHAPE[1] * np.dtype(np.float32).itemsize
    # Opening the tab once first, so that what is measured is the picture
    # being made and not whatever a widget does the first time it is shown.
    # The cuts are then moved, since the picture is otherwise kept rather
    # than made again and there would be nothing to measure.
    big_maker.widget.selected_index = 2
    big_maker.widget.selected_index = 0
    big_maker.level_sliders["red"].value = (100.0, 900.0)

    tracemalloc.start()
    try:
        big_maker.widget.selected_index = 2
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # The picture was made, and is what was measured.
    assert big_maker.widget.children[2].children[3].value
    # A few bands' worth, since the cuts and the stretch do not make one
    # array of a band, and a few bands are a fraction of the image.
    assert peak < 4 * band_bytes < saved_bytes


def test_saving_makes_nothing_the_size_of_the_saved_image(
    big_maker, tmp_path, monkeypatch
):
    """Writing the file does not build the whole image to hand to Pillow.

    Saving used to make the image as one array of bytes, 48 MB at 4096 by
    4096, which Pillow then copied into a buffer of its own before
    encoding it: a hundred-odd MB on top of the three frames, at the one
    moment a student is sure to reach. The file is now pasted together a
    band of rows at a time, so nothing the size of the image is made.
    """
    monkeypatch.chdir(tmp_path)
    saved_bytes = 3 * big_maker.data["red"].size
    band_bytes = BAND_ROWS * BIG_IMAGE_SHAPE[1] * np.dtype(np.float32).itemsize
    filename_input, button_row = big_maker.widget.children[2].children[:2]
    filename_input.value = "mem"

    tracemalloc.start()
    try:
        button_row.children[0].click()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # The file was written, and is what was measured.
    saved = tmp_path / "ngc 7331-mem-color.png"
    assert saved.exists()
    assert Image.open(saved).size == (BIG_IMAGE_SHAPE[1], BIG_IMAGE_SHAPE[0])
    # A few bands' worth, since the cuts and the stretch do not make one
    # array of a band, and a few bands are a fraction of the image.
    assert peak < 4 * band_bytes < saved_bytes


def test_save_writes_the_images_loaded_now(maker, tmp_path, monkeypatch):
    """Saving writes the object that is loaded, not one loaded earlier.

    The save tab used to keep the full size image it made when it was
    opened. Loading another directory while the tab was showing left that
    image in place, so the first object's pixels were written under the
    second object's name.
    """
    monkeypatch.chdir(tmp_path)
    maker.widget.selected_index = 2
    other_dir = _write_combined_images(tmp_path / "other", "ngc 7331", seed=7)

    maker.image_directory = str(other_dir)
    maker.widget.children[2].children[1].children[0].click()

    saved = np.asarray(Image.open(tmp_path / "ngc 7331--color.png"))
    np.testing.assert_array_equal(saved, _full_res_rgb(maker))


def test_setting_image_directory_redraws_what_is_on_screen(maker, tmp_path):
    """Loading another directory replaces the images already being shown.

    Loading draws nothing, because a new widget has nothing on screen yet.
    After a reload, though, the picture on the save tab, if that tab was
    open, was still of the object loaded before, even though saving wrote
    the new one. The preview is not on screen here, so it is only marked
    as out of date, and shows the new object when its tab is opened.
    """
    maker.widget.selected_index = 2
    shown = maker.widget.children[2].children[3]
    png_before = shown.value
    red_before = maker._preview_plane("red").copy()
    other_dir = _write_combined_images(tmp_path / "other", "ngc 7331", seed=7)

    maker.image_directory = str(other_dir)

    assert shown.value != png_before
    assert maker.preview_planes == {}

    maker.widget.selected_index = 1

    assert set(maker.preview_planes) == set(COLORS)
    assert not np.allclose(red_before, maker.preview_planes["red"])


def test_setting_image_directory_redraws_a_preview_that_is_on_screen(
    maker, tmp_path
):
    """Loading another directory while the preview is open redraws it.

    With the preview's tab on screen there is no tab change to wait for,
    so the new object has to be drawn as soon as it is loaded.
    """
    maker.widget.selected_index = 1
    red_before = maker.preview_planes["red"].copy()
    other_dir = _write_combined_images(tmp_path / "other", "ngc 7331", seed=7)

    maker.image_directory = str(other_dir)

    assert set(maker.preview_planes) == set(COLORS)
    assert not np.allclose(red_before, maker.preview_planes["red"])


def test_preview_is_drawn_only_while_its_tab_is_open(maker, monkeypatch):
    """The preview is drawn when it can be seen, and not otherwise.

    Every drawing is a pass over a full size frame for each colour that
    has changed, and a matplotlib figure. On the first tab that used to
    be paid on every move of a level slider, for a picture on another
    tab. Now nothing is drawn until the preview's tab is opened, which
    also means the tab is no longer blank the first time it is opened;
    it is not drawn again on coming back to it with nothing changed; and
    while it is open a slider redraws it straight away.
    """
    drawn = []
    update = maker._update_preview
    monkeypatch.setattr(
        maker, "_update_preview", lambda: (drawn.append(1), update())[1]
    )

    maker.level_sliders["red"].value = (200.0, 900.0)
    maker.stretch_chooser.value = "sqrt"
    maker.subtract_bkgd_checkbox.value = True
    assert drawn == []
    assert maker.preview_planes == {}

    maker.widget.selected_index = 1
    assert len(drawn) == 1
    assert set(maker.preview_planes) == set(COLORS)

    maker.widget.selected_index = 0
    maker.widget.selected_index = 1
    assert len(drawn) == 1

    maker.mix_sliders["red"].value = 0.8
    assert len(drawn) == 2


def test_the_save_picture_is_made_only_when_it_has_changed(maker, monkeypatch):
    """The save tab makes its picture when it must, and not on every visit.

    Making it is a pass over all three frames, about half a second on
    full size images, and it used to be paid every time the tab was
    opened, for a picture that could not have changed since the last
    one. It is made the first time the tab is opened, not again on
    coming back to it with nothing changed, and again once a level
    slider, the mixer, the stretch or the background box has changed
    what it would show.
    """
    made = []
    refresh = maker._refresh_save
    monkeypatch.setattr(
        maker, "_refresh_save", lambda: (made.append(1), refresh())[1]
    )

    maker.widget.selected_index = 2
    assert len(made) == 1

    maker.widget.selected_index = 0
    maker.widget.selected_index = 2
    assert len(made) == 1

    maker.widget.selected_index = 0
    maker.level_sliders["red"].value = (200.0, 900.0)
    # Nothing is drawn for a tab nobody is looking at.
    assert len(made) == 1
    assert maker._save_stale

    maker.widget.selected_index = 2
    assert len(made) == 2

    for change in [
        lambda: setattr(maker.mix_sliders["blue"], "value", 0.2),
        lambda: setattr(maker.stretch_chooser, "value", "sqrt"),
        lambda: setattr(maker.subtract_bkgd_checkbox, "value", True),
    ]:
        maker.widget.selected_index = 0
        change()
        maker.widget.selected_index = 2
    assert len(made) == 5


def test_changing_the_picture_while_the_save_tab_is_open_redraws_it(maker):
    """A change made while the save tab is on screen reaches the picture.

    There is no tab change to wait for then, so the picture has to be
    made again straight away, as the preview is, or the tab would go on
    showing an image the widget would no longer save.
    """
    maker.widget.selected_index = 2
    shown = maker.widget.children[2].children[3]
    png_before = shown.value

    maker.mix_sliders["red"].value = 0.9

    assert shown.value != png_before
    assert not maker._save_stale
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(shown.value))),
        maker._reduced_rgb(),
    )


def test_setting_image_directory_reloads(maker, tmp_path):
    """Assigning ``image_directory`` loads the images from the new directory.

    The object name and the image data both change, and the viewers are
    again given the level slider's cuts, which goes through the same
    viewer calls as construction.
    """
    red_before = maker.data_sm["red"].copy()
    other_dir = _write_combined_images(tmp_path / "other", "ngc 7331", seed=7)

    maker.image_directory = str(other_dir)

    assert maker.object_name == "ngc 7331"
    assert not np.allclose(red_before, maker.data_sm["red"])
    for viewer in maker.image_widgets.values():
        assert isinstance(viewer.get_cuts(), ManualInterval)


# ----------------------------------------------------------------------
# The pieces the preview and the saved image are both made of
# ----------------------------------------------------------------------

# Tall enough to be split into more than one band, with a last band that
# is shorter than the rest.
BANDED_SHAPE = (600, 48)


def _full_res_rgb(maker):
    """The finished image at full size, the bytes the file is written from.

    Saving pastes the image into the file a band of rows at a time and
    never holds the whole thing, so a test that wants all of it to
    compare against asks `rgb_uint8` for it here instead.
    """
    return rgb_uint8(maker.data, {c: maker._scaling(c) for c in maker._colors})


def _block_means(image, factor=REDUCE):
    """The mean of each block of the image, written out block by block.

    A slow but obvious reference for the block means: it works for any image
    size, and a block that hangs off the edge of the image is the mean of
    the pixels that are really there.
    """
    return np.array(
        [
            [
                image[row : row + factor, col : col + factor].mean()
                for col in range(0, image.shape[1], factor)
            ]
            for row in range(0, image.shape[0], factor)
        ]
    )


def _old_style_rgb(frames, intervals, stretch, weights):
    """The colour image the way the notebook used to make it.

    The arithmetic of the old ``_rgb_scaling``, copied here so that the
    image that gets saved can be compared with what the old code would
    have produced: cut and stretch each frame, multiply by the mixer
    weight and by two, and pull anything above one back down to one.
    """
    comb = np.zeros(list(frames[COLORS[0]].shape) + [3])
    for plane, color in enumerate(COLORS):
        comb[:, :, plane] = weights[color] * stretch(
            intervals[color](frames[color])
        )
    comb = 2 * comb
    comb[comb > 1] = 1.0
    return comb


@pytest.fixture
def ragged_frames():
    """Three frames with different pixels missing along their edges.

    Reprojection leaves a different set of empty pixels in each filter,
    which is what makes the colour of the frame edge worth testing. The
    shape is a multiple of the reduction but not of the band height.
    """
    rng = np.random.default_rng(1234)
    frames = {
        color: rng.uniform(0.0, 900.0, size=BANDED_SHAPE).astype(np.float32)
        for color in COLORS
    }
    frames["red"][:3, :] = np.nan
    frames["green"][:1, :] = np.nan
    frames["green"][:, -5:] = np.nan
    frames["blue"][-2:, :] = np.nan
    return frames


def _scalings(intervals, stretch, weights):
    """Put cuts, a stretch and weights in the form ``rgb_uint8`` takes them.

    That is, for each colour, the keyword arguments that scale it.
    """
    return {
        color: dict(interval=intervals[color], stretch=stretch,
                    weight=weights[color])
        for color in COLORS
    }


@pytest.fixture
def cuts_and_weights():
    """Black and white points and mixer weights, one set per colour.

    Different values for each colour so that a plane ending up in the
    wrong place would show.
    """
    intervals = {
        "red": ManualInterval(0.0, 500.0),
        "green": ManualInterval(50.0, 700.0),
        "blue": ManualInterval(-10.0, 300.0),
    }
    weights = {"red": 0.8, "green": 0.5, "blue": 0.65}
    return intervals, weights


def test_iter_bands_covers_every_row():
    """The bands run in order, do not overlap, and reach the last row.

    The last band is short when the frame does not divide evenly, and
    every band but that one is the full height. Each band comes with the
    rows of the reduced image it averages into, which run on from one
    band to the next and take in the short block at the bottom.
    """
    bands = list(iter_bands(600))

    assert [(start, stop) for start, stop, _ in bands] == [
        (0, 256), (256, 512), (512, 600)
    ]
    assert [rows for _, _, rows in bands] == [
        slice(0, 32), slice(32, 64), slice(64, 75)
    ]
    assert [rows for _, _, rows in iter_bands(512, factor=SAVE_REDUCE)] == [
        slice(0, 64), slice(64, 128)
    ]


def test_block_mean_and_noise_matches_block_reduce():
    """Averaging band by band gives what astropy gives in one go.

    ``block_reduce`` is what the viewers' images used to be made with,
    and it makes an array the size of the frame on the way, which is the
    only reason it is no longer used.
    """
    rng = np.random.default_rng(99)
    image = rng.uniform(0.0, 1000.0, size=BANDED_SHAPE).astype(np.float32)

    reduced, _ = block_mean_and_noise(image)

    assert reduced.dtype == np.float32
    np.testing.assert_allclose(
        reduced, block_reduce(image, REDUCE, func=np.mean), rtol=1e-6
    )


def test_the_noise_is_the_sky_and_not_the_stars():
    """The noise found is the sky's, whatever else is in the frame.

    The frame has a bright sky with Gaussian noise of a known size, stars
    much brighter than the noise, an edge with no data, and a shape that
    is not a whole number of blocks. None of those may move the answer:
    a star's own shape is not noise, a block with a missing pixel has no
    scatter to give, and the short blocks at the edges are left out. The
    sky is bright on purpose, since that is where single precision would
    lose the scatter in the rounding of the mean.
    """
    rng = np.random.default_rng(5)
    image = rng.normal(40_000.0, 30.0, size=(603, 515)).astype(np.float32)
    for row, col in rng.integers(20, 500, size=(40, 2)):
        image[row:row + 4, col:col + 4] += 5000.0
    image[:, :11] = np.nan

    assert block_mean_and_noise(image)[1] == pytest.approx(30.0, rel=0.03)


def test_the_noise_of_a_frame_with_no_whole_block_is_zero():
    """A frame too small, or too empty, to measure gets no noise added.

    Nothing can be said about the noise of a frame smaller than one block
    or of one with no data in it, and zero leaves the viewers' images as
    they were rather than filling them with NaN.
    """
    assert block_mean_and_noise(np.ones((5, 40), dtype=np.float32))[1] == 0.0
    assert block_mean_and_noise(
        np.full((64, 64), np.nan, dtype=np.float32)
    )[1] == 0.0


@pytest.mark.parametrize("shape", [(603, 515), (5, 40)])
def test_block_mean_of_a_frame_that_is_not_whole_blocks(shape):
    """Every block is the mean of the pixels it really has.

    The reduced image is what the background is fitted to and what the
    viewers show. The shapes are a frame that is not a whole number of
    blocks, whose last blocks are short, and one too small to hold a
    single whole block.
    """
    rng = np.random.default_rng(13)
    image = rng.normal(40_000.0, 30.0, size=shape).astype(np.float32)

    reduced, _ = block_mean_and_noise(image)

    np.testing.assert_allclose(reduced, _block_means(image), rtol=1e-6)


def test_block_mean_and_noise_leaves_out_blocks_with_no_data():
    """A block with a missing pixel gives no noise and does not poison it.

    The scatter of such a block is a blank, and a blank among the numbers
    the median is taken of would make the noise itself a blank, which
    would then be added to every pixel of the viewer's image.
    """
    rng = np.random.default_rng(17)
    image = rng.normal(300.0, 20.0, size=(64, 64)).astype(np.float32)
    image[:, :11] = np.nan

    _, noise = block_mean_and_noise(image)

    assert noise == pytest.approx(20.0, rel=0.2)


def test_block_mean_band_leaves_missing_pixels_missing():
    """A block containing a pixel with no data has no data either.

    The reduced images the viewers show are made this way, so a missing
    pixel must not quietly turn into a number.
    """
    band = np.ones((8, 16), dtype=np.float32)
    band[3, 3] = np.nan

    means = block_mean_band(band)

    assert np.isnan(means[0, 0])
    assert means[0, 1] == 1.0


def test_blank_missing_pixels_blanks_the_union(ragged_frames):
    """A pixel missing from one frame is blanked in all three.

    Left alone, such a pixel keeps its colour in the frames that do have
    it and comes out as a coloured fringe along the edge of the saved
    image.
    """
    frames = [ragged_frames[color] for color in COLORS]
    missing = np.zeros(BANDED_SHAPE, dtype=bool)
    for frame in frames:
        missing |= np.isnan(frame)

    blank_missing_pixels(frames)

    for frame in frames:
        np.testing.assert_array_equal(np.isnan(frame), missing)


def test_scaled_band_is_the_clipped_weighted_stretch(cuts_and_weights):
    """One band is cut, stretched, weighted, doubled and clipped.

    This is the one expression the preview and the saved image are both
    made of. Pixels with no data come out black rather than as a blank,
    because a blank cannot be written into an image file.
    """
    intervals, weights = cuts_and_weights
    band = np.linspace(-50.0, 900.0, 64, dtype=np.float32).reshape(8, 8)
    band[0, 0] = np.nan
    stretch = LogStretch()

    scaled = scaled_band(band, intervals["red"], stretch, weights["red"])

    expected = 2 * weights["red"] * stretch(intervals["red"](band))
    expected = np.nan_to_num(np.clip(expected, 0, 1))
    np.testing.assert_allclose(scaled, expected, rtol=1e-12)


def test_scaled_band_leaves_the_frame_it_is_given_alone(cuts_and_weights):
    """Scaling a band never writes into the frame it came from.

    The cuts are put back into the array that subtracting the background
    makes, which saves a copy of every band, and that array belongs to
    this function. With no background to take off, though, the band is a
    view of the caller's frame, and the frame has to come out of the pass
    exactly as it went in.
    """
    intervals, weights = cuts_and_weights
    frame = np.linspace(-50.0, 900.0, 256, dtype=np.float32).reshape(16, 16)
    background_sm = np.linspace(0.0, 20.0, 4, dtype=np.float32).reshape(2, 2)
    before = frame.copy()

    scaled_band(frame, intervals["green"], LogStretch(), weights["green"])

    np.testing.assert_array_equal(frame, before)

    scaled = scaled_band(frame, intervals["green"], LogStretch(),
                         weights["green"], background=background_sm)

    np.testing.assert_array_equal(frame, before)
    # Taking the background off first gives an array of its own, which is
    # the copy the in-place path is meant to be identical to.
    np.testing.assert_array_equal(
        scaled,
        scaled_band(frame - _repeated_background(background_sm, frame.shape),
                    intervals["green"], LogStretch(), weights["green"]),
    )


def test_subtract_background_band_spreads_each_value_over_its_block():
    """A background fitted to the reduced image scales back up by block.

    Each value of the fit covers one block of the frame, and the rows it
    is taken off are the rows of the frame that the band holds, even when
    the band starts part way down the fit.
    """
    background_sm = np.arange(12, dtype=np.float32).reshape(3, 4)
    band = np.zeros((12, 30), dtype=np.float32)

    taken_off = -subtract_background_band(band, background_sm, start=8)

    assert taken_off.shape == (12, 30)
    # Row 8 of the frame is the second row of the reduced background.
    np.testing.assert_array_equal(taken_off[0, :REDUCE], np.full(REDUCE, 4.0))
    np.testing.assert_array_equal(taken_off[7, :REDUCE], np.full(REDUCE, 4.0))
    # The band stops four rows into the third row of the background, and
    # the frame stops six columns into its fourth value.
    np.testing.assert_array_equal(taken_off[8, :REDUCE], np.full(REDUCE, 8.0))
    np.testing.assert_array_equal(taken_off[8, -6:], np.full(6, 11.0))


@pytest.mark.parametrize("shape", [(256, 128), (100, 70), (12, 30), (5, 6)])
def test_subtract_background_band_is_the_old_repeat_bit_for_bit(shape):
    """Broadcasting the background off a band gives what repeating it gave.

    Repeating the fit up to the size of the band cost a second array as
    big as the band, which is what this replaces, so the answer has to be
    identical to the last bit and not merely close: the bytes written to
    the saved file are made of it. The shapes include frames whose rows
    and columns are not whole numbers of blocks, where the blocks along
    two edges are short.
    """
    rng = np.random.default_rng(3)
    band = rng.normal(500.0, 30.0, size=shape).astype(np.float32)
    background_sm = rng.normal(
        480.0, 5.0, size=(-(-(shape[0] + 8) // REDUCE), -(-shape[1] // REDUCE))
    ).astype(np.float32)

    subtracted = subtract_background_band(band, background_sm, start=REDUCE)

    assert subtracted.dtype == np.float32
    np.testing.assert_array_equal(
        subtracted,
        band - _repeated_background(background_sm, shape, start=REDUCE),
    )


def test_subtract_background_band_leaves_the_band_alone():
    """The band a caller hands over is a view of a frame it keeps.

    Writing the answer into it would take the background off the frame
    the viewers and the saved image are made from, once per band, every
    time the preview was redrawn.
    """
    rng = np.random.default_rng(11)
    frame = rng.normal(500.0, 30.0, size=(32, 64)).astype(np.float32)
    background_sm = rng.normal(480.0, 5.0, size=(4, 8)).astype(np.float32)
    before = frame.copy()

    subtract_background_band(frame[8:24], background_sm, start=8)

    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize(
    "stretch", [LinearStretch(), SqrtStretch(), LogStretch()]
)
def test_preview_plane_is_the_block_mean_of_the_full_size_result(
    ragged_frames, cuts_and_weights, stretch
):
    """The preview is the full size image averaged over blocks.

    This is the whole point of the change: the weight, the clip and the
    stretch are applied to every pixel and the averaging happens
    afterwards, so the preview shows what the saved image looks like at
    one eighth the size. The old code averaged first, which lifted the
    sky.
    """
    intervals, weights = cuts_and_weights
    frame = ragged_frames["green"]

    plane = preview_plane(frame, intervals["green"], stretch, weights["green"])

    full_size = scaled_band(frame, intervals["green"], stretch, weights["green"])
    np.testing.assert_allclose(plane, _block_means(full_size), rtol=1e-6)


def test_preview_plane_subtracts_the_scaled_up_background(
    ragged_frames, cuts_and_weights
):
    """A background given to the preview is the one the frame loses.

    The fit is made on the reduced image and scaled back up a band at a
    time, so the answer must match subtracting the scaled up background
    from the whole frame at once.
    """
    intervals, weights = cuts_and_weights
    frame = ragged_frames["blue"]
    background_sm = np.linspace(
        0.0, 40.0, (BANDED_SHAPE[0] // REDUCE) * (BANDED_SHAPE[1] // REDUCE)
    ).reshape(BANDED_SHAPE[0] // REDUCE, BANDED_SHAPE[1] // REDUCE)

    plane = preview_plane(
        frame,
        intervals["blue"],
        LogStretch(),
        weights["blue"],
        background=background_sm.astype(np.float32),
    )

    full_background = np.repeat(
        np.repeat(background_sm.astype(np.float32), REDUCE, axis=0), REDUCE, axis=1
    )
    full_size = scaled_band(
        frame - full_background, intervals["blue"], LogStretch(), weights["blue"]
    )
    np.testing.assert_allclose(plane, _block_means(full_size), rtol=1e-6)


@pytest.mark.parametrize(
    "stretch", [LinearStretch(), SqrtStretch(), LogStretch()]
)
def test_saved_image_is_what_the_old_path_wrote(
    ragged_frames, cuts_and_weights, stretch
):
    """The uint8 image is byte for byte the PNG the old code saved.

    The old code built a float image of the whole frame and handed it to
    ``matplotlib.image.imsave``, which truncated it to bytes and turned
    the pixels with no data black. Writing the bytes band by band has to
    give the same file, which it does once the pixels missing from any
    one frame have been blanked in all three.
    """
    intervals, weights = cuts_and_weights
    frames = {color: ragged_frames[color] for color in COLORS}
    blank_missing_pixels([frames[color] for color in COLORS])

    new = rgb_uint8(frames, _scalings(intervals, stretch, weights))

    buffer = io.BytesIO()
    mimg.imsave(buffer, _old_style_rgb(frames, intervals, stretch, weights),
                format="png")
    buffer.seek(0)
    old = np.asarray(Image.open(buffer).convert("RGB"))
    np.testing.assert_array_equal(new, old)


def test_reduced_rgb_uint8_is_the_saved_image_averaged(
    ragged_frames, cuts_and_weights
):
    """The picture the Save tab shows is the saved image, seen smaller.

    It is built band by band at a quarter of the size, so it never holds
    the saved image to reduce it, and each of its pixels has to come out
    as the mean of the block of saved bytes it stands for. The frame is
    several bands tall, with a short last band, so the rows of the
    picture a band fills are worth checking too, and the pixels with no
    data must be the black they are in the file rather than a blank.
    """
    intervals, weights = cuts_and_weights
    scalings = _scalings(intervals, LinearStretch(), weights)
    blank_missing_pixels([ragged_frames[color] for color in COLORS])

    shown = reduced_rgb_uint8(ragged_frames, scalings)

    saved = rgb_uint8(ragged_frames, scalings)
    assert shown.dtype == np.uint8
    assert shown.shape == (
        -(-BANDED_SHAPE[0] // SAVE_REDUCE), -(-BANDED_SHAPE[1] // SAVE_REDUCE), 3
    )
    for plane in range(3):
        np.testing.assert_allclose(
            shown[:, :, plane],
            _block_means(saved[:, :, plane], factor=SAVE_REDUCE),
            atol=1,
        )


def test_png_bytes_encodes_the_picture_it_is_given(ragged_frames, cuts_and_weights):
    """The Save tab is given a PNG of the picture, at the picture's size.

    Drawing the full size image as a matplotlib figure is what used to
    make the save tab unusable; a small PNG shows the same thing for a
    fraction of the memory.
    """
    intervals, weights = cuts_and_weights
    scalings = _scalings(intervals, LinearStretch(), weights)
    png = png_bytes(reduced_rgb_uint8(ragged_frames, scalings))

    shown = Image.open(io.BytesIO(png))
    assert shown.format == "PNG"
    assert shown.size == (
        -(-BANDED_SHAPE[1] // SAVE_REDUCE), -(-BANDED_SHAPE[0] // SAVE_REDUCE)
    )


def _write_images_with_empty_edges(directory, object_name):
    """Write combined images whose empty edges differ from filter to filter.

    Reprojecting the frames of a night onto one grid leaves pixels with
    no data around the edges, and not the same pixels in each filter.
    Each file here is missing a different strip, and the directory is
    returned.
    """
    directory.mkdir()
    rng = np.random.default_rng(3)
    edges = {"rp": (np.s_[:2, :]), "V": (np.s_[:, -3:]), "B": (np.s_[-1:, :])}
    for filt in FILTERS:
        data = rng.uniform(100.0, 1000.0, size=IMAGE_SHAPE).astype(np.float32)
        data[edges[filt]] = np.nan
        hdu = fits.PrimaryHDU(data)
        hdu.header["BUNIT"] = "adu"
        hdu.header["OBJECT"] = object_name
        hdu.writeto(directory / f"combined_light_filter_{filt}.fit")
    return directory


def test_loading_blanks_pixels_missing_from_any_one_filter(tmp_path):
    """A pixel missing from one filter is blank in all three once loaded.

    The frames keep their own empty edges otherwise, and a pixel that is
    black in the saved image because one colour has no data there must be
    black in the preview too.
    """
    directory = _write_images_with_empty_edges(tmp_path / "edges", "m 101")

    maker = ColorImageMaker(str(directory))

    missing = np.isnan(maker.data["red"])
    # The three strips together, with no pixel counted twice.
    assert missing.sum() == 2 * IMAGE_SHAPE[1] + 3 * IMAGE_SHAPE[0] - 6 + IMAGE_SHAPE[1] - 3
    for color in COLORS:
        assert maker.data[color].dtype == np.float32
        np.testing.assert_array_equal(np.isnan(maker.data[color]), missing)


def test_preview_is_the_saved_image_averaged(maker):
    """What the widget previews is what it would save, seen smaller.

    The preview used to average the frames first and stretch afterwards,
    which lifted the sky by as much as a tenth of the full range, so a
    student tuned the black point on an image that was not the one they
    got. The cuts, the stretch and the mixer weight now all come from the
    widgets and are applied before the averaging.
    """
    maker.level_sliders["green"].value = (30.0, 800.0)
    maker.stretch_chooser.value = "log"
    maker.mix_sliders["green"].value = 0.35

    plane = maker._preview_plane("green")

    full_size = scaled_band(
        maker.data["green"], ManualInterval(30.0, 800.0), LogStretch(), 0.35
    )
    np.testing.assert_allclose(plane, _block_means(full_size), rtol=1e-6)


def test_slider_move_makes_nothing_the_size_of_a_frame(big_maker):
    """Moving a slider does not allocate another full size image.

    Every copy of a frame counts against the gigabyte a student has on
    the hub, and the old code made several: a float64 image of the whole
    frame for each colour on every slider move. Working a band at a time
    should cost a small fraction of one frame.
    """
    frame_bytes = big_maker.data["red"].nbytes
    # With the preview on screen a slider move remakes its plane at once.
    # Opening it also warms up, so that what is measured is the move alone.
    big_maker.widget.selected_index = 1

    tracemalloc.start()
    try:
        big_maker.level_sliders["red"].value = (100.0, 800.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # The pass over the frame was made, and is what was measured.
    assert "red" in big_maker.preview_planes
    assert peak < frame_bytes


def test_background_on_then_off_gives_back_the_same_preview(maker):
    """Unticking the background box puts the preview back as it was.

    The background is no longer taken off a second copy of each frame but
    off a band at a time wherever the frame is used, so the frames
    themselves have to come through the round trip untouched, and the
    preview has to change while the box is ticked.
    """
    maker.level_sliders["red"].value = (50.0, 800.0)
    before = {color: maker._preview_plane(color).copy() for color in COLORS}
    frame_before = maker.data["red"].copy()

    maker.subtract_bkgd_checkbox.value = True
    assert not np.allclose(before["red"], maker._preview_plane("red"))

    maker.subtract_bkgd_checkbox.value = False

    for color in COLORS:
        np.testing.assert_array_equal(before[color], maker._preview_plane(color))
    np.testing.assert_array_equal(frame_before, maker.data["red"])


def test_one_background_fit_serves_the_preview_and_the_file(maker):
    """Preview and saved image lose the same background, fitted once.

    There used to be two fits, one on the reduced image for the preview
    and another on the frame itself for the file, so the two disagreed
    about the sky and the second cost a copy of every frame. The reduced
    fit is now the only one, stretched back up block by block.
    """
    maker.level_sliders["green"].value = (0.0, 700.0)

    maker.subtract_bkgd_checkbox.value = True

    assert set(maker.bkgd_sm) == set(COLORS)
    assert maker.bkgd_sm["green"].shape == REDUCED_SHAPE
    full_background = np.repeat(
        np.repeat(maker.bkgd_sm["green"], REDUCE, axis=0), REDUCE, axis=1
    )
    scaling = maker._scaling("green")
    full_size = scaled_band(
        maker.data["green"] - full_background,
        ManualInterval(0.0, 700.0),
        scaling["stretch"],
        scaling["weight"],
    )
    np.testing.assert_allclose(
        maker._preview_plane("green"), _block_means(full_size), rtol=1e-6
    )
    saved = _full_res_rgb(maker)
    np.testing.assert_array_equal(saved[:, :, 1], (full_size * 255).astype(np.uint8))


def _write_noisy_sky_images(directory, object_name, sky=300.0, noise=30.0):
    """Write combined images that are a flat sky with Gaussian noise.

    This is the case the viewers on the first tab used to get wrong: a
    black point set at the level of the sky. The directory is returned.
    """
    directory.mkdir()
    rng = np.random.default_rng(21)
    for filt in FILTERS:
        data = rng.normal(sky, noise, size=IMAGE_SHAPE).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["BUNIT"] = "adu"
        hdu.header["OBJECT"] = object_name
        hdu.writeto(directory / f"combined_light_filter_{filt}.fit")
    return directory


@pytest.mark.parametrize("subtract,black_point", [(False, 300.0), (True, 0.0)])
def test_viewer_shows_the_sky_as_bright_as_the_preview_does(
    tmp_path, subtract, black_point
):
    """With the black point on the sky, the viewer and the preview agree.

    The preview applies the cuts and the stretch to pixels that still
    have their noise and then averages; the viewer is given an averaged
    image and applies them afterwards. With a log stretch and the black
    point at the sky level that used to leave the viewer's sky at about
    a third of the brightness of the preview's, so levels chosen on the
    first tab gave a different picture on the second. With the noise put
    back the two agree on average, which the averaged image alone, also
    checked here, does not.

    The same holds with the background taken off, when the sky and so
    the black point are at zero. The noise is measured on the frame as
    it was read, which is the right size either way: the background that
    comes off is the same across a block, and the noise is the scatter
    within one.
    """
    maker = ColorImageMaker(str(_write_noisy_sky_images(tmp_path / "sky", "sky")))
    maker.stretch_chooser.value = "log"
    maker.subtract_bkgd_checkbox.value = subtract
    cuts = (black_point, black_point + 1200.0)
    maker.level_sliders["red"].value = cuts
    # A weight of 0.5 in the mix leaves the preview plane as the plain
    # stretched image, which is what a viewer shows.
    assert maker.mix_sliders["red"].value == 0.5

    def shown(image):
        scaled = LogStretch()(ManualInterval(*cuts)(image))
        return float(np.clip(scaled, 0, 1).mean())

    preview = float(maker._preview_plane("red").mean())
    without_noise = maker.data_sm["red"] - maker.noise_sm["red"]

    assert shown(maker.data_sm["red"]) == pytest.approx(preview, rel=0.05)
    assert shown(without_noise) < 0.5 * preview


def test_viewer_noise_is_as_big_as_the_noise_of_the_frame(maker):
    """What is added to a viewer's image is noise the size of the frame's.

    The image the background is fitted to stays the plain block mean,
    and what the viewer shows differs from it by noise with the scatter
    `block_mean_and_noise` finds in the full size frame.
    """
    for color in COLORS:
        reduced, noise = block_mean_and_noise(maker.data[color])
        np.testing.assert_array_equal(maker.data_sm_raw[color], reduced)
        added = maker.data_sm[color] - maker.data_sm_raw[color]
        assert added.std() == pytest.approx(noise, rel=0.05)
        assert abs(added.mean()) < 0.05 * added.std()


def test_loading_the_same_images_again_shows_the_same_noise(combined_dir):
    """Two widgets made from the same images show identical viewers.

    The noise is made up, but it is seeded, so a student who runs the
    notebook again sees the image they saw before rather than one that
    has shimmered. The colours do not share their noise.
    """
    first = ColorImageMaker(str(combined_dir))
    second = ColorImageMaker(str(combined_dir))

    for color in COLORS:
        np.testing.assert_array_equal(first.data_sm[color], second.data_sm[color])
    assert not np.array_equal(first.noise_sm["red"], first.noise_sm["green"])


def test_read_frame_gives_the_image_and_its_header(combined_dir):
    """A frame read band by band is the frame the file holds.

    The file is big-endian, and the frames are kept as native float32
    because they are the only full size arrays there are.
    """
    path = combined_dir / "combined_light_filter_V.fit"

    frame, header = read_frame(str(path))

    assert frame.dtype == np.float32
    assert frame.dtype.byteorder in "=|"
    assert header["OBJECT"] == "m 101"
    np.testing.assert_array_equal(frame, fits.getdata(str(path)))


def test_read_frame_finds_an_image_that_is_not_in_the_primary_hdu(tmp_path):
    """The image is read from the first HDU that has data.

    A compressed file, or one written with several extensions, has an
    empty primary HDU and its image, with the ``OBJECT`` keyword, in the
    first extension. ``CCDData.read`` reads such a file, so reading the
    frames band by band must too.
    """
    data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48)
    image_hdu = fits.ImageHDU(data)
    image_hdu.header["OBJECT"] = "m 101"
    path = tmp_path / "image_in_extension.fit"
    fits.HDUList([fits.PrimaryHDU(), image_hdu]).writeto(path)

    frame, header = read_frame(str(path))

    assert header["OBJECT"] == "m 101"
    np.testing.assert_array_equal(frame, data)


def test_box_ticked_with_no_background_fitted_takes_nothing_off(maker):
    """With the box ticked but no background fitted, nothing is subtracted.

    The box can be left ticked with no fit behind it, when the fit was
    skipped because no images were loaded or a reload failed after the
    old fits were thrown away. Reading the controls then raised a
    ``KeyError`` inside an observer, and the widget stopped responding
    without saying why.
    """
    maker.subtract_bkgd_checkbox.value = True
    maker.bkgd_sm.clear()
    maker.widget.selected_index = 1

    maker.level_sliders["red"].value = (150.0, 900.0)

    for color in COLORS:
        assert maker._scaling(color)["background"] is None
    assert "red" in maker.preview_planes


def test_loading_other_images_holds_no_more_than_three_frames(tmp_path):
    """Loading another set of images lets the old frames go first.

    The frames are the biggest things there are, and a student who
    combines again and points the widget at the result is loading a
    second set. Reading the new red frame while the old red, green and
    blue were all still held made four frames at the peak where three
    are ever needed.
    """
    directories = []
    for seed, name in enumerate(["first", "second"]):
        directory = tmp_path / name
        directory.mkdir()
        rng = np.random.default_rng(seed)
        for filt in FILTERS:
            data = rng.uniform(100.0, 1000.0, size=BIG_IMAGE_SHAPE)
            hdu = fits.PrimaryHDU(data.astype(np.float32))
            hdu.header["BUNIT"] = "adu"
            hdu.header["OBJECT"] = name
            hdu.writeto(directory / f"combined_light_filter_{filt}.fit")
        directories.append(str(directory))

    # Traced from before the first frames are read, so that letting them
    # go counts.
    tracemalloc.start()
    try:
        big = ColorImageMaker(directories[0])
        frame_bytes = big.data["red"].nbytes
        tracemalloc.reset_peak()
        big.image_directory = directories[1]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert big.object_name == "second"
    # Three frames and the small things made from them come to about
    # three and a half; with the old frames still held it was over four.
    assert peak < 3.9 * frame_bytes

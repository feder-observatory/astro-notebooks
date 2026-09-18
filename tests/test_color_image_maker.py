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
    COLORS,
    REDUCE,
    ColorImageMaker,
    background_band,
    blank_missing_pixels,
    block_mean,
    block_mean_band,
    iter_bands,
    preview_plane,
    reduced_png_bytes,
    rgb_uint8,
    scaled_band,
)

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
        assert maker.data_raw_unmod[color].shape == IMAGE_SHAPE


def test_level_slider_sets_cuts(maker):
    """Moving one channel's level slider sets that viewer's cuts.

    That channel of the preview is made again with the new cuts, and the
    other two are left as they were rather than being recomputed, which
    would mean two more passes over a frame for nothing.
    """
    green_before = maker._preview_plane("green")
    red_before = maker._preview_plane("red").copy()

    maker.level_sliders["red"].value = (200.0, 900.0)

    cuts = maker.image_widgets["red"].get_cuts()
    assert (cuts.vmin, cuts.vmax) == (200.0, 900.0)
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
    part of every colour of the preview, so all three are made again.
    """
    before = {c: maker._preview_plane(c).copy() for c in COLORS}

    maker.stretch_chooser.value = name

    for color, viewer in maker.image_widgets.items():
        assert isinstance(viewer.get_stretch(), stretch_class)
        assert not np.allclose(before[color], maker.preview_planes[color])


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
    maker.r_slider.value = 0.7

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


def _block_means(image, factor=REDUCE):
    """The mean of each block of the image, written out block by block.

    A slow but obvious reference for `block_mean`: it works for any image
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
    every band but that one is the full height.
    """
    bands = list(iter_bands(600, band_rows=256))

    assert bands == [(0, 256), (256, 512), (512, 600)]
    assert list(iter_bands(512, band_rows=256)) == [(0, 256), (256, 512)]


def test_block_mean_matches_block_reduce():
    """Averaging band by band gives what astropy gives in one go.

    ``block_reduce`` is what the viewers' images used to be made with,
    and it makes an array the size of the frame on the way, which is the
    only reason it is no longer used.
    """
    rng = np.random.default_rng(99)
    image = rng.uniform(0.0, 1000.0, size=BANDED_SHAPE).astype(np.float32)

    reduced = block_mean(image)

    assert reduced.dtype == np.float32
    np.testing.assert_allclose(
        reduced, block_reduce(image, REDUCE, func=np.mean), rtol=1e-6
    )


def test_block_mean_of_a_frame_that_does_not_divide_evenly():
    """A frame that is not a whole number of blocks still reduces.

    The blocks along the bottom and right-hand edges are short, and each
    one is the mean of the pixels it does have rather than of pixels that
    are not there.
    """
    rng = np.random.default_rng(7)
    image = rng.uniform(0.0, 1000.0, size=(100, 120)).astype(np.float32)

    reduced = block_mean(image)

    assert reduced.shape == (13, 15)
    np.testing.assert_allclose(reduced, _block_means(image), rtol=1e-6)


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


def test_background_band_repeats_each_reduced_value_over_its_block():
    """A background fitted to the reduced image scales back up by repeat.

    Each value of the fit covers one block of the frame, and the rows
    asked for are the rows of the frame that the band holds, even when
    the band ends part way through a block.
    """
    background_sm = np.arange(12, dtype=np.float32).reshape(3, 4)

    band = background_band(background_sm, 8, 20, n_cols=30)

    assert band.shape == (12, 30)
    # Row 8 of the frame is the second row of the reduced background.
    np.testing.assert_array_equal(band[0, :REDUCE], np.full(REDUCE, 4.0))
    np.testing.assert_array_equal(band[7, :REDUCE], np.full(REDUCE, 4.0))
    # The band stops four rows into the third row of the background, and
    # the frame stops six columns into its fourth value.
    np.testing.assert_array_equal(band[8, :REDUCE], np.full(REDUCE, 8.0))
    np.testing.assert_array_equal(band[8, -6:], np.full(6, 11.0))


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

    new = rgb_uint8(frames, intervals, stretch, weights)

    buffer = io.BytesIO()
    mimg.imsave(buffer, _old_style_rgb(frames, intervals, stretch, weights),
                format="png")
    buffer.seek(0)
    old = np.asarray(Image.open(buffer).convert("RGB"))
    np.testing.assert_array_equal(new, old)


def test_reduced_png_bytes_is_a_smaller_png(ragged_frames, cuts_and_weights):
    """The Save tab is given a PNG smaller than the image it shows.

    Drawing the full size image as a matplotlib figure is what used to
    make the save tab unusable; a reduced PNG shows the same thing for a
    fraction of the memory.
    """
    intervals, weights = cuts_and_weights
    png = reduced_png_bytes(
        rgb_uint8(ragged_frames, intervals, LinearStretch(), weights), factor=4
    )

    shown = Image.open(io.BytesIO(png))
    assert shown.format == "PNG"
    assert shown.size == (BANDED_SHAPE[1] // 4, BANDED_SHAPE[0] // 4)


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

    missing = np.isnan(maker.data_raw_unmod["red"])
    # The three strips together, with no pixel counted twice.
    assert missing.sum() == 2 * IMAGE_SHAPE[1] + 3 * IMAGE_SHAPE[0] - 6 + IMAGE_SHAPE[1] - 3
    for color in COLORS:
        assert maker.data_raw_unmod[color].dtype == np.float32
        np.testing.assert_array_equal(np.isnan(maker.data_raw_unmod[color]), missing)


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
    maker.g_slider.value = 0.35

    plane = maker.preview_planes["green"]

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
    # Warm up, so that what is measured is the slider move alone.
    big_maker._update_preview()

    tracemalloc.start()
    try:
        big_maker.level_sliders["red"].value = (100.0, 800.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < frame_bytes

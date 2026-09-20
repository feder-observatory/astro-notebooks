"""Interactive RGB colour image composer for astronomical FITS data."""

import io
import os

import ipywidgets as ipw
import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.visualization import (
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)
from astrowidgets.bqplot import ImageWidget
from matplotlib import pyplot as plt
from photutils.background import Background2D, MeanBackground

from .image_quality import _image_hdu

#: Which filter's frame becomes which colour of the image, in the order
#: the colours are stored in it.
COLOR_FILTERS = {'red': 'rp', 'green': 'V', 'blue': 'B'}

#: The colours of the image, in the order they are stored in it.
COLORS = list(COLOR_FILTERS)

#: The colour whose file's header the object's name is read from. All
#: three frames are of the same object, so any one of them would do.
OBJECT_NAME_COLOR = 'blue'

#: How many rows of a full size frame are worked on at a time. Nothing the
#: size of a whole frame is ever made; a band of this many rows is. It is a
#: multiple of `REDUCE`, so that the bands fall on the boundaries of the
#: blocks the preview averages over.
BAND_ROWS = 256

#: How many pixels on a side are averaged into one pixel of the preview and
#: of the images the viewers on the first tab show.
REDUCE = 8

#: How many pixels on a side are averaged into one pixel of the picture the
#: Save tab shows. It divides `BAND_ROWS`, so that every band but the last
#: is a whole number of blocks.
SAVE_REDUCE = 4


# ----------------------------------------------------------------------
# Working a band of rows at a time
#
# The preview and the image that gets saved are the same expression,
# ``clip(2 * weight * stretch(cuts(data)), 0, 1)``, applied to every pixel
# of the full size frames. The functions below are what they are both made
# of, so that the two cannot drift apart, and they hand back only reduced
# or byte-sized arrays, so that the frames themselves are the only full
# size floating point arrays in memory.
# ----------------------------------------------------------------------


def _n_blocks(n_pixels, factor=REDUCE):
    """
    How many blocks an axis of `n_pixels` pixels reduces to.

    Parameters
    ----------
    n_pixels : int
        Length of the axis.
    factor : int, optional
        How many pixels on a side go into one block.

    Returns
    -------
    int
        The number of blocks, counting the short one at the end of an
        axis that does not divide evenly.
    """
    return -(-n_pixels // factor)


def iter_bands(n_rows, factor=REDUCE):
    """
    Split a frame into bands of rows, with the blocks each band fills.

    Everything that walks a frame walks it this way, and everything that
    reduces one as it goes writes into the rows given here, so the band
    and block arithmetic is written once rather than in each of them.

    Parameters
    ----------
    n_rows : int
        Number of rows in the frame.
    factor : int, optional
        How many pixels on a side go into one block of a reduced image.

    Yields
    ------
    start, stop : int
        First row of the band and the row after its last, in order and
        covering every row of the frame. The last band is shorter than
        the rest if the frame does not divide evenly.
    rows : slice
        The rows of the reduced image the band averages into, which a
        caller that reduces nothing ignores.
    """
    for start in range(0, n_rows, BAND_ROWS):
        stop = min(start + BAND_ROWS, n_rows)
        yield start, stop, slice(start // factor, _n_blocks(stop, factor))


def _block_counts(n_pixels, factor=REDUCE):
    """
    Number of real pixels in each block along one axis of a frame.

    Parameters
    ----------
    n_pixels : int
        Length of the axis.
    factor : int, optional
        How many pixels on a side go into one block.

    Returns
    -------
    `numpy.ndarray`
        One count per block, all `factor` except the last, which is
        short if the axis does not divide evenly.
    """
    counts = np.full(_n_blocks(n_pixels, factor), float(factor))
    counts[-1] = n_pixels - factor * (len(counts) - 1)
    return counts


def _whole_blocks(array, factor=REDUCE):
    """
    See an array as the blocks of `factor` by `factor` pixels it holds.

    Parameters
    ----------
    array : `numpy.ndarray`
        Any two dimensional array. Rows and columns past the last whole
        block are left out, so an axis whose length is not a multiple of
        `factor` loses its short block.
    factor : int, optional
        How many pixels on a side go into one block.

    Returns
    -------
    `numpy.ndarray`
        A ``(row blocks, factor, column blocks, factor)`` view of the
        array. Reshaping the whole blocks of an array whose columns do
        not divide evenly copies them instead, which costs a second
        array and leaves anything written into it nowhere.
    """
    row_blocks, col_blocks = (n // factor for n in array.shape)
    row_stride, col_stride = array.strides
    return np.lib.stride_tricks.as_strided(
        array,
        shape=(row_blocks, factor, col_blocks, factor),
        strides=(factor * row_stride, row_stride, factor * col_stride, col_stride),
    )


def block_mean_band(band, factor=REDUCE):
    """
    Average one band of rows over blocks of `factor` by `factor` pixels.

    Parameters
    ----------
    band : `numpy.ndarray`
        The rows to average.
    factor : int, optional
        How many pixels on a side go into one block. `REDUCE` for the
        preview and the viewers; the picture on the Save tab is made of
        the bytes of the saved image four at a time.

    Returns
    -------
    `numpy.ndarray`
        The mean of each block. A block that runs off the edge of the
        frame is the mean of the pixels it does have, so an axis whose
        length is not a multiple of `factor` still reduces to
        ``ceil(length / factor)`` values.
    """
    rows, cols = band.shape
    pad_rows, pad_cols = -rows % factor, -cols % factor
    if pad_rows or pad_cols:
        band = np.pad(band, ((0, pad_rows), (0, pad_cols)))
        counts = np.outer(_block_counts(rows, factor), _block_counts(cols, factor))
    else:
        counts = factor * factor
    sums = band.reshape(band.shape[0] // factor, factor,
                        band.shape[1] // factor, factor).sum(axis=(1, 3))
    return sums / counts


def _band_scatter(band):
    """
    Scatter of the pixels within each whole block of a band of rows.

    Parameters
    ----------
    band : `numpy.ndarray`
        The rows to measure. Blocks that run off the edge of the frame
        are left out, since a short block is a worse measure of the noise
        than a whole one and there are only ever a few of them.

    Returns
    -------
    `numpy.ndarray`
        The standard deviation, with one degree of freedom taken by the
        mean, of the pixels of each whole block. A block with a missing
        pixel comes out as NaN.
    """
    blocks = _whole_blocks(band)
    n_pixels = REDUCE * REDUCE
    # The sums are taken in double precision, or the scatter of a bright
    # sky is lost in the rounding of its mean. Only the sums are: asking
    # for the deviations themselves in double precision would make a
    # double precision copy of the band, which is four bands of memory
    # where the answer is a small array.
    sums = np.einsum('ijkl->ik', blocks, dtype=np.float64)
    squares = np.einsum('ijkl,ijkl->ik', blocks, blocks, dtype=np.float64)
    variance = (squares - sums * sums / n_pixels) / (n_pixels - 1)
    # A block whose pixels are all the same value can come out a hair
    # below zero, and the square root of that is a warning and a NaN.
    np.maximum(variance, 0.0, out=variance)
    return np.sqrt(variance, out=variance)


def block_mean_and_noise(image):
    """
    Average a whole image over blocks and measure its noise, in one pass.

    Both are wanted for every frame that is loaded and both are made of
    the same blocks of the same bands, so they are worked out together:
    the frame is walked once rather than twice.

    Parameters
    ----------
    image : `numpy.ndarray`
        The full size frame.

    Returns
    -------
    reduced : `numpy.ndarray`
        The image averaged over blocks, as float32: the reduced image
        the background is fitted to and the viewers on the first tab
        show. It is the same answer `astropy.nddata.block_reduce` gives
        for a frame that divides evenly, without making anything the
        size of the frame along the way.
    noise : float
        The median, over the whole blocks of the frame, of the standard
        deviation of the pixels within a block. The median keeps the
        stars out of it: inside a block with a star in it the scatter is
        the shape of the star rather than noise. Blocks with missing
        pixels are left out, and the answer is 0 if no block is left.
        Noise of this size, added to the reduced image, makes it respond
        to the cuts the way the frame does.
    """
    out = np.empty((_n_blocks(image.shape[0]), _n_blocks(image.shape[1])),
                   dtype=np.float32)
    # One scatter per whole block of the frame, kept in an array of its
    # own rather than gathered up band by band afterwards: the bands
    # between them hold every whole block there is, since a band is a
    # whole number of blocks deep.
    scatter = np.empty((image.shape[0] // REDUCE, image.shape[1] // REDUCE),
                       dtype=np.float32)
    for start, stop, rows in iter_bands(image.shape[0]):
        band = image[start:stop]
        out[rows] = block_mean_band(band)
        # The whole blocks of the band, which are the rows it fills apart
        # from a short last one, since a short block has no scatter.
        whole = slice(rows.start, rows.start + (stop - start) // REDUCE)
        scatter[whole] = _band_scatter(band)
    scatter = scatter[np.isfinite(scatter)]
    if not scatter.size:
        return out, 0.0
    # The finite scatters are an array of this function's own, so the
    # median can shuffle it rather than take a copy to shuffle.
    return out, float(np.median(scatter, overwrite_input=True))


def read_frame(path):
    """
    Read one frame from a FITS file, a band of rows at a time.

    The file holds big-endian floats, so reading it in one go would put
    a whole frame in the file's own type in memory beside the array that
    keeps it. A band at a time costs a band.

    Parameters
    ----------
    path : str
        A FITS file. Its image is taken from the first HDU that has
        data, which is where ``CCDData.read`` looks for it too, so a file
        with an empty primary HDU and the image in an extension, as a
        compressed file has, can be read.

    Returns
    -------
    frame : `numpy.ndarray`
        The image, as float32 in this machine's byte order.
    header : `astropy.io.fits.Header`
        The header of the HDU the image came from.
    """
    with fits.open(path, memmap=True) as hdu_list:
        hdu = _image_hdu(hdu_list)
        frame = np.empty(hdu.shape, dtype=np.float32)
        for start, stop, _ in iter_bands(hdu.shape[0]):
            frame[start:stop] = hdu.section[start:stop]
        return frame, hdu.header


def blank_missing_pixels(frames):
    """
    Blank, in every frame, each pixel that is missing from any one of them.

    Reprojection leaves pixels with no data around the edges of a frame,
    and not the same pixels in each filter. A pixel that has data in one
    colour but not the others would be a coloured fringe along the edge
    of the saved image. Blanking them everywhere makes them black
    instead, which is what `matplotlib.image.imsave` used to do to the
    image on its way into the file.

    Parameters
    ----------
    frames : list of `numpy.ndarray`
        Frames of the same shape. They are modified in place.
    """
    for start, stop, _ in iter_bands(frames[0].shape[0]):
        missing = np.isnan(frames[0][start:stop])
        for frame in frames[1:]:
            missing |= np.isnan(frame[start:stop])
        for frame in frames:
            frame[start:stop][missing] = np.nan


def subtract_background_band(band, background_sm, start=0):
    """
    Take a background fitted to the reduced image off a band of a frame.

    Each value of the background describes one block of `REDUCE` by
    `REDUCE` pixels of the frame, so scaling it back up is a plain
    repeat. Repeating it makes a second array the size of the band, and
    the band a caller hands over is a view of the frame it keeps, so the
    answer has to be an array of its own: that is two bands where one
    will do. The same arithmetic falls out of the band seen in blocks
    with the reduced background broadcast against it, which is one array
    the size of the band and nothing else.

    Parameters
    ----------
    band : `numpy.ndarray`
        Rows of one frame. It is left as it is.
    background_sm : `numpy.ndarray`
        Background fitted to the reduced image of the whole frame.
    start : int, optional
        First row of the band, in full size rows. A multiple of `REDUCE`,
        since the bands fall on the boundaries of the blocks.

    Returns
    -------
    `numpy.ndarray`
        The band with its background taken off, as float32.
    """
    rows, cols = band.shape
    background = background_sm[start // REDUCE:_n_blocks(start + rows)]
    out = np.empty((rows, cols), dtype=np.float32)
    blocks = _whole_blocks(band)
    n_row_blocks, n_col_blocks = blocks.shape[0], blocks.shape[2]
    np.subtract(
        blocks,
        background[:n_row_blocks, np.newaxis, :n_col_blocks, np.newaxis],
        out=_whole_blocks(out),
    )
    # The pixels of a block that runs off the bottom or the right-hand
    # edge of a frame that does not divide evenly. Their background is
    # the last value of the row or the column of blocks they fall in,
    # stretched out one value per pixel, which is a row or a column of
    # them rather than a band of them.
    whole_rows, whole_cols = n_row_blocks * REDUCE, n_col_blocks * REDUCE
    if whole_cols < cols:
        edge = np.repeat(background[:n_row_blocks, -1], REDUCE)
        np.subtract(band[:whole_rows, whole_cols:], edge[:, np.newaxis],
                    out=out[:whole_rows, whole_cols:])
    if whole_rows < rows:
        edge = np.repeat(background[-1], REDUCE)[:cols]
        np.subtract(band[whole_rows:], edge, out=out[whole_rows:])
    return out


def scaled_band(band, interval, stretch, weight, background=None, start=0):
    """
    Turn one band of one colour into the values that get displayed.

    This is ``clip(2 * weight * stretch(cuts(band)), 0, 1)``: the cuts
    the user set with the level slider, the stretch chosen for all three
    colours, and the weight of this colour in the mix. Both the preview
    and the image that gets saved are made of this, so what the preview
    shows is what ends up in the file.

    Parameters
    ----------
    band : `numpy.ndarray`
        Rows of one frame.
    interval : `astropy.visualization.BaseInterval`
        The black and white points for this colour.
    stretch : `astropy.visualization.BaseStretch`
        The stretch applied between them.
    weight : float
        Where this colour's slider in the mixer sits, 0 to 1.
    background : `numpy.ndarray`, optional
        Background fitted to the reduced image of the whole frame, to
        subtract from the band first.
    start : int, optional
        First row of the band in the frame, so that the right rows of the
        background come off it. Zero, the default, is right for a band
        that is the whole frame.

    Returns
    -------
    `numpy.ndarray`
        Values from 0 to 1, one per pixel of the band. Pixels with no
        data come out as 0, which is black.
    """
    if background is not None:
        # Taking the background off makes an array of this band alone, and
        # it is ours, so the cuts can go back into it.
        band = subtract_background_band(band, background, start)
        values = interval(band, out=band)
    else:
        # Here band is a view of the caller's frame, which has to be left
        # as it is, so the cuts go into a floating point copy of the band.
        values = interval(band)
    values = stretch(values, out=values, clip=False)
    values *= 2 * weight
    # The cuts have already clipped to 0..1, so what can be out of range by
    # now is a value the doubling took past 1, and a NaN where there is no
    # data. Of a NaN and a number fmax gives back the number, which blacks
    # out those pixels in place, with none of the masks the size of the
    # band that nan_to_num makes.
    np.fmax(values, 0, out=values)
    np.minimum(values, 1, out=values)
    return values


def preview_plane(image, interval, stretch, weight, background=None):
    """
    Make one colour plane of the preview from a full size frame.

    The preview is the saved image seen `REDUCE` times smaller, so the
    cuts, the stretch, the weight and the clip are applied to every pixel
    of the frame and the result is then averaged over blocks. Averaging
    first, as the notebook used to, lifts the sky, because clipping at
    black and a log or sqrt stretch are not linear and the noise of a
    single pixel is several times the noise of a block.

    Parameters
    ----------
    image : `numpy.ndarray`
        The full size frame of one colour.
    interval, stretch, weight, background
        As for `scaled_band`, applied to every pixel of the frame.

    Returns
    -------
    `numpy.ndarray`
        The plane, `REDUCE` times smaller than the frame, as float32.
    """
    out = np.empty((_n_blocks(image.shape[0]), _n_blocks(image.shape[1])),
                   dtype=np.float32)
    for start, stop, rows in iter_bands(image.shape[0]):
        out[rows] = block_mean_band(
            scaled_band(image[start:stop], interval, stretch, weight,
                        background=background, start=start)
        )
    return out


def _band_bytes(image, start, stop, **scaling):
    """
    Turn rows of one full size frame into the bytes that get written.

    The image that gets saved and the picture the Save tab shows are both
    made of these, so that the two cannot be made of different
    arithmetic.

    Parameters
    ----------
    image : `numpy.ndarray`
        The full size frame of one colour.
    start, stop : int
        First row wanted and the row after the last.
    **scaling
        The ``interval``, ``stretch``, ``weight`` and ``background``
        arguments of `scaled_band`.

    Returns
    -------
    `numpy.ndarray`
        One byte per pixel of those rows. The values are truncated rather
        than rounded, which is what `matplotlib.image.imsave` did to the
        floating point image the notebook used to hand it, so that the
        file that is saved is unchanged.
    """
    scaled = scaled_band(image[start:stop], start=start, **scaling)
    # scaled is this band's own array, so the bytes can be made of it in
    # place rather than of a copy.
    scaled *= 255
    return scaled.astype(np.uint8)


def _rgb_band(frames, scalings, start, stop):
    """
    Turn rows of all three frames into the bytes of a band of the image.

    Parameters
    ----------
    frames : dict
        The full size frame of each colour in `COLORS`.
    scalings : dict
        How each colour is to be scaled, as for `rgb_uint8`.
    start, stop : int
        First row wanted and the row after the last.

    Returns
    -------
    `numpy.ndarray`
        Those rows of the image, ``(stop - start, columns, 3)`` of uint8.
    """
    out = np.empty((stop - start, frames[COLORS[0]].shape[1], 3), dtype=np.uint8)
    for plane, color in enumerate(COLORS):
        out[:, :, plane] = _band_bytes(frames[color], start, stop,
                                       **scalings[color])
    return out


def save_rgb_image(frames, scalings, path):
    """
    Write the finished image to a file, a band of rows at a time.

    Each band is pasted into the file's image as it is made, so neither
    the whole image nor a copy of it in Pillow is ever in memory: at 4096
    by 4096 that is a 48 MB array and a 64 MB buffer beside the three
    frames. The bytes written are the same either way.

    Parameters
    ----------
    frames : dict
        The full size frame of each colour in `COLORS`.
    scalings : dict
        How each colour is to be scaled, as for `rgb_uint8`.
    path : str
        Where to write the image. Pillow takes the format from the
        extension, as it does for an image built in one piece.
    """
    shape = frames[COLORS[0]].shape
    image = Image.new('RGB', (shape[1], shape[0]))
    for start, stop, _ in iter_bands(shape[0]):
        image.paste(Image.fromarray(_rgb_band(frames, scalings, start, stop)),
                    (0, start))
    image.save(path)


def rgb_uint8(frames, scalings):
    """
    Make the image that gets saved, one byte per colour per pixel.

    This is what the file holds, as an array. Saving does not go this
    way, since the array is as big as the file; `save_rgb_image` writes
    the same bytes without ever holding them all.

    Parameters
    ----------
    frames : dict
        The full size frame of each colour in `COLORS`.
    scalings : dict
        How each colour is to be scaled: for each colour a dict of the
        ``interval``, ``stretch``, ``weight`` and, optionally,
        ``background`` arguments of `preview_plane`.

    Returns
    -------
    `numpy.ndarray`
        The image, ``(rows, columns, 3)`` of uint8.
    """
    shape = frames[COLORS[0]].shape
    out = np.empty(shape + (3,), dtype=np.uint8)
    for start, stop, _ in iter_bands(shape[0]):
        out[start:stop] = _rgb_band(frames, scalings, start, stop)
    return out


def reduced_rgb_uint8(frames, scalings):
    """
    Make the image that gets saved, `SAVE_REDUCE` times smaller.

    This is the picture the Save tab shows. Each band is scaled to the
    bytes that would be written and averaged over blocks as it is made,
    so the picture is the saved image seen smaller, and the saved image
    itself is never in memory: at 4096 by 4096 that is a 50 MB array, and
    about as much again in Pillow, which has no way of encoding an RGB
    array without copying it first.

    Parameters
    ----------
    frames : dict
        The full size frame of each colour in `COLORS`.
    scalings : dict
        How each colour is to be scaled, as for `rgb_uint8`.

    Returns
    -------
    `numpy.ndarray`
        The picture, ``(rows, columns, 3)`` of uint8, each pixel the
        rounded mean of the block of the saved image it stands for.
    """
    shape = frames[COLORS[0]].shape
    out = np.empty((_n_blocks(shape[0], SAVE_REDUCE),
                    _n_blocks(shape[1], SAVE_REDUCE), 3), dtype=np.uint8)
    for start, stop, rows in iter_bands(shape[0], factor=SAVE_REDUCE):
        for plane, color in enumerate(COLORS):
            band = _band_bytes(frames[color], start, stop, **scalings[color])
            out[rows, :, plane] = np.round(block_mean_band(band, SAVE_REDUCE))
    return out


def png_bytes(rgb):
    """
    Encode a picture as a PNG.

    Parameters
    ----------
    rgb : `numpy.ndarray`
        The picture, ``(rows, columns, 3)`` of uint8.

    Returns
    -------
    bytes
        The PNG, ready for an `ipywidgets.Image`.
    """
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='png')
    return buffer.getvalue()


class ColorImageMaker:
    """
    Interactive widget for composing, adjusting, and saving RGB colour images
    from separate red, green, and blue FITS frames.

    Parameters
    ----------
    image_directory : str
        Path to the directory containing the three FITS files:
        ``combined_light_filter_rp.fit``, ``combined_light_filter_V.fit``,
        ``combined_light_filter_B.fit``.
    """

    _colors = COLORS
    _stretches = {
        'linear': LinearStretch(),
        'log': LogStretch(),
        'sqrt': SqrtStretch(),
    }

    def __init__(self, image_directory):
        self._image_directory = image_directory
        self.object_name = ''

        # Data storage. The three frames are the only arrays the size of
        # an image; everything else is reduced or a band of rows.
        self.data = {}
        self.data_sm_raw = {}
        self.data_sm = {}
        self.bkgd_sm = {}
        # Noise the size of the frame's own, which the viewers' reduced
        # images are shown with; averaging took the real noise out.
        self.noise_sm = {}
        # One plane of the preview per colour, each of them a pass over a
        # whole frame, so they are kept until something changes them.
        self.preview_planes = {}
        # Whether what the preview shows is out of date. It is drawn only
        # while its tab is on screen, and brought up to date when the tab
        # is opened, so that a slider on the first tab does not pay for a
        # pass over a frame to draw something nobody is looking at.
        self._preview_stale = True
        # The same for the picture on the Save tab, which is another pass
        # over all three frames.
        self._save_stale = True

        self._build_widgets()
        self._load_data()

    @property
    def image_directory(self):
        return self._image_directory

    @image_directory.setter
    def image_directory(self, value):
        self._image_directory = value
        # Reload data if widgets have already been built
        if hasattr(self, 'image_widgets'):
            self.bkgd_sm = {}
            self._load_data()
            # Loading draws nothing, which is right for a new widget, but
            # here the old object may already be on screen.
            self._settings_changed()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self):
        self.image_widgets = {c: ImageWidget() for c in self._colors}

        self.level_sliders = {c: self._make_level_slider() for c in self._colors}
        self.stretch_chooser = ipw.Dropdown(
            options=list(self._stretches), description='Stretch'
        )
        self.subtract_bkgd_checkbox = ipw.Checkbox(
            value=False,
            description='Subtract background',
            style={'description_width': 'initial'},
        )

        self.mix_sliders = {
            c: self._make_rgb_slider(c.capitalize()) for c in self._colors
        }

        self._build_bw_tab()
        self._build_color_tab()
        save_widget, self._refresh_save = self._build_save_tab()

        self.widget = ipw.Tab()
        self.widget.children = [self.vb, self.rgb_mixer, save_widget]
        self.widget.titles = ['1. Adjust B/W', '2. Adjust colors', '3. Save']
        self.widget.observe(self._on_tab_change, names='selected_index')

    def _make_level_slider(self):
        return ipw.FloatRangeSlider(
            min=0, max=100_000 / 64, step=100 / 64,
            description='Set black and white',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout={'width': '100%'},
        )

    def _make_rgb_slider(self, label):
        return ipw.FloatSlider(
            value=0.5, min=0, max=1, step=0.01,
            description=label,
            style={'description_width': 'initial'},
            continuous_update=False,
            layout={'width': '100%'},
        )

    def _build_bw_tab(self):
        """Build tab 1: per-channel B/W sliders and stretch/background controls."""
        for color in self._colors:
            self.level_sliders[color].observe(
                self._make_level_observer(color), names='value'
            )
        self.stretch_chooser.observe(self._stretch_observer, names='value')

        tab_set = ipw.Tab()
        tab_set.children = [
            ipw.VBox([self.level_sliders[c], self.image_widgets[c]])
            for c in self._colors
        ]
        tab_set.titles = self._colors

        self.vb = ipw.VBox(
            children=[
                ipw.HBox([self.subtract_bkgd_checkbox, self.stretch_chooser]),
                tab_set,
            ],
            layout={'width': '100%'},
        )

    def _build_color_tab(self):
        """Build tab 2: RGB mix sliders and live preview."""
        self.preview_output = ipw.Output()
        self.rgb_mixer = ipw.VBox(
            [*self.mix_sliders.values(), self.preview_output],
            layout={'width': '90%'},
        )

        # A mixer slider changes one colour of the preview; the sliders and
        # the dropdown on the first tab change one or all three, and their
        # observers redraw the preview themselves.
        for color, slider in self.mix_sliders.items():
            slider.observe(
                lambda change, color=color: self._settings_changed(color),
                names='value',
            )
        self.subtract_bkgd_checkbox.observe(self._on_subtract_change, names='value')

    def _build_save_tab(self):
        """Build tab 3: a look at the finished image and the save controls."""
        filename_input = ipw.Text(
            description='Add to filename:',
            value='',
            placeholder='e.g. your name',
            style={'description_width': 'initial'},
            layout={'width': '400px'},
        )
        status_html = ipw.HTML('')
        # A PNG of the finished image, made smaller as it is made, so that
        # neither the kernel nor the browser is asked to hold sixteen
        # million pixels.
        reduced_display = ipw.Image(format='png', layout={'width': '100%'})
        save_button = ipw.Button(description='Save image', button_style='success')
        save_status_label = ipw.Label('')

        def refresh():
            status_html.value = '<p style="padding:10px 0">Making the picture…</p>'
            reduced_display.value = png_bytes(self._reduced_rgb())
            status_html.value = ''
            self._save_stale = False

        def _reset_save_button():
            save_button.description = 'Save image'
            save_button.button_style = 'success'
            save_status_label.value = ''

        def on_save(b):
            suffix = filename_input.value
            filename = f'{self.object_name}-{suffix}-color.png'

            # If file exists and we haven't yet confirmed, ask for confirmation
            if os.path.exists(filename) and save_button.button_style != 'danger':
                save_button.description = 'Overwrite?'
                save_button.button_style = 'danger'
                save_status_label.value = f'{filename} already exists. Click again to overwrite.'
                return

            # The file is written a band of rows at a time, so saving
            # costs a band rather than the 50 MB of the whole image and
            # as much again in Pillow, beside the three frames.
            save_rgb_image(
                self.data, {c: self._scaling(c) for c in self._colors}, filename
            )
            save_status_label.value = f'Saved: {filename}'
            _reset_save_button()

        # Reset confirmation state when the filename changes
        filename_input.observe(lambda change: _reset_save_button(), names='value')
        save_button.on_click(on_save)

        widget = ipw.VBox([
            filename_input,
            ipw.HBox([save_button, save_status_label]),
            status_html,
            reduced_display,
        ])
        return widget, refresh

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """
        Read the three frames and set up everything made from them.

        The frames are read, the pixels missing from any one of them are
        blanked in all three, and each is averaged over blocks for the
        viewer on the first tab. Nothing at full size is worked out here:
        the preview is made the first time it is drawn, and the image
        that gets saved when the Save tab is opened.

        The images the viewers are given have noise of their own added,
        for the reason the comment where it is added gives.
        """
        # Let go of the frames of any images loaded before. Each new frame
        # would otherwise be read while all three old ones were still
        # held, which is four frames at once where three are needed.
        self.data.clear()
        self.data_sm_raw.clear()
        for color, filter_name in COLOR_FILTERS.items():
            path = os.path.join(
                self.image_directory, f'combined_light_filter_{filter_name}.fit'
            )
            self.data[color], header = read_frame(path)
            if color == OBJECT_NAME_COLOR:
                self.object_name = header['OBJECT']

        blank_missing_pixels([self.data[c] for c in self._colors])

        for seed, color in enumerate(self._colors):
            self.data_sm_raw[color], noise = block_mean_and_noise(self.data[color])
            # The made-up noise stays, on purpose. The block mean alone
            # understates the noise of the frame, so the same cuts leave
            # a viewer much darker than the preview and the saved file,
            # and making the viewers' images the way the preview is made,
            # by scaling every pixel and averaging afterwards, is too slow
            # to do while images are loading.
            #
            # Seeded, so that loading the same images again shows the
            # same thing, and differently for each colour.
            rng = np.random.default_rng(seed)
            self.noise_sm[color] = noise * rng.standard_normal(
                self.data_sm_raw[color].shape, dtype=np.float32
            )

        if self.subtract_bkgd_checkbox.value:
            self._compute_backgrounds()
        self._apply_background(self.subtract_bkgd_checkbox.value)

        # Give the viewers the cuts the level sliders are set to. The
        # preview planes are made the first time the preview is drawn.
        for color in self._colors:
            self.image_widgets[color].set_cuts(
                ManualInterval(*self.level_sliders[color].value)
            )
        self.preview_planes.clear()

    def _compute_backgrounds(self):
        """
        Fit a background to each reduced image (done once, when first asked).

        The fit is made on the 8x reduced image only. Its boxes are 64
        reduced pixels across, so it describes the same grid of sky
        patches a fit to the frame itself would, and one fit now serves
        both the preview and the image that gets saved, which cannot
        therefore disagree about the sky.
        """
        for color, sm_image in self.data_sm_raw.items():
            bkgd = Background2D(sm_image, (64, 64), filter_size=(3, 3), bkg_estimator=MeanBackground())
            self.bkgd_sm[color] = bkgd.background.astype(np.float32)

    def _apply_background(self, subtract):
        """
        Make the reduced images the viewers show and load them.

        Each viewer is given the stretch the chooser is set to along
        with its image, so that a reload cannot leave it showing
        something other than what was chosen.

        Parameters
        ----------
        subtract : bool
            Whether to take the background off them. The frames
            themselves are left alone; the background comes off them a
            band at a time, wherever they are used.
        """
        for color in self._colors:
            # A new array, so that nothing a viewer does to the image it
            # is given can reach the one the background is fitted to,
            # which has no noise added to it.
            shown = self.data_sm_raw[color] + self.noise_sm[color]
            if subtract:
                shown -= self.bkgd_sm[color]
            self.data_sm[color] = shown
            self.image_widgets[color].load_image(shown)
            self.image_widgets[color].set_stretch(
                self._stretches[self.stretch_chooser.value]
            )

    # ------------------------------------------------------------------
    # Image scaling and rendering
    # ------------------------------------------------------------------

    def _scaling(self, color):
        """
        How one colour is to be scaled, as the controls are set now.

        The preview and the image that gets saved both ask here, so that
        they cannot read the controls differently.

        Parameters
        ----------
        color : str
            One of `COLORS`.

        Returns
        -------
        dict
            The ``interval``, ``stretch``, ``weight`` and ``background``
            arguments of `preview_plane`.
        """
        subtract = self.subtract_bkgd_checkbox.value
        return dict(
            interval=ManualInterval(*self.level_sliders[color].value),
            stretch=self._stretches[self.stretch_chooser.value],
            weight=self.mix_sliders[color].value,
            # The box can be ticked with nothing fitted, if the fit was
            # never made or a reload failed part way; nothing comes off then.
            background=self.bkgd_sm.get(color) if subtract else None,
        )

    def _reduced_rgb(self):
        """
        Build the picture the Save tab shows, band by band.

        Returns
        -------
        `numpy.ndarray`
            The finished image four times smaller on a side,
            ``(rows, columns, 3)`` of uint8.
        """
        return reduced_rgb_uint8(
            self.data, {c: self._scaling(c) for c in self._colors}
        )

    def _preview_plane(self, color):
        """
        The preview plane of one colour, made if it is not already in hand.

        A plane costs one pass over the full size frame, so the three are
        kept: moving one slider changes one colour of the preview, or, for
        the stretch and the background, all three.

        Parameters
        ----------
        color : str
            One of `COLORS`.

        Returns
        -------
        `numpy.ndarray`
            The plane, `REDUCE` times smaller than the frame.
        """
        if color not in self.preview_planes:
            self.preview_planes[color] = preview_plane(
                self.data[color], **self._scaling(color)
            )
        return self.preview_planes[color]

    # ------------------------------------------------------------------
    # Observers / callbacks
    # ------------------------------------------------------------------

    def _make_level_observer(self, color):
        """
        Make the observer for one colour's black and white point slider.

        Parameters
        ----------
        color : str
            One of `COLORS`.

        Returns
        -------
        callable
            Observer that gives the viewer the new cuts and marks that
            colour of the preview as out of date.
        """
        def observer(change):
            self.image_widgets[color].set_cuts(ManualInterval(*change['new']))
            self._settings_changed(color)
        return observer

    def _settings_changed(self, *colors):
        """
        Note that the pictures are out of date, and redraw the one on screen.

        The cuts, the stretch, the background and the weight in the mix
        are all applied before an image is averaged, so a change to any
        of them means another pass over the frames for the preview and
        another for the picture on the Save tab. That pass is made now
        only for the tab that is on screen; the other is left until its
        tab is opened.

        Parameters
        ----------
        *colors : str
            The colours, from `COLORS`, whose preview planes are out of
            date. None at all still redraws, for when the planes have
            already been thrown away.
        """
        for color in colors:
            self.preview_planes.pop(color, None)
        self._preview_stale = True
        self._save_stale = True
        if self.widget.selected_index == 1:
            self._update_preview()
        elif self.widget.selected_index == 2:
            self._refresh_save()

    def _stretch_observer(self, change):
        """
        Give every viewer the chosen stretch; the whole preview is out of date.

        Parameters
        ----------
        change : dict
            The traitlets change, whose ``'new'`` is the name of a stretch.
        """
        for color in self._colors:
            self.image_widgets[color].set_stretch(self._stretches[change['new']])
        self._settings_changed(*self._colors)

    def _update_preview(self):
        """Draw the colour preview from the three planes."""
        comb = np.stack([self._preview_plane(c) for c in self._colors], axis=-1)
        maxes = [round(float(comb[:, :, i].max()), 3) for i in range(3)]
        max_img = max(maxes)
        r, g, b = (slider.value for slider in self.mix_sliders.values())
        with self.preview_output:
            self.preview_output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f'{max_img=:.3f} {r=:.2f} {g=:.2f} {b=:.2f}\n{maxes=}')
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.imshow(comb, vmin=0, vmax=1)
            plt.show()
            # The inline backend closes the figure when it shows it; any
            # other backend would keep one per slider move.
            plt.close(fig)
        self._preview_stale = False

    def _on_tab_change(self, change):
        """
        Bring the tab that has just been opened up to date.

        Each of the two pictures costs a pass over all three frames, so
        neither is made again for a tab that is opened with nothing
        changed since it was last looked at.

        Parameters
        ----------
        change : dict
            The traitlets change, whose ``'new'`` is the tab now shown.
        """
        if change['new'] == 1 and self._preview_stale:
            self._update_preview()
        elif change['new'] == 2 and self._save_stale:
            self._refresh_save()

    def _on_subtract_change(self, change):
        """
        Take the sky background off the images, or put it back.

        The backgrounds are fitted the first time the box is ticked.
        Every colour of the preview changes, and so do the images the
        viewers show, which are reloaded without losing the cuts and the
        stretch already chosen.

        Parameters
        ----------
        change : dict
            The traitlets change, whose ``'new'`` says whether the box is
            now ticked.
        """
        if not self.data_sm_raw:
            return
        if change['new'] and not self.bkgd_sm:
            self._compute_backgrounds()
        self._apply_background(change['new'])
        self._settings_changed(*self._colors)

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

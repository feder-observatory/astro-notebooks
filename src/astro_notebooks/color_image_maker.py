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

#: The colours of the image, in the order they are stored in it.
COLORS = ['red', 'green', 'blue']

#: How many rows of a full size frame are worked on at a time. Nothing the
#: size of a whole frame is ever made; a band of this many rows is.
BAND_ROWS = 256

#: How many pixels on a side are averaged into one pixel of the preview and
#: of the images the viewers on the first tab show.
REDUCE = 8


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


def iter_bands(n_rows, band_rows=BAND_ROWS):
    """
    Split a frame into bands of rows.

    Parameters
    ----------
    n_rows : int
        Number of rows in the frame.
    band_rows : int, optional
        Number of rows in a band. A multiple of `REDUCE`, so that the
        bands fall on the boundaries of the blocks the preview averages
        over.

    Yields
    ------
    start, stop : int
        First row of the band and the row after its last, in order and
        covering every row of the frame. The last band is shorter than
        the rest if the frame does not divide evenly.
    """
    for start in range(0, n_rows, band_rows):
        yield start, min(start + band_rows, n_rows)


def _block_counts(n_pixels):
    """
    Number of real pixels in each block along one axis of a frame.

    Parameters
    ----------
    n_pixels : int
        Length of the axis.

    Returns
    -------
    `numpy.ndarray`
        One count per block, all `REDUCE` except the last, which is
        short if the axis does not divide evenly.
    """
    counts = np.full(-(-n_pixels // REDUCE), float(REDUCE))
    counts[-1] = n_pixels - REDUCE * (len(counts) - 1)
    return counts


def block_mean_band(band):
    """
    Average one band of rows over blocks of `REDUCE` by `REDUCE` pixels.

    Parameters
    ----------
    band : `numpy.ndarray`
        The rows to average.

    Returns
    -------
    `numpy.ndarray`
        The mean of each block. A block that runs off the edge of the
        frame is the mean of the pixels it does have, so an axis whose
        length is not a multiple of `REDUCE` still reduces to
        ``ceil(length / REDUCE)`` values.
    """
    rows, cols = band.shape
    pad_rows, pad_cols = -rows % REDUCE, -cols % REDUCE
    if pad_rows or pad_cols:
        band = np.pad(band, ((0, pad_rows), (0, pad_cols)))
        counts = np.outer(_block_counts(rows), _block_counts(cols))
    else:
        counts = REDUCE * REDUCE
    sums = band.reshape(band.shape[0] // REDUCE, REDUCE,
                        band.shape[1] // REDUCE, REDUCE).sum(axis=(1, 3))
    return sums / counts


def block_mean(image):
    """
    Average a whole image over blocks, a band of rows at a time.

    This is the reduced image the background is fitted to, and what the
    viewers on the first tab show once the noise has been put back (see
    `pixel_noise`). It is the same answer `astropy.nddata.block_reduce`
    gives for a frame that divides evenly, without making anything the
    size of the frame along the way.

    Parameters
    ----------
    image : `numpy.ndarray`
        The image to average.

    Returns
    -------
    `numpy.ndarray`
        The reduced image, as float32.
    """
    out = np.empty((-(-image.shape[0] // REDUCE), -(-image.shape[1] // REDUCE)),
                   dtype=np.float32)
    for start, stop in iter_bands(image.shape[0]):
        out[start // REDUCE:-(-stop // REDUCE)] = block_mean_band(
            image[start:stop]
        )
    return out


def pixel_noise(image):
    """
    Typical scatter of the pixels of a frame about the mean of their block.

    Averaging over blocks takes most of the noise out of an image, so the
    cuts and the stretch do something different to the reduced image than
    they do to the frame: a sky that sits on the black point is black in
    the one and a grey glow in the other. Noise of this size, added to
    the reduced image, makes it respond to the cuts the way the frame
    does.

    Parameters
    ----------
    image : `numpy.ndarray`
        The full size frame.

    Returns
    -------
    float
        The median, over the whole blocks of the frame, of the standard
        deviation of the pixels within a block. The median keeps the
        stars out of it: inside a block with a star in it the scatter is
        the shape of the star rather than noise. Blocks with missing
        pixels are left out, and the answer is 0 if no block is left.
    """
    scatter = []
    for start, stop in iter_bands(image.shape[0]):
        band = image[start:stop]
        rows, cols = (n - n % REDUCE for n in band.shape)
        blocks = band[:rows, :cols].reshape(rows // REDUCE, REDUCE,
                                            cols // REDUCE, REDUCE)
        # In double precision, or the scatter of a bright sky is lost in
        # the rounding of its mean.
        scatter.append(blocks.std(axis=(1, 3), ddof=1, dtype=np.float64).ravel())
    scatter = np.concatenate(scatter)
    scatter = scatter[np.isfinite(scatter)]
    return float(np.median(scatter)) if scatter.size else 0.0


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
        for start, stop in iter_bands(hdu.shape[0]):
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
    for start, stop in iter_bands(frames[0].shape[0]):
        missing = np.isnan(frames[0][start:stop])
        for frame in frames[1:]:
            missing |= np.isnan(frame[start:stop])
        for frame in frames:
            frame[start:stop][missing] = np.nan


def background_band(background_sm, start, stop, n_cols):
    """
    Scale rows of a reduced background back up to the size of the frame.

    Each value of a background fitted to the reduced image describes one
    block of `REDUCE` by `REDUCE` pixels of the frame, so scaling it back
    up is a plain repeat.

    Parameters
    ----------
    background_sm : `numpy.ndarray`
        Background fitted to the reduced image.
    start, stop : int
        First row of the band and the row after its last, in full size
        rows.
    n_cols : int
        Number of columns in the frame.

    Returns
    -------
    `numpy.ndarray`
        The background of this band, the same shape as the band.
    """
    rows = background_sm[start // REDUCE:-(-stop // REDUCE)]
    full = np.repeat(np.repeat(rows, REDUCE, axis=0), REDUCE, axis=1)
    return full[:stop - start, :n_cols]


def scaled_band(band, interval, stretch, weight, background=None):
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
        Background to subtract from the band first.

    Returns
    -------
    `numpy.ndarray`
        Values from 0 to 1, one per pixel of the band. Pixels with no
        data come out as 0, which is black.
    """
    if background is not None:
        band = band - background
    values = interval(band)      # a floating point copy of this band alone
    values = stretch(values, out=values, clip=False)
    values *= 2 * weight
    np.clip(values, 0, 1, out=values)
    np.nan_to_num(values, copy=False)
    return values


def _scaled_rows(image, start, stop, interval, stretch, weight,
                 background=None):
    """
    Turn rows of one full size frame into the values that get displayed.

    Parameters
    ----------
    image : `numpy.ndarray`
        The full size frame of one colour.
    start, stop : int
        First row wanted and the row after the last.
    interval, stretch, weight
        As for `scaled_band`.
    background : `numpy.ndarray`, optional
        Background of the reduced image, to subtract from the frame.

    Returns
    -------
    `numpy.ndarray`
        Values from 0 to 1, one per pixel of those rows.
    """
    if background is not None:
        background = background_band(background, start, stop, image.shape[1])
    return scaled_band(image[start:stop], interval, stretch, weight,
                       background=background)


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
    interval : `astropy.visualization.BaseInterval`
        The black and white points for this colour.
    stretch : `astropy.visualization.BaseStretch`
        The stretch applied between them.
    weight : float
        Where this colour's slider in the mixer sits, 0 to 1.
    background : `numpy.ndarray`, optional
        Background of the reduced image, to subtract from the frame.

    Returns
    -------
    `numpy.ndarray`
        The plane, `REDUCE` times smaller than the frame, as float32.
    """
    out = np.empty((-(-image.shape[0] // REDUCE), -(-image.shape[1] // REDUCE)),
                   dtype=np.float32)
    for start, stop in iter_bands(image.shape[0]):
        scaled = _scaled_rows(image, start, stop, interval, stretch, weight,
                              background)
        out[start // REDUCE:-(-stop // REDUCE)] = block_mean_band(scaled)
    return out


def rgb_uint8(frames, scalings):
    """
    Make the image that gets saved, one byte per colour per pixel.

    The values are truncated rather than rounded, which is what
    `matplotlib.image.imsave` did to the floating point image the
    notebook used to hand it, so that the file that is saved is
    unchanged.

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
    for start, stop in iter_bands(shape[0]):
        for plane, color in enumerate(COLORS):
            scaled = _scaled_rows(frames[color], start, stop,
                                  **scalings[color])
            out[start:stop, :, plane] = (scaled * 255).astype(np.uint8)
    return out


def reduced_png_bytes(rgb, factor=4):
    """
    Encode a smaller copy of the saved image as a PNG.

    The Save tab shows this instead of drawing a 20 by 20 inch figure of
    the full size image.

    Parameters
    ----------
    rgb : `numpy.ndarray`
        The image, ``(rows, columns, 3)`` of uint8.
    factor : int, optional
        How many times smaller to make it.

    Returns
    -------
    bytes
        The PNG, ready for an `ipywidgets.Image`.
    """
    buffer = io.BytesIO()
    Image.fromarray(rgb).reduce(factor).save(buffer, format='png')
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
    _filters = ['rp', 'V', 'B']
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
            self._remake_preview()
            if self.widget.selected_index == 2:
                self._refresh_save()

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

        self.r_slider = self._make_rgb_slider('Red')
        self.g_slider = self._make_rgb_slider('Green')
        self.b_slider = self._make_rgb_slider('Blue')
        self.mix_sliders = dict(zip(
            self._colors, [self.r_slider, self.g_slider, self.b_slider]
        ))

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
            [self.r_slider, self.g_slider, self.b_slider, self.preview_output],
            layout={'width': '90%'},
        )

        # A mixer slider changes one colour of the preview; the sliders and
        # the dropdown on the first tab change one or all three, and their
        # observers redraw the preview themselves.
        for color, slider in self.mix_sliders.items():
            slider.observe(
                lambda change, color=color: self._remake_preview(color),
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
        # A PNG of the finished image, made smaller so that the browser is
        # not asked to show sixteen million pixels.
        full_res_display = ipw.Image(format='png', layout={'width': '100%'})
        save_button = ipw.Button(description='Save image', button_style='success')
        save_status_label = ipw.Label('')

        def refresh():
            status_html.value = '<p style="padding:10px 0">Generating full resolution image…</p>'
            full_res_display.value = reduced_png_bytes(self._full_res_rgb())
            status_html.value = ''

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

            # The full size image is made again here rather than kept from
            # when the tab was opened: it takes half a second, and keeping
            # it would hold 50 MB for as long as the widget lives.
            Image.fromarray(self._full_res_rgb()).save(filename)
            save_status_label.value = f'Saved: {filename}'
            _reset_save_button()

        # Reset confirmation state when the filename changes
        filename_input.observe(lambda change: _reset_save_button(), names='value')
        save_button.on_click(on_save)

        widget = ipw.VBox([
            filename_input,
            ipw.HBox([save_button, save_status_label]),
            status_html,
            full_res_display,
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

        The viewers show the averaged image with made-up noise as big as
        the frame's own added to it. Without it a viewer shows a sky much
        darker than the same cuts give in the preview and in the file,
        where they are applied to pixels that still have their noise.
        """
        # Let go of the frames of any images loaded before. Each new frame
        # would otherwise be read while all three old ones were still
        # held, which is four frames at once where three are needed.
        self.data.clear()
        self.data_sm_raw.clear()
        for color, filter_name in zip(self._colors, self._filters):
            path = os.path.join(
                self.image_directory, f'combined_light_filter_{filter_name}.fit'
            )
            self.data[color], header = read_frame(path)
            if color == 'blue':
                self.object_name = header['OBJECT']

        blank_missing_pixels([self.data[c] for c in self._colors])

        for seed, color in enumerate(self._colors):
            self.data_sm_raw[color] = block_mean(self.data[color])
            # Seeded, so that loading the same images again shows the
            # same thing, and differently for each colour.
            rng = np.random.default_rng(seed)
            self.noise_sm[color] = pixel_noise(self.data[color]) * rng.standard_normal(
                self.data_sm_raw[color].shape, dtype=np.float32
            )

        if self.subtract_bkgd_checkbox.value:
            self._compute_backgrounds()
        self._apply_background(self.subtract_bkgd_checkbox.value)

        for color in self._colors:
            self.image_widgets[color].load_image(self.data_sm[color])
            self.image_widgets[color].set_stretch(
                self._stretches[self.stretch_chooser.value]
            )

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
        Set the reduced images the viewers show on the first tab.

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

    def _full_res_rgb(self):
        """
        Build the finished image at full size, ready to be written out.

        Returns
        -------
        `numpy.ndarray`
            The image, ``(rows, columns, 3)`` of uint8.
        """
        return rgb_uint8(self.data, {c: self._scaling(c) for c in self._colors})

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
            self._remake_preview(color)
        return observer

    def _remake_preview(self, *colors):
        """
        Throw away some colours of the preview, and redraw it if it is seen.

        The cuts, the stretch, the background and the weight in the mix
        are all applied before the image is averaged, so a change to any
        of them means another pass over the frame. That pass is made now
        only if the preview's tab is the one on screen; otherwise it is
        left until the tab is opened.

        Parameters
        ----------
        *colors : str
            The colours, from `COLORS`, whose planes are out of date.
            None at all still redraws, for when the planes have already
            been thrown away.
        """
        for color in colors:
            self.preview_planes.pop(color, None)
        self._preview_stale = True
        if self.widget.selected_index == 1:
            self._update_preview()

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
        self._remake_preview(*self._colors)

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

        The preview is drawn if anything has changed since it was last
        drawn, and the finished image is built for the Save tab.

        Parameters
        ----------
        change : dict
            The traitlets change, whose ``'new'`` is the tab now shown.
        """
        if change['new'] == 1 and self._preview_stale:
            self._update_preview()
        elif change['new'] == 2:
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
        for color in self._colors:
            self.image_widgets[color].load_image(self.data_sm[color])
            self.image_widgets[color].set_stretch(
                self._stretches[self.stretch_chooser.value]
            )
        self._remake_preview(*self._colors)

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

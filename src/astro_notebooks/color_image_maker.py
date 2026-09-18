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


def _block_counts(n_pixels, factor):
    """
    Number of real pixels in each block along one axis of a frame.

    Parameters
    ----------
    n_pixels : int
        Length of the axis.
    factor : int
        Length of a block along the axis.

    Returns
    -------
    `numpy.ndarray`
        One count per block, all ``factor`` except the last, which is
        short if the axis does not divide evenly.
    """
    counts = np.full(-(-n_pixels // factor), float(factor))
    counts[-1] = n_pixels - factor * (len(counts) - 1)
    return counts


def block_mean_band(band, factor=REDUCE):
    """
    Average one band of rows over blocks of ``factor`` by ``factor`` pixels.

    Parameters
    ----------
    band : `numpy.ndarray`
        The rows to average.
    factor : int, optional
        Pixels on a side of a block.

    Returns
    -------
    `numpy.ndarray`
        The mean of each block. A block that runs off the edge of the
        frame is the mean of the pixels it does have, so an axis whose
        length is not a multiple of ``factor`` still reduces to
        ``ceil(length / factor)`` values.
    """
    rows, cols = band.shape
    pad_rows, pad_cols = -rows % factor, -cols % factor
    if pad_rows or pad_cols:
        band = np.pad(band, ((0, pad_rows), (0, pad_cols)))
        counts = np.outer(_block_counts(rows, factor),
                          _block_counts(cols, factor))
    else:
        counts = factor * factor
    sums = band.reshape(band.shape[0] // factor, factor,
                        band.shape[1] // factor, factor).sum(axis=(1, 3))
    return sums / counts


def block_mean(image, factor=REDUCE):
    """
    Average a whole image over blocks, a band of rows at a time.

    This is the reduced image the viewers on the first tab are given. It
    is the same answer `astropy.nddata.block_reduce` gives for a frame
    that divides evenly, without making anything the size of the frame
    along the way.

    Parameters
    ----------
    image : `numpy.ndarray`
        The image to average.
    factor : int, optional
        Pixels on a side of a block.

    Returns
    -------
    `numpy.ndarray`
        The reduced image, as float32.
    """
    out = np.empty((-(-image.shape[0] // factor), -(-image.shape[1] // factor)),
                   dtype=np.float32)
    for start, stop in iter_bands(image.shape[0]):
        out[start // factor:-(-stop // factor)] = block_mean_band(
            image[start:stop], factor
        )
    return out


def read_frame(path):
    """
    Read one frame from a FITS file, a band of rows at a time.

    The file holds big-endian floats, so reading it in one go would put
    a whole frame in the file's own type in memory beside the array that
    keeps it. A band at a time costs a band.

    Parameters
    ----------
    path : str
        A FITS file with its image in the primary HDU.

    Returns
    -------
    frame : `numpy.ndarray`
        The image, as float32 in this machine's byte order.
    header : `astropy.io.fits.Header`
        The header of the primary HDU.
    """
    with fits.open(path, memmap=True) as hdu_list:
        hdu = hdu_list[0]
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


def background_band(background_sm, start, stop, n_cols, factor=REDUCE):
    """
    Scale rows of a reduced background back up to the size of the frame.

    Each value of a background fitted to the reduced image describes one
    block of ``factor`` by ``factor`` pixels of the frame, so scaling it
    back up is a plain repeat.

    Parameters
    ----------
    background_sm : `numpy.ndarray`
        Background fitted to the reduced image.
    start, stop : int
        First row of the band and the row after its last, in full size
        rows.
    n_cols : int
        Number of columns in the frame.
    factor : int, optional
        Pixels on a side of a block.

    Returns
    -------
    `numpy.ndarray`
        The background of this band, the same shape as the band.
    """
    rows = background_sm[start // factor:-(-stop // factor)]
    full = np.repeat(np.repeat(rows, factor, axis=0), factor, axis=1)
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


def preview_plane(image, interval, stretch, weight, background=None,
                  factor=REDUCE):
    """
    Make one colour plane of the preview from a full size frame.

    The preview is the saved image seen ``factor`` times smaller, so the
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
    factor : int, optional
        Pixels on a side of a block.

    Returns
    -------
    `numpy.ndarray`
        The plane, ``factor`` times smaller than the frame, as float32.
    """
    out = np.empty((-(-image.shape[0] // factor), -(-image.shape[1] // factor)),
                   dtype=np.float32)
    for start, stop in iter_bands(image.shape[0]):
        rows = (None if background is None
                else background_band(background, start, stop, image.shape[1],
                                     factor))
        scaled = scaled_band(image[start:stop], interval, stretch, weight,
                             background=rows)
        out[start // factor:-(-stop // factor)] = block_mean_band(scaled, factor)
    return out


def rgb_uint8(frames, intervals, stretch, weights, backgrounds=None):
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
    intervals : dict
        The black and white points of each colour.
    stretch : `astropy.visualization.BaseStretch`
        The stretch applied to all three colours.
    weights : dict
        Where each colour's slider in the mixer sits, 0 to 1.
    backgrounds : dict, optional
        Background of the reduced image of each colour, to subtract.

    Returns
    -------
    `numpy.ndarray`
        The image, ``(rows, columns, 3)`` of uint8.
    """
    shape = frames[COLORS[0]].shape
    out = np.empty(shape + (3,), dtype=np.uint8)
    for start, stop in iter_bands(shape[0]):
        for plane, color in enumerate(COLORS):
            rows = (None if backgrounds is None
                    else background_band(backgrounds[color], start, stop,
                                         shape[1]))
            scaled = scaled_band(frames[color][start:stop], intervals[color],
                                 stretch, weights[color], background=rows)
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
    Image.fromarray(rgb, mode='RGB').reduce(factor).save(buffer, format='png')
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
        # One plane of the preview per colour, each of them a pass over a
        # whole frame, so they are kept until something changes them.
        self.preview_planes = {}

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
        for color, slider in zip(
            self._colors, [self.r_slider, self.g_slider, self.b_slider]
        ):
            slider.observe(self._make_mix_observer(color), names='value')
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

        # Cache the last full-resolution composite so save doesn't recompute it
        cached = {'full_res_rgb': None}

        def refresh():
            status_html.value = '<p style="padding:10px 0">Generating full resolution image…</p>'
            cached['full_res_rgb'] = self._full_res_rgb()
            full_res_display.value = reduced_png_bytes(cached['full_res_rgb'])
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

            Image.fromarray(cached['full_res_rgb'], mode='RGB').save(filename)
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
        """Load FITS files, compute backgrounds, populate data dicts, init observers."""
        for color, filter_name in zip(self._colors, self._filters):
            path = os.path.join(
                self.image_directory, f'combined_light_filter_{filter_name}.fit'
            )
            self.data[color], header = read_frame(path)
            if color == 'blue':
                self.object_name = header['OBJECT']

        blank_missing_pixels([self.data[c] for c in self._colors])

        for color in self._colors:
            self.data_sm_raw[color] = block_mean(self.data[color])

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
        for color, interval in self._intervals().items():
            self.image_widgets[color].set_cuts(interval)
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
            if subtract:
                self.data_sm[color] = self.data_sm_raw[color] - self.bkgd_sm[color]
            else:
                self.data_sm[color] = self.data_sm_raw[color]

    # ------------------------------------------------------------------
    # Image scaling and rendering
    # ------------------------------------------------------------------

    def _intervals(self):
        """
        The black and white points the level sliders are set to.

        Returns
        -------
        dict
            A `~astropy.visualization.ManualInterval` for each colour.
        """
        return {
            color: ManualInterval(*self.level_sliders[color].value)
            for color in self._colors
        }

    def _stretch(self):
        """
        The stretch the dropdown is set to, used for all three colours.

        Returns
        -------
        `astropy.visualization.BaseStretch`
            The chosen stretch.
        """
        return self._stretches[self.stretch_chooser.value]

    def _weights(self):
        """
        How much of each colour the mixer sliders ask for.

        Returns
        -------
        dict
            A number from 0 to 1 for each colour.
        """
        return dict(zip(
            self._colors,
            [self.r_slider.value, self.g_slider.value, self.b_slider.value],
        ))

    def _backgrounds(self):
        """
        The background to take off each frame, if the box is ticked.

        Returns
        -------
        dict or None
            The reduced background of each colour, or None when the
            background is to be left in.
        """
        return self.bkgd_sm if self.subtract_bkgd_checkbox.value else None

    def _full_res_rgb(self):
        """
        Build the finished image at full size, ready to be written out.

        Returns
        -------
        `numpy.ndarray`
            The image, ``(rows, columns, 3)`` of uint8.
        """
        return rgb_uint8(self.data, self._intervals(), self._stretch(),
                         self._weights(), backgrounds=self._backgrounds())

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
            backgrounds = self._backgrounds()
            self.preview_planes[color] = preview_plane(
                self.data[color], self._intervals()[color], self._stretch(),
                self._weights()[color],
                background=None if backgrounds is None else backgrounds[color],
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
            Observer that gives the viewer the new cuts and remakes that
            colour of the preview.
        """
        def observer(change):
            minval, maxval = change['new']
            self.image_widgets[color].set_cuts(ManualInterval(minval, maxval))
            self.preview_planes.pop(color, None)
            self._update_preview()
        return observer

    def _make_mix_observer(self, color):
        """
        Make the observer for one colour's slider in the mixer.

        Parameters
        ----------
        color : str
            One of `COLORS`.

        Returns
        -------
        callable
            Observer that remakes that colour of the preview. The weight
            is applied before the image is averaged, so the plane really
            does have to be made again from the frame.
        """
        def observer(change):
            self.preview_planes.pop(color, None)
            self._update_preview()
        return observer

    def _stretch_observer(self, change):
        """
        Give every viewer the chosen stretch and remake the whole preview.

        Parameters
        ----------
        change : dict
            The traitlets change, whose ``'new'`` is the name of a stretch.
        """
        for color in self._colors:
            self.image_widgets[color].set_stretch(self._stretches[change['new']])
        self.preview_planes.clear()
        self._update_preview()

    def _update_preview(self, change=None):
        """
        Draw the colour preview from the three planes.

        Parameters
        ----------
        change : dict, optional
            Ignored; lets this be used as an observer.
        """
        comb = np.stack([self._preview_plane(c) for c in self._colors], axis=-1)
        maxes = [round(float(comb[:, :, i].max()), 3) for i in range(3)]
        max_img = max(maxes)
        r, g, b = (self._weights()[c] for c in self._colors)
        with self.preview_output:
            self.preview_output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f'{max_img=:.3f} {r=:.2f} {g=:.2f} {b=:.2f}\n{maxes=}')
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.imshow(comb, vmin=0, vmax=1)
            plt.show()

    def _on_tab_change(self, change):
        if change['new'] == 2:
            self._refresh_save()

    def _on_subtract_change(self, change):
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
        self.preview_planes.clear()
        self._update_preview()

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

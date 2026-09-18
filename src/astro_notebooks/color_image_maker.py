"""Interactive RGB colour image composer for astronomical FITS data."""

import os

import ipywidgets as ipw
import matplotlib.image as mimg
import numpy as np
from IPython.display import display
from astropy.nddata import CCDData, block_reduce
from astropy.visualization import (
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)
from astrowidgets.bqplot import ImageWidget
from matplotlib import pyplot as plt
from photutils.background import Background2D, MedianBackground, MeanBackground


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

    _colors = ['red', 'green', 'blue']
    _stretches = {
        'linear': LinearStretch(),
        'log': LogStretch(),
        'sqrt': SqrtStretch(),
    }

    def __init__(self, image_directory):
        self._image_directory = image_directory
        self.object_name = ''

        # Data storage
        self.data_sm = {}
        self.data = {}
        self.sc_raw = {}
        self.sc_raw_f = {}
        self.data_sm_raw = {}
        self.data_raw_unmod = {}
        self.bkgd_sm = {}
        self.bkgd_f = {}

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
            self.bkgd_f = {}
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

        for slider in [self.r_slider, self.g_slider, self.b_slider]:
            slider.observe(self._update_preview, names='value')
        for color in self._colors:
            self.level_sliders[color].observe(self._update_preview, names='value')
        self.stretch_chooser.observe(self._update_preview, names='value')
        self.subtract_bkgd_checkbox.observe(self._on_subtract_change, names='value')

    def _build_save_tab(self):
        """Build tab 3: full-resolution preview and save controls."""
        filename_input = ipw.Text(
            description='Add to filename:',
            value='',
            placeholder='e.g. your name',
            style={'description_width': 'initial'},
            layout={'width': '400px'},
        )
        status_html = ipw.HTML('')
        full_res_output = ipw.Output()
        save_button = ipw.Button(description='Save image', button_style='success')
        save_status_label = ipw.Label('')

        # Cache the last full-resolution composite so save doesn't recompute it
        cached = {'full_res_rgb': None}

        def refresh():
            status_html.value = '<p style="padding:10px 0">Generating full resolution image…</p>'
            r, g, b = self.r_slider.value, self.g_slider.value, self.b_slider.value
            cached['full_res_rgb'], _ = self._rgb_scaling(self.sc_raw_f, r, g, b)
            with full_res_output:
                full_res_output.clear_output()
                self._full_res_color_rgb(r, g, b, cached['full_res_rgb'])
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

            mimg.imsave(filename, cached['full_res_rgb'])
            save_status_label.value = f'Saved: {filename}'
            _reset_save_button()

        # Reset confirmation state when the filename changes
        filename_input.observe(lambda change: _reset_save_button(), names='value')
        save_button.on_click(on_save)

        widget = ipw.VBox([
            filename_input,
            ipw.HBox([save_button, save_status_label]),
            status_html,
            full_res_output,
        ])
        return widget, refresh

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load FITS files, compute backgrounds, populate data dicts, init observers."""
        red = CCDData.read(os.path.join(self.image_directory, 'combined_light_filter_rp.fit'))
        greenish = CCDData.read(os.path.join(self.image_directory, 'combined_light_filter_V.fit'))
        blue = CCDData.read(os.path.join(self.image_directory, 'combined_light_filter_B.fit'))

        reduce_fac = 8
        red_sm = block_reduce(red.data, reduce_fac, func=np.mean)
        green_sm = block_reduce(greenish.data, reduce_fac, func=np.mean)
        blue_sm = block_reduce(blue.data, reduce_fac, func=np.mean)

        for sm_image, color in zip([red_sm, green_sm, blue_sm], self._colors):
            self.data_sm_raw[color] = sm_image.copy()

        for raw_array, color in zip([red.data, greenish.data, blue.data], self._colors):
            self.data_raw_unmod[color] = raw_array.copy()

        self._apply_background(False)

        for color in self._colors:
            self.image_widgets[color].load_image(self.data_sm[color])
            self.image_widgets[color].set_stretch(
                self._stretches[self.stretch_chooser.value]
            )

        # Initialise sc_raw / sc_raw_f using current slider cuts
        for color in self._colors:
            self._make_level_observer(color)(dict(new=self.level_sliders[color].value))

        self.object_name = blue.header['OBJECT']

    def _compute_backgrounds(self):
        """Compute and store background models for all channels (called lazily)."""
        for color, sm_image in self.data_sm_raw.items():
            bkgd = Background2D(sm_image, (64, 64), filter_size=(3, 3), bkg_estimator=MeanBackground())
            self.bkgd_sm[color] = bkgd.background
        for color, raw_array in self.data_raw_unmod.items():
            bkgd = Background2D(raw_array.copy(), (512, 512), filter_size=(3, 3), bkg_estimator=MeanBackground())
            self.bkgd_f[color] = bkgd.background

    def _apply_background(self, subtract):
        """Fill data_sm and data from raw arrays, optionally subtracting the background."""
        for color in self._colors:
            if subtract:
                self.data_sm[color] = self.data_sm_raw[color] - self.bkgd_sm[color]
                self.data[color] = self.data_raw_unmod[color] - self.bkgd_f[color]
            else:
                self.data_sm[color] = self.data_sm_raw[color].copy()
                self.data[color] = self.data_raw_unmod[color].copy()

    # ------------------------------------------------------------------
    # Image scaling and rendering
    # ------------------------------------------------------------------

    def _rgb_scaling(self, sc_data, r=0.5, g=0.5, b=0.5):
        red_sc = r * sc_data['red']
        green_sc = g * sc_data['green']
        blue_sc = b * sc_data['blue']
        comb = np.zeros(list(red_sc.shape) + [3])
        comb[:, :, 0] = red_sc
        comb[:, :, 1] = green_sc
        comb[:, :, 2] = blue_sc
        maxes = [np.nanmax(red_sc), np.nanmax(green_sc), np.nanmax(blue_sc)]
        comb = 2 * comb
        comb[comb > 1] = 1.0
        return comb, maxes

    def _quick_color_rgb(self, r=0.5, g=0.5, b=0.5, output=None):
        comb, maxes = self._rgb_scaling(self.sc_raw, r, g, b)
        max_img = np.nanmax(comb.flatten())
        if output is None:
            return
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f'{max_img=:.3f} {r=:.2f} {g=:.2f} {b=:.2f}\n{maxes=}')
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.imshow(comb, vmin=0, vmax=1)
            plt.show()

    def _full_res_color_rgb(self, r=0.5, g=0.5, b=0.5, comb=None):
        if comb is None:
            comb, _ = self._rgb_scaling(self.sc_raw_f, r, g, b)
        fig, ax = plt.subplots(figsize=(20, 20))
        max_img = np.nanmax(comb.flatten())
        min_img = np.nanmin(comb.flatten())
        maxes = [np.nanmax(comb[:, :, i]) for i in range(3)]
        ax.set_title(f'{max_img=:.3f} {min_img=:.3f} {r=:.2f} {g=:.2f} {b=:.2f}\n{maxes=}')
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.imshow(comb, vmin=0, vmax=1)
        display(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Observers / callbacks
    # ------------------------------------------------------------------

    def _get_scaled_image_data(self, viewer, data):
        return viewer.get_stretch()(viewer.get_cuts()(data))

    def _make_level_observer(self, color):
        def observer(change):
            minval, maxval = change['new']
            self.image_widgets[color].set_cuts(ManualInterval(minval, maxval))
            self.sc_raw[color] = self._get_scaled_image_data(
                self.image_widgets[color], self.data_sm[color]
            )
            self.sc_raw_f[color] = self._get_scaled_image_data(
                self.image_widgets[color], self.data[color]
            )
        return observer

    def _stretch_observer(self, change):
        for color in self._colors:
            self.image_widgets[color].set_stretch(self._stretches[change['new']])
            self.sc_raw[color] = self._get_scaled_image_data(
                self.image_widgets[color], self.data_sm[color]
            )
            self.sc_raw_f[color] = self._get_scaled_image_data(
                self.image_widgets[color], self.data[color]
            )

    def _update_preview(self, change):
        self._quick_color_rgb(
            self.r_slider.value, self.g_slider.value, self.b_slider.value,
            output=self.preview_output,
        )

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
            self._make_level_observer(color)(dict(new=self.level_sliders[color].value))
        self._update_preview(None)

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

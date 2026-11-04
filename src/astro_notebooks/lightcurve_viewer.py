import ipywidgets as ipw
import numpy as np
from astropy.table import Table
from bqplot import Axis, Figure, LinearScale, Lines, Scatter
from bqplot.colorschemes import CATEGORY10

# Marker names (in the source list) of the stars a student most likely
# wants to see first. VSX is deliberately not here: a field can have dozens
# of VSX stars, and starting with all of them plotted is a mess.
DEFAULT_MARKER_NAMES = ("TESS Targets",)

# Room beyond the data on each axis, as a fraction of the range of the data.
AXIS_PADDING = 0.05

# bqplot's own colour cycle, so each star gets a colour of its own.
COLORS = list(CATEGORY10)

X_AXIS_CHOICES = ("BJD", "Hours since first image", "Airmass")

# label -> (value column or None, error column, is a magnitude)
# The differential magnitude has no column of its own; it is computed from
# the relative flux.
QUANTITIES = {
    "Instrumental magnitude": ("mag_inst", "mag_error", True),
    "Relative flux": ("relative_flux", "relative_flux_error", False),
    "Differential magnitude": (None, "relative_flux_error", True),
    "Calibrated magnitude": ("mag_cal", "mag_cal_error", True),
}

# Column that has to be present for each quantity to be shown.
_REQUIRED_COLUMN = {
    label: (value_col if value_col is not None else "relative_flux")
    for label, (value_col, _, _) in QUANTITIES.items()
}


def _as_float_array(column):
    """
    Turn a table column into a plain float array, with NaN for masked values.

    Parameters
    ----------
    column : `astropy.table.Column`, `astropy.units.Quantity`, `astropy.time.Time` or array-like
        The column to convert.

    Returns
    -------
    numpy.ndarray
        Float values; masked entries become NaN.
    """
    if hasattr(column, "jd"):
        column = column.jd
    column = getattr(column, "value", column)
    return np.ma.filled(np.ma.asarray(column).astype(float), np.nan)


def available_quantities(table):
    """
    Work out which quantities can be shown for a table.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table.

    Returns
    -------
    available : list of str
        Labels of the quantities whose value column is in the table, in
        the order of `QUANTITIES`.
    missing : dict
        Label -> name of the missing column, for the quantities left out.
    """
    available = []
    missing = {}
    for label, column in _REQUIRED_COLUMN.items():
        if column in table.colnames:
            available.append(label)
        else:
            missing[label] = column
    return available, missing


def quantity_values(table, quantity):
    """
    Get the values, and their errors, of one of the quantities.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table.
    quantity : str
        One of the keys of `QUANTITIES`.

    Returns
    -------
    values : numpy.ndarray
        The values to plot.
    errors : numpy.ndarray or None
        One-sigma errors in the same units as ``values``, or None if the
        table has no error column for this quantity.
    """
    value_col, error_col, _ = QUANTITIES[quantity]
    if quantity == "Differential magnitude":
        flux = _as_float_array(table["relative_flux"])
        with np.errstate(divide="ignore", invalid="ignore"):
            values = -2.5 * np.log10(flux)
            if error_col in table.colnames:
                flux_error = _as_float_array(table[error_col])
                errors = 2.5 / np.log(10) * flux_error / flux
            else:
                errors = None
        return values, errors

    values = _as_float_array(table[value_col])
    errors = _as_float_array(table[error_col]) if error_col in table.colnames else None
    return values, errors


def x_axis_values(table, choice, reference_bjd=None):
    """
    Get the values to plot along the x axis.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table.
    choice : str
        One of `X_AXIS_CHOICES`.
    reference_bjd : float, optional
        BJD that counts as hour zero for "Hours since first image". The
        earliest BJD in ``table`` if not given.

    Returns
    -------
    values : numpy.ndarray
        The x values.
    label : str
        Label for the axis.
    """
    if choice == "BJD":
        return _as_float_array(table["bjd"]), "BJD"
    if choice == "Hours since first image":
        bjd = _as_float_array(table["bjd"])
        if reference_bjd is None:
            reference_bjd = np.nanmin(bjd)
        return (bjd - reference_bjd) * 24, "Hours since first image"
    if choice == "Airmass":
        return _as_float_array(table["airmass"]), "Airmass"
    raise ValueError(f"Unknown x axis choice: {choice!r}")


def _marker_names(source_list):
    """Map star_id -> marker name for a source list, empty if no list."""
    if source_list is None or "marker name" not in source_list.colnames:
        return {}
    return {
        int(star_id): str(name)
        for star_id, name in zip(source_list["star_id"], source_list["marker name"])
    }


def star_options(table, source_list=None):
    """
    List the stars in a table, labelled for a selection widget.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table (already limited to one passband if wanted).
    source_list : `astropy.table.Table`, optional
        Table with columns ``star_id`` and ``marker name``.

    Returns
    -------
    list of tuple
        ``(label, star_id)`` for each star in the table, sorted by id. The
        label is the id and the marker name, or only the id if there is no
        marker name for the star.
    """
    names = _marker_names(source_list)
    options = []
    star_ids = np.asarray(table["star_id"]).astype(int)
    for star_id in sorted({int(s) for s in star_ids}):
        if star_id in names:
            label = f"{star_id:3d} {names[star_id]}"
        else:
            label = f"{star_id:3d}"
        options.append((label, star_id))
    return options


def default_stars(options, source_list=None):
    """
    Pick the stars to show when nothing has been selected yet.

    Parameters
    ----------
    options : list of tuple
        Output of `star_options`.
    source_list : `astropy.table.Table`, optional
        Table with columns ``star_id`` and ``marker name``.

    Returns
    -------
    tuple
        Ids of the stars marked as one of `DEFAULT_MARKER_NAMES`, or the
        first star if there are none, or nothing if there are no stars.
    """
    names = _marker_names(source_list)
    chosen = tuple(
        star_id
        for _, star_id in options
        if names.get(star_id) in DEFAULT_MARKER_NAMES
    )
    if not chosen and options:
        chosen = (options[0][1],)
    return chosen


class LightCurveViewer(ipw.HBox):
    """
    Interactive light curves: the plot on the left, the controls on the right.

    Parameters
    ----------
    photometry : str, `pathlib.Path` or `astropy.table.Table`
        The photometry file written by stellarphot (read with
        `stellarphot.PhotometryData.read`), or a table already loaded.
    source_list : str, `pathlib.Path` or `astropy.table.Table`, optional
        The source list file, with columns ``star_id`` and ``marker name``,
        or a table already loaded. Used to label the stars.
    **kwargs
        Passed on to `ipywidgets.HBox`.

    Attributes
    ----------
    figure : `bqplot.Figure`
        The plot. It is never replaced, only its ``marks`` are.
    """

    def __init__(self, photometry, source_list=None, **kwargs):
        super().__init__(**kwargs)
        self.table = self._load_photometry(photometry)
        if source_list is None or isinstance(source_list, Table):
            self.source_list = source_list
        else:
            self.source_list = Table.read(source_list)

        # Hours since first image is measured from the first image in the
        # whole file so that the passbands line up.
        if "bjd" in self.table.colnames:
            self._reference_bjd = np.nanmin(_as_float_array(self.table["bjd"]))
        else:
            self._reference_bjd = None

        self._building = True

        self.passband = ipw.Dropdown(
            options=sorted({str(p) for p in self.table["passband"]}),
            description="Passband",
        )

        quantities, missing = available_quantities(self.table)
        self.quantity = ipw.Dropdown(options=quantities, description="Show")
        # Faded so that it does not compete with the controls.
        self.quantity_note = ipw.HTML(layout=ipw.Layout(max_width="420px"))

        x_choices = [
            c
            for c in X_AXIS_CHOICES
            if ("airmass" if c == "Airmass" else "bjd") in self.table.colnames
        ]
        self.x_axis = ipw.Dropdown(options=x_choices, description="X axis")

        self.stars = ipw.SelectMultiple(options=[], rows=10, description="Stars")
        self.offset = ipw.FloatSlider(
            min=0, max=1, step=0.01, value=0, description="Offset"
        )
        self.error_bars = ipw.Checkbox(value=True, description="Error bars")
        self.title = ipw.Text(description="Title")

        self._missing = missing
        self._update_star_options(default=True)

        self.x_scale = LinearScale()
        self.y_scale = LinearScale()
        self.figure = Figure(
            marks=[],
            axes=[
                Axis(scale=self.x_scale, label="", num_ticks=6),
                Axis(scale=self.y_scale, orientation="vertical", label=""),
            ],
            title="",
            legend_location="top-right",
            min_aspect_ratio=1.2,
            fig_margin=dict(top=50, bottom=50, left=70, right=20),
        )
        self.figure.layout.width = "650px"
        self.figure.layout.height = "450px"

        self.controls = ipw.VBox(
            [
                self.passband,
                self.quantity,
                self.quantity_note,
                self.x_axis,
                self.stars,
                self.offset,
                self.error_bars,
                self.title,
            ]
        )
        self.children = [self.figure, self.controls]

        self._building = False
        self._update_note()
        self.passband.observe(self._on_passband, names="value")
        for widget in (
            self.quantity,
            self.x_axis,
            self.stars,
            self.offset,
            self.error_bars,
        ):
            widget.observe(self._on_change, names="value")
        self.title.observe(self._on_title, names="value")
        self._redraw()

    @staticmethod
    def _load_photometry(photometry):
        """Read the photometry file if given a path, and make sure of a bjd."""
        if isinstance(photometry, Table):
            table = photometry
        else:
            from stellarphot import PhotometryData

            table = PhotometryData.read(photometry)
        if "bjd" not in table.colnames and hasattr(table, "add_bjd_col"):
            table.add_bjd_col()
        return table

    def _passband_table(self):
        """The rows of the table in the chosen passband."""
        in_band = np.asarray(self.table["passband"]).astype(str) == self.passband.value
        return self.table[in_band]

    def _update_note(self):
        """Say which quantities are not on offer, and why."""
        notes = [
            f"{label} not available: no {column} column in the file"
            for label, column in self._missing.items()
        ]
        quantity = self.quantity.value
        if quantity is not None:
            error_col = QUANTITIES[quantity][1]
            if error_col not in self.table.colnames:
                notes.append(
                    f"{quantity}: no {error_col} column in the file, "
                    "so error bars are not available"
                )
        self.quantity_note.value = (
            '<span style="opacity: 0.6; font-size: 85%">' + "<br>".join(notes) + "</span>"
        )

    def _update_star_options(self, default=False):
        """Refresh the star list for the passband, keeping what is still there."""
        old = set(self.stars.value)
        options = star_options(self._passband_table(), self.source_list)
        present = {star_id for _, star_id in options}
        if default:
            selected = default_stars(options, self.source_list)
        else:
            selected = tuple(s for s in sorted(old) if s in present)
            if not selected:
                selected = default_stars(options, self.source_list)
        self.stars.options = options
        self.stars.value = selected

    def _on_passband(self, change):
        # Changing the options fires the star observer with a selection
        # that does not match them yet, so hold off drawing until it is done.
        self._building = True
        try:
            self._update_star_options()
        finally:
            self._building = False
        self._redraw()

    def _on_title(self, change):
        self.figure.title = change["new"]

    def _on_change(self, change):
        if change["owner"] is self.quantity:
            self._update_note()
        self._redraw()

    def _redraw(self):
        """Rebuild the marks of the figure from the current controls."""
        if self._building:
            return
        quantity = self.quantity.value
        choice = self.x_axis.value
        selected = list(self.stars.value)
        self.figure.title = self.title.value
        if quantity is None or choice is None:
            self._install_marks([])
            return

        _, _, is_magnitude = QUANTITIES[quantity]
        self.y_scale.reverse = is_magnitude

        data = self._passband_table()
        x_all, x_label = x_axis_values(data, choice, self._reference_bjd)
        y_all, err_all = quantity_values(data, quantity)
        ids = np.asarray(data["star_id"]).astype(int)
        self.figure.axes[0].label = x_label
        self.figure.axes[1].label = quantity

        show_errors = self.error_bars.value and err_all is not None
        marks = []
        x_values = []
        y_low, y_high = [], []
        for i, star_id in enumerate(selected):
            rows = ids == star_id
            x = x_all[rows]
            # The y axis of a magnitude plot is reversed, so subtract to
            # move each successive star up on the screen.
            shift = i * self.offset.value
            y = y_all[rows] - shift if is_magnitude else y_all[rows] + shift
            err = err_all[rows] if err_all is not None else None
            good = np.isfinite(x) & np.isfinite(y)
            x, y = x[good], y[good]
            label = next(lab for lab, sid in self.stars.options if sid == star_id)
            color = COLORS[i % len(COLORS)]
            scatter = Scatter(
                x=x,
                y=y,
                scales={"x": self.x_scale, "y": self.y_scale},
                colors=[color],
                labels=[label],
                display_legend=True,
                default_size=24,
            )
            marks.append(scatter)
            x_values.append(x)
            y_low.append(y)
            y_high.append(y)
            if show_errors:
                # A point with no error is still plotted, just without a bar.
                e = err[good]
                has_error = np.isfinite(e)
                xb, yb, e = x[has_error], y[has_error], e[has_error]
                bars = Lines(
                    x=np.column_stack([xb, xb]),
                    y=np.column_stack([yb - e, yb + e]),
                    scales={"x": self.x_scale, "y": self.y_scale},
                    colors=[color],
                    display_legend=False,
                )
                marks.append(bars)
                y_low.append(yb - e)
                y_high.append(yb + e)
        self._install_marks(marks)
        self._set_range(self.x_scale, x_values)
        self._set_range(self.y_scale, y_low + y_high)

    @staticmethod
    def _set_range(scale, arrays):
        """
        Fit ``scale`` to the plotted data in ``arrays``.

        Setting the limits explicitly, rather than leaving the scale to fit
        itself, makes sure each axis follows the current selection instead
        of hanging on to the range of whatever was plotted before.
        """
        values = np.concatenate([np.ravel(v) for v in arrays] or [[]])
        values = values[np.isfinite(values)]
        if values.size == 0:
            scale.min = None
            scale.max = None
            return
        low, high = values.min(), values.max()
        pad = (high - low) * AXIS_PADDING or 0.05
        scale.min = float(low - pad)
        scale.max = float(high + pad)

    def _install_marks(self, marks):
        """Put ``marks`` on the figure and close the marks they replace.

        Assigning ``figure.marks`` only detaches the old marks; ipywidgets
        keeps them (and their data arrays) registered until they are closed.
        """
        old = list(self.figure.marks)
        self.figure.marks = marks
        for mark in old:
            mark.close()

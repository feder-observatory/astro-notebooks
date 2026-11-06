import re
import warnings

import ipywidgets as ipw
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from bqplot import Axis, Figure, LinearScale, Lines, PanZoom, Scatter
from bqplot.colorschemes import CATEGORY10

# Marker names (in the source list) of the stars a student most likely
# wants to see first. VSX is deliberately not here: a field can have dozens
# of VSX stars, and starting with all of them plotted is a mess.
DEFAULT_MARKER_NAMES = ("TESS Targets",)

# Room beyond the data on each axis, as a fraction of the range of the data.
AXIS_PADDING = 0.05

# bqplot's own colour cycle, so each star gets a colour of its own.
COLORS = list(CATEGORY10)

# Marker shapes, one per night, when more than one night is plotted at once.
MARKERS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "plus")

# File name of the PNG when the plot has no title.
DEFAULT_PNG_NAME = "light-curve"

X_AXIS_CHOICES = ("BJD", "Hours since first image", "Airmass")

# Value of the night dropdown that means every night in the file.
ALL_NIGHTS = "all"

# Images further apart than this, in days, are on different nights when the
# table has no night column to say so.
NIGHT_GAP = 0.5

# Longitude, in degrees east, that nights are told apart by when the table
# does not say where it was observed from. Greenwich: a night key is then
# the UTC date of the noon before the observation.
DEFAULT_LONGITUDE = 0.0

# label -> (value column or None, error column, is a magnitude)
# The differential magnitude has no column of its own; it is computed from
# the relative flux. Both are scaled star by star so that the median
# relative flux of each star is 1, and its median differential magnitude 0.
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


def _star_medians(table, values):
    """
    The median of ``values`` for each star, repeated for the rows of the star.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table. All rows count as one star if there is no
        ``star_id`` column.
    values : numpy.ndarray
        One value per row of ``table``; NaN is ignored.

    Returns
    -------
    numpy.ndarray
        The median of the star for each row. Rows of a star with no finite,
        positive median get 1, so that dividing by the result leaves them
        unchanged.
    """
    medians = np.ones_like(values)
    if "star_id" in table.colnames:
        ids = np.asarray(table["star_id"]).astype(int)
    else:
        ids = np.zeros(len(values), dtype=int)
    for star in np.unique(ids):
        rows = ids == star
        with warnings.catch_warnings():
            # An all-NaN star is a warning here and a NaN below, not an error.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values[rows])
        if np.isfinite(median) and median > 0:
            medians[rows] = median
    return medians


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
        The values to plot. The relative flux is divided by the median
        relative flux of each star, so each star has a median of 1; the
        differential magnitude is computed from that scaled flux, so each
        star has a median of 0.
    errors : numpy.ndarray or None
        One-sigma errors in the same units as ``values``, or None if the
        table has no error column for this quantity.
    """
    value_col, error_col, _ = QUANTITIES[quantity]
    if quantity in ("Relative flux", "Differential magnitude"):
        flux = _as_float_array(table["relative_flux"])
        medians = _star_medians(table, flux)
        flux = flux / medians
        if error_col in table.colnames:
            flux_error = _as_float_array(table[error_col]) / medians
        else:
            flux_error = None
        if quantity == "Relative flux":
            return flux, flux_error
        with np.errstate(divide="ignore", invalid="ignore"):
            values = -2.5 * np.log10(flux)
            if flux_error is not None:
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


def observatory_longitude(table):
    """
    The longitude the table was observed from, if it says.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table. A `stellarphot.PhotometryData` carries the
        observatory as a table attribute; a plain table read from a
        stellarphot file has it, serialised, in
        ``meta["__attributes__"]["observatory"]``.

    Returns
    -------
    float or None
        Longitude in degrees, east positive, between -180 and 180; None if
        the table does not say.
    """
    observatory = getattr(table, "observatory", None)
    if observatory is None:
        observatory = table.meta.get("__attributes__", {}).get("observatory")
    if isinstance(observatory, dict):
        longitude = observatory.get("longitude")
    else:
        longitude = getattr(observatory, "longitude", None)
    if longitude is None:
        return None
    try:
        longitude = float(u.Quantity(longitude, u.deg).to_value(u.deg))
    except (TypeError, ValueError, u.UnitConversionError):
        return None
    if not np.isfinite(longitude):
        return None
    return (longitude + 180) % 360 - 180


def night_keys(table, longitude=None):
    """
    Work out which night each row of a table belongs to.

    Parameters
    ----------
    table : `astropy.table.Table`
        Photometry table. The ``night`` column stellarphot writes, the MJD of
        local noon before the evening of the observation, is used if there is
        one. Otherwise images are grouped by gaps of more than `NIGHT_GAP`
        days in ``bjd``, and each group is keyed the way stellarphot does it:
        the MJD of the local noon before its first image.
    longitude : float, optional
        Longitude of the observatory in degrees east, used to find local
        noon when there is no ``night`` column. The longitude in the table,
        if there is one (see `observatory_longitude`), or
        `DEFAULT_LONGITUDE` if not given.

    Returns
    -------
    numpy.ndarray
        Integer key of the night of each row; -1 for a row whose night
        cannot be told. Different nights always have different keys.
    """
    if "night" in table.colnames:
        night = _as_float_array(table["night"])
        return np.where(np.isfinite(night), night, -1).astype(int)
    keys = np.full(len(table), -1, dtype=int)
    if "bjd" not in table.colnames:
        return keys
    bjd = _as_float_array(table["bjd"])
    finite = np.isfinite(bjd)
    if not finite.any():
        return keys
    if longitude is None:
        longitude = observatory_longitude(table)
    if longitude is None:
        longitude = DEFAULT_LONGITUDE
    order = np.argsort(bjd[finite], kind="stable")
    sorted_bjd = bjd[finite][order]
    starts = np.concatenate([[True], np.diff(sorted_bjd) > NIGHT_GAP])
    group = np.cumsum(starts) - 1
    # Local time is UTC plus a whole number of hours, as in stellarphot, so
    # that a file with a night column and one without label a night alike.
    hours_east = int(longitude / 15)
    local_mjd = sorted_bjd[starts] - 2400000.5 + hours_east / 24
    group_key = np.floor(local_mjd - 0.5).astype(int)
    # Two nights could only share a key if one had a lone image within a
    # few hours of noon; still, each night gets a key of its own.
    for i in range(1, group_key.size):
        group_key[i] = max(group_key[i], group_key[i - 1] + 1)
    sorted_keys = np.empty(sorted_bjd.size, dtype=int)
    sorted_keys[order] = group_key[group]
    keys[finite] = sorted_keys
    return keys


def night_label(key):
    """
    The date of a night, as text, from its key.

    The key is the MJD of local noon before the night, so the label is the
    date of the evening the night began.
    """
    return Time(int(key), format="mjd", scale="utc").to_value("iso", subfmt="date")


def png_file_name(title):
    """
    The name of the PNG file to save a plot as.

    Parameters
    ----------
    title : str
        Title of the plot.

    Returns
    -------
    str
        The title with characters that cannot be in a file name replaced,
        and ``.png`` on the end; or ``light-curve.png`` if the title is
        empty.
    """
    name = re.sub(r'[\\/:*?"<>|]+', "-", title)
    name = " ".join(name.split()).strip(" .-")
    return (name or DEFAULT_PNG_NAME) + ".png"


def night_options(keys):
    """
    The nights to offer in a dropdown.

    Parameters
    ----------
    keys : numpy.ndarray
        Output of `night_keys`.

    Returns
    -------
    list of tuple
        ``(label, key)`` for each night, earliest first. Rows with no night
        are left out.
    """
    return [(night_label(k), int(k)) for k in np.unique(keys) if k >= 0]


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
    night : `ipywidgets.Dropdown`
        The night to plot, or `ALL_NIGHTS`. Only shown in the controls when
        the file has more than one night.
    pan_zoom : `ipywidgets.ToggleButton`
        Lets the mouse pan (drag) and zoom (scroll) the plot.
    reset_view : `ipywidgets.Button`
        Fits the axes to the plotted points again.
    save_png : `ipywidgets.Button`
        Downloads the plot as a PNG image, named after the title.
    """

    def __init__(self, photometry, source_list=None, **kwargs):
        super().__init__(**kwargs)
        self.table = self._load_photometry(photometry)
        if source_list is None or isinstance(source_list, Table):
            self.source_list = source_list
        else:
            self.source_list = Table.read(source_list)

        self._night_keys = night_keys(self.table)
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

        self.night = ipw.Dropdown(
            options=[("All nights", ALL_NIGHTS)], description="Night"
        )
        self.stars = ipw.SelectMultiple(options=[], rows=10, description="Stars")
        self.offset = ipw.FloatSlider(
            min=0, max=1, step=0.01, value=0, description="Offset"
        )
        self.error_bars = ipw.Checkbox(value=True, description="Error bars")
        self.title = ipw.Text(description="Title")

        self._missing = missing
        self._update_night_options()
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
            # bqplot's own toolbar is left out: its reset button puts back
            # the axis ranges from when pan/zoom was first switched on, which
            # are stale once another star or quantity has been plotted.
            display_toolbar=False,
        )
        self.figure.layout.width = "650px"
        self.figure.layout.height = "450px"

        self._pan_zoom = PanZoom(scales={"x": [self.x_scale], "y": [self.y_scale]})
        button_layout = ipw.Layout(width="auto")
        self.pan_zoom = ipw.ToggleButton(
            value=False,
            description="Pan/zoom",
            icon="arrows",
            tooltip="Drag to pan, scroll to zoom",
            layout=button_layout,
        )
        self.reset_view = ipw.Button(
            description="Reset view",
            icon="refresh",
            tooltip="Fit the axes to the plotted points",
            layout=button_layout,
        )
        self.save_png = ipw.Button(
            description="Save PNG",
            icon="camera",
            tooltip="Download the plot as an image",
            layout=button_layout,
        )
        self.view_buttons = ipw.HBox([self.pan_zoom, self.reset_view, self.save_png])
        self.plot = ipw.VBox([self.figure, self.view_buttons])

        controls = [
            self.passband,
            self.quantity,
            self.quantity_note,
            self.x_axis,
            self.night,
            self.stars,
            self.offset,
            self.error_bars,
            self.title,
        ]
        if len(night_options(self._night_keys)) < 2:
            controls.remove(self.night)
        self.controls = ipw.VBox(controls)
        self.children = [self.plot, self.controls]

        self._building = False
        self._update_note()
        self.passband.observe(self._on_passband, names="value")
        for widget in (
            self.quantity,
            self.x_axis,
            self.night,
            self.stars,
            self.offset,
            self.error_bars,
        ):
            widget.observe(self._on_change, names="value")
        self.title.observe(self._on_title, names="value")
        self.pan_zoom.observe(self._on_pan_zoom, names="value")
        self.reset_view.on_click(lambda button: self.fit_axes())
        self.save_png.on_click(
            lambda button: self.figure.save_png(png_file_name(self.title.value))
        )
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

    def _passband_rows(self):
        """Mask of the rows of the table in the chosen passband."""
        return np.asarray(self.table["passband"]).astype(str) == self.passband.value

    def _passband_table(self):
        """The rows of the table in the chosen passband."""
        return self.table[self._passband_rows()]

    def _reference_bjd(self, night):
        """
        The BJD that is hour zero of "Hours since first image".

        It is the first image of ``night`` in the whole file, whatever the
        passband, so that the passbands line up; or the first image of the
        whole file when every night is shown.
        """
        if "bjd" not in self.table.colnames:
            return None
        bjd = _as_float_array(self.table["bjd"])
        if night != ALL_NIGHTS:
            bjd = bjd[self._night_keys == night]
        if not np.isfinite(bjd).any():
            return None
        return np.nanmin(bjd)

    def _update_night_options(self):
        """Offer the nights in the passband, keeping the chosen one if it is there."""
        old = self.night.value
        options = [("All nights", ALL_NIGHTS)]
        options += night_options(self._night_keys[self._passband_rows()])
        self.night.options = options
        self.night.value = old if old in {key for _, key in options} else ALL_NIGHTS

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
            self._update_night_options()
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

    def _on_pan_zoom(self, change):
        self.figure.interaction = self._pan_zoom if change["new"] else None

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

        night = self.night.value
        in_band = self._passband_rows()
        data = self.table[in_band]
        x_all, x_label = x_axis_values(data, choice, self._reference_bjd(night))
        # The whole passband goes in, so that each star is scaled by its
        # median over every night, not only the night on show.
        y_all, err_all = quantity_values(data, quantity)
        ids = np.asarray(data["star_id"]).astype(int)
        if night == ALL_NIGHTS:
            in_night = np.ones(len(data), dtype=bool)
        else:
            in_night = self._night_keys[in_band] == night
        self.figure.axes[0].label = x_label
        self.figure.axes[1].label = quantity

        # With several nights on the plot, each gets a marker of its own.
        night_marker = {}
        if night == ALL_NIGHTS:
            nights = [key for _, key in self.night.options if key != ALL_NIGHTS]
            if len(nights) > 1:
                night_marker = {key: MARKERS[i % len(MARKERS)] for i, key in enumerate(nights)}
        keys = self._night_keys[in_band]

        show_errors = self.error_bars.value and err_all is not None
        marks = []
        for i, star_id in enumerate(selected):
            rows = (ids == star_id) & in_night
            x = x_all[rows]
            # The y axis of a magnitude plot is reversed, so subtract to
            # move each successive star up on the screen.
            shift = i * self.offset.value
            y = y_all[rows] - shift if is_magnitude else y_all[rows] + shift
            err = err_all[rows] if err_all is not None else None
            good = np.isfinite(x) & np.isfinite(y)
            x, y = x[good], y[good]
            star_keys = keys[rows][good]
            label = next(lab for lab, sid in self.stars.options if sid == star_id)
            color = COLORS[i % len(COLORS)]
            if night_marker:
                groups = [
                    (star_keys == key, f"{label}, {night_label(key)}", marker)
                    for key, marker in night_marker.items()
                    if (star_keys == key).any()
                ]
            else:
                groups = [(np.ones(x.size, dtype=bool), label, MARKERS[0])]
            for in_group, group_label, marker in groups:
                scatter = Scatter(
                    x=x[in_group],
                    y=y[in_group],
                    scales={"x": self.x_scale, "y": self.y_scale},
                    colors=[color],
                    labels=[group_label],
                    marker=marker,
                    display_legend=True,
                    default_size=24,
                )
                marks.append(scatter)
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
        self._install_marks(marks)
        self.fit_axes()

    def fit_axes(self):
        """
        Fit both axes to the marks on the figure, with a little room around them.

        This is what the "Reset view" button does, and what happens after
        every redraw, so a zoomed plot always comes back to the points that
        are plotted now.
        """
        marks = self.figure.marks
        self._set_range(self.x_scale, [m.x for m in marks])
        self._set_range(self.y_scale, [m.y for m in marks])

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

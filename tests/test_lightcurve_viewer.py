import numpy as np
import pytest
from astropy.time import Time
from astropy.table import MaskedColumn, QTable, Table

from astro_notebooks.lightcurve_viewer import (
    LightCurveViewer,
    available_quantities,
    default_stars,
    quantity_values,
    star_options,
    x_axis_values,
)

COLUMNS = [
    "star_id",
    "passband",
    "bjd",
    "airmass",
    "mag_inst",
    "mag_error",
    "relative_flux",
    "relative_flux_error",
    "mag_cal",
    "mag_cal_error",
]
N_POINTS = 6


def make_table(drop=()):
    rng = np.random.default_rng(3)
    rows = []
    # V has stars 1, 2, 3; R has stars 2, 3, 4
    for passband, stars in (("V", (1, 2, 3)), ("R", (2, 3, 4))):
        for star in stars:
            for k in range(N_POINTS):
                bjd = 2459910.0 + k * 0.01 + (0.5 if passband == "R" else 0)
                rows.append(
                    (
                        star,
                        passband,
                        bjd,
                        1.2 + 0.01 * k,
                        15.0 + rng.normal(0, 0.1),
                        0.01,
                        1.0 + rng.normal(0, 0.02),
                        0.02,
                        16.0 + rng.normal(0, 0.1),
                        0.03,
                    )
                )
    table = QTable(rows=rows, names=COLUMNS)
    # a few NaNs
    table["mag_inst"][0] = np.nan  # star 1, V
    table["relative_flux"][1] = np.nan  # star 1, V
    table.remove_columns(list(drop))
    return table


@pytest.fixture
def table():
    return make_table()


@pytest.fixture
def source_list():
    return Table(
        {
            "star_id": [1, 2, 3],
            "marker name": ["VSX", "APASS comparison", "TESS Targets"],
        }
    )


def n_marks(viewer, kind):
    return sum(type(m).__name__ == kind for m in viewer.figure.marks)


def test_instrumental_values_and_errors(table):
    values, errors = quantity_values(table, "Instrumental magnitude")
    np.testing.assert_allclose(values, table["mag_inst"], equal_nan=True)
    np.testing.assert_allclose(errors, 0.01)


def test_relative_flux_values_and_errors(table):
    values, errors = quantity_values(table, "Relative flux")
    np.testing.assert_allclose(values, table["relative_flux"], equal_nan=True)
    np.testing.assert_allclose(errors, 0.02)


def test_calibrated_values_and_errors(table):
    values, errors = quantity_values(table, "Calibrated magnitude")
    np.testing.assert_allclose(values, table["mag_cal"])
    np.testing.assert_allclose(errors, 0.03)


def test_differential_magnitude_by_hand():
    t = QTable(
        {"relative_flux": [1.0, 0.5, 2.0], "relative_flux_error": [0.1, 0.1, 0.1]}
    )
    values, errors = quantity_values(t, "Differential magnitude")
    np.testing.assert_allclose(values, [0.0, 0.75257498916, -0.75257498916])
    # 2.5 / ln(10) * 0.1 / flux
    np.testing.assert_allclose(
        errors, [0.10857362, 0.21714724, 0.05428681], rtol=1e-6
    )


def test_missing_error_column_gives_none(table):
    t = make_table(drop=["mag_error"])
    values, errors = quantity_values(t, "Instrumental magnitude")
    assert errors is None
    assert len(values) == len(t)


def test_all_quantities_available(table):
    available, missing = available_quantities(table)
    assert available == [
        "Instrumental magnitude",
        "Relative flux",
        "Differential magnitude",
        "Calibrated magnitude",
    ]
    assert missing == {}


def test_quantities_left_out_when_columns_missing():
    t = make_table(drop=["mag_cal", "relative_flux"])
    available, missing = available_quantities(t)
    assert available == ["Instrumental magnitude"]
    assert missing == {
        "Relative flux": "relative_flux",
        "Differential magnitude": "relative_flux",
        "Calibrated magnitude": "mag_cal",
    }


def test_x_axis_choices(table):
    values, label = x_axis_values(table, "BJD")
    np.testing.assert_allclose(values, table["bjd"])
    assert label == "BJD"

    hours, label = x_axis_values(table, "Hours since first image")
    assert hours.min() == 0
    np.testing.assert_allclose(hours[1], 0.01 * 24)
    assert "Hours" in label

    airmass, label = x_axis_values(table, "Airmass")
    np.testing.assert_allclose(airmass, table["airmass"])
    assert label == "Airmass"


def test_hours_uses_reference_bjd(table):
    hours, _ = x_axis_values(table, "Hours since first image", reference_bjd=2459909.0)
    np.testing.assert_allclose(hours[0], 24.0)


def test_star_options_labels(table, source_list):
    options = star_options(table, source_list)
    assert options == [
        ("  1 VSX", 1),
        ("  2 APASS comparison", 2),
        ("  3 TESS Targets", 3),
        ("  4", 4),
    ]


def test_star_options_without_source_list(table):
    assert [s for _, s in star_options(table)] == [1, 2, 3, 4]
    assert star_options(table)[0][0] == "  1"


def test_default_stars(table, source_list):
    options = star_options(table, source_list)
    # only the TESS target: VSX can mark dozens of stars in a field
    assert default_stars(options, source_list) == (3,)
    # no marker of interest: first star
    other = Table({"star_id": [1, 2], "marker name": ["a", "b"]})
    assert default_stars(star_options(table, other), other) == (1,)
    assert default_stars([], None) == ()


def test_viewer_defaults(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    assert viewer.passband.value == "R"  # first of sorted
    assert viewer.stars.value == (3,)
    viewer.passband.value = "V"
    assert [s for _, s in viewer.stars.options] == [1, 2, 3]


def test_viewer_default_selection_is_marked_star(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    # R opens with stars 2, 3, 4; only star 3 is marked, and star 2 (the
    # first option) must not be chosen instead.
    assert viewer.stars.options[0][1] == 2
    assert viewer.stars.value == (3,)
    assert n_marks(viewer, "Scatter") == 1


def test_scatter_per_star_with_labels(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (1, 2, 3)
    scatters = [m for m in viewer.figure.marks if type(m).__name__ == "Scatter"]
    assert len(scatters) == 3
    assert [s.labels[0] for s in scatters] == [
        "  1 VSX",
        "  2 APASS comparison",
        "  3 TESS Targets",
    ]
    assert len({s.colors[0] for s in scatters}) == 3
    assert all(s.display_legend for s in scatters)


def test_passband_change_keeps_overlap(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (1, 2)
    viewer.passband.value = "R"
    assert [s for _, s in viewer.stars.options] == [2, 3, 4]
    assert viewer.stars.value == (2,)
    assert n_marks(viewer, "Scatter") == 1


def test_passband_change_with_no_overlap(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (1,)
    viewer.passband.value = "R"
    # none of the old selection is in R, so the default is chosen
    assert viewer.stars.value == (3,)
    assert n_marks(viewer, "Scatter") == 1


def test_offset_applied_per_star(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (2, 3)
    # (quantity, sign): magnitudes are plotted with a reversed y axis, so
    # moving up on the screen means smaller values.
    for quantity, sign in (
        ("Calibrated magnitude", -1),
        ("Instrumental magnitude", -1),
        ("Relative flux", 1),
    ):
        viewer.offset.value = 0
        viewer.quantity.value = quantity
        before = [
            m.y.copy() for m in viewer.figure.marks if type(m).__name__ == "Scatter"
        ]
        viewer.offset.value = 0.5
        after = [m.y for m in viewer.figure.marks if type(m).__name__ == "Scatter"]
        for i, (b, a) in enumerate(zip(before, after)):
            np.testing.assert_allclose(a - b, sign * 0.5 * i)


def test_error_bars_follow_checkbox(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    assert n_marks(viewer, "Lines") == 1
    lines = [m for m in viewer.figure.marks if type(m).__name__ == "Lines"][0]
    scatter = [m for m in viewer.figure.marks if type(m).__name__ == "Scatter"][0]
    assert lines.x.shape == (len(scatter.x), 2)
    np.testing.assert_allclose(lines.y[:, 1] - lines.y[:, 0], 2 * 0.01)
    assert lines.colors == scatter.colors
    viewer.error_bars.value = False
    assert n_marks(viewer, "Lines") == 0
    viewer.error_bars.value = True
    assert n_marks(viewer, "Lines") == 1


def test_missing_error_column_means_no_error_bars(source_list):
    viewer = LightCurveViewer(make_table(drop=["mag_error"]), source_list)
    assert n_marks(viewer, "Scatter") == 1
    assert n_marks(viewer, "Lines") == 0
    assert "mag_error" in viewer.quantity_note.value
    viewer.quantity.value = "Relative flux"
    assert n_marks(viewer, "Lines") == 1


def test_note_lists_missing_quantities(source_list):
    viewer = LightCurveViewer(make_table(drop=["mag_cal"]), source_list)
    assert "Calibrated magnitude" not in viewer.quantity.options
    assert (
        "Calibrated magnitude not available: no mag_cal column in the file"
        in viewer.quantity_note.value
    )


def test_nan_points_dropped(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (1,)
    scatter = viewer.figure.marks[0]
    # star 1 has one NaN mag_inst out of N_POINTS
    assert len(scatter.x) == N_POINTS - 1
    assert np.all(np.isfinite(scatter.y))
    viewer.quantity.value = "Relative flux"
    assert len(viewer.figure.marks[0].x) == N_POINTS - 1


def test_nan_error_keeps_point_but_not_bar():
    t = QTable(
        {
            "star_id": [1] * 6,
            "passband": ["V"] * 6,
            "bjd": 2459910.0 + np.arange(6) * 0.01,
            "mag_inst": [15.0, 15.1, np.nan, np.nan, 15.2, 15.3],
            "mag_error": [0.01, np.nan, 0.01, 0.01, np.nan, 0.01],
        }
    )
    viewer = LightCurveViewer(t)
    scatter = [m for m in viewer.figure.marks if type(m).__name__ == "Scatter"][0]
    assert len(scatter.x) == 4
    assert n_marks(viewer, "Lines") == 1
    lines = [m for m in viewer.figure.marks if type(m).__name__ == "Lines"][0]
    assert lines.x.shape == (2, 2)
    viewer.error_bars.value = False
    assert n_marks(viewer, "Lines") == 0
    scatter = [m for m in viewer.figure.marks if type(m).__name__ == "Scatter"][0]
    assert len(scatter.x) == 4


def test_source_list_without_marker_name_column(table):
    assert star_options(table, Table({"star_id": [1, 2]}))[0] == ("  1", 1)


def test_masked_time_and_column_give_nan():
    mask = [False, True, False]
    t = QTable(
        {
            "bjd": Time(
                np.ma.array([2459910.0, 2459910.1, 2459910.2], mask=mask),
                format="jd",
                scale="tdb",
            ),
            "relative_flux": MaskedColumn([1.0, 2.0, 3.0], mask=mask),
        }
    )
    x, _ = x_axis_values(t, "BJD")
    np.testing.assert_allclose(x, [2459910.0, np.nan, 2459910.2])
    values, errors = quantity_values(t, "Relative flux")
    np.testing.assert_allclose(values, [1.0, np.nan, 3.0])
    assert errors is None


def test_empty_selection_gives_no_marks(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.stars.value = ()
    assert viewer.figure.marks == []


def test_y_axis_reversed_for_magnitudes(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    assert viewer.y_scale.reverse is True
    viewer.quantity.value = "Relative flux"
    assert viewer.y_scale.reverse is False
    viewer.quantity.value = "Differential magnitude"
    assert viewer.y_scale.reverse is True
    assert viewer.figure.axes[1].label == "Differential magnitude"


def test_x_axis_dropdown_changes_marks_and_label(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.x_axis.value = "Airmass"
    assert viewer.figure.axes[0].label == "Airmass"
    assert viewer.figure.marks[0].x.max() < 2
    viewer.x_axis.value = "Hours since first image"
    # R starts half a day after V in the whole table
    assert viewer.figure.marks[0].x.min() == pytest.approx(12.0)


def test_title(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.title.value = "My star"
    assert viewer.figure.title == "My star"


def test_figure_is_kept(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    figure = viewer.figure
    viewer.quantity.value = "Relative flux"
    viewer.passband.value = "V"
    assert viewer.figure is figure
    assert viewer.children[0] is figure


def test_loads_from_files(tmp_path, table, source_list):
    photometry_path = tmp_path / "photometry.ecsv"
    source_path = tmp_path / "source_locations.ecsv"
    table.write(photometry_path)
    source_list.write(source_path)

    # PhotometryData needs a full set of stellarphot columns, so read the
    # file as a plain table here.
    from unittest import mock

    with mock.patch("stellarphot.PhotometryData.read", QTable.read):
        viewer = LightCurveViewer(photometry_path, source_path)
    assert viewer.stars.value == (3,)
    assert viewer.stars.options[0][0] == "  2 APASS comparison"


def test_redraw_closes_replaced_marks(table, source_list):
    from ipywidgets import Widget

    def ids(marks):
        return {m.model_id for m in marks}

    viewer = LightCurveViewer(table, source_list)
    viewer.error_bars.value = True
    first = list(viewer.figure.marks)
    first_ids = ids(first)
    registered = set(Widget._active_widgets)
    # The offset slider fires continuously while dragging, so a redraw must
    # not leave the marks it replaces registered.
    for i in range(20):
        viewer.offset.value = 0.1 * (i + 1)
    assert all(m.comm is None for m in first)
    assert not first_ids & set(Widget._active_widgets)
    # Only the current marks are new since the baseline.
    assert set(Widget._active_widgets) - registered == ids(viewer.figure.marks)
    # Clearing the plot closes the last marks too.
    last = list(viewer.figure.marks)
    viewer.stars.value = ()
    assert viewer.figure.marks == []
    assert all(m.comm is None for m in last)


def test_y_range_follows_selection(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.error_bars.value = False

    def extent(star):
        rows = (table["passband"] == "V") & (table["star_id"] == star)
        y = np.asarray(table["mag_inst"][rows], dtype=float)
        return np.nanmin(y), np.nanmax(y)

    for star in (2, 3):
        viewer.stars.value = (star,)
        low, high = extent(star)
        pad = (high - low) * 0.05
        assert viewer.y_scale.min == pytest.approx(low - pad)
        assert viewer.y_scale.max == pytest.approx(high + pad)
    # error bars widen the range
    viewer.error_bars.value = True
    low, high = extent(3)
    assert viewer.y_scale.min < low - 0.01
    assert viewer.y_scale.max > high + 0.01
    # nothing plotted: back to automatic
    viewer.stars.value = ()
    assert viewer.y_scale.min is None and viewer.y_scale.max is None


def test_x_range_follows_axis_choice(table, source_list):
    viewer = LightCurveViewer(table, source_list)
    viewer.passband.value = "V"
    viewer.stars.value = (2,)
    rows = (table["passband"] == "V") & (table["star_id"] == 2)
    bjd = np.asarray(table["bjd"][rows], dtype=float)
    first = np.nanmin(np.asarray(table["bjd"], dtype=float))
    expected = {
        "BJD": bjd,
        "Hours since first image": (bjd - first) * 24,
        "Airmass": np.asarray(table["airmass"][rows], dtype=float),
    }
    # Visit the choices twice so each change starts from a different range.
    for choice in list(expected) + list(expected):
        viewer.x_axis.value = choice
        low, high = np.nanmin(expected[choice]), np.nanmax(expected[choice])
        pad = (high - low) * 0.05
        assert viewer.x_scale.min == pytest.approx(low - pad)
        assert viewer.x_scale.max == pytest.approx(high + pad)


def test_note_is_faded(source_list):
    viewer = LightCurveViewer(make_table(drop=["mag_cal"]), source_list)
    assert "opacity: 0.6" in viewer.quantity_note.value
    assert viewer.quantity_note.layout.max_width == "420px"

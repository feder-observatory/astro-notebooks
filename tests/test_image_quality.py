import os

import numpy as np
import pytest

from astro_notebooks.image_quality import (
    FWHM_PER_SIGMA,
    find_stars_in_tile,
    measure_frame,
    measure_star,
    select_reference_stars,
    summarize_metrics,
)
from astro_notebooks.image_selector import ImageSelect

from .conftest import (
    BROAD_FRAME,
    DIM_FRAME,
    SATURATED_POSITION,
    STAR_FLUXES,
    STAR_POSITIONS,
    STAR_SIGMA,
    star_image,
    write_star_image,
)

TRUE_FWHM = FWHM_PER_SIGMA * STAR_SIGMA


def _sorted_positions(positions):
    """Round (x, y) positions to whole pixels and sort them, for comparing."""
    return sorted((round(x), round(y)) for x, y in positions)


def _near(position, expected, tolerance=1.5):
    """Whether two (x, y) positions are within ``tolerance`` pixels of each other."""
    return np.hypot(position[0] - expected[0],
                    position[1] - expected[1]) < tolerance


def test_find_stars_in_tile_skips_saturated_star():
    """The star finder returns the real stars and never the saturated one.

    A saturated star has a flat top, so its fitted width is meaningless
    and it would make every frame look bad. Also checks that the stars
    come back brightest first, which is the order the reference stars are
    chosen in, and that the rough FWHM is close to the truth.
    """
    stars = find_stars_in_tile(star_image(seed=11))
    found = _sorted_positions((s['x'], s['y']) for s in stars)
    assert found == _sorted_positions(STAR_POSITIONS)
    assert not any(_near((s['x'], s['y']), SATURATED_POSITION, tolerance=5)
                   for s in stars)
    # brightest first
    assert [s['peak'] for s in stars] == sorted(
        (s['peak'] for s in stars), reverse=True)
    for star in stars:
        assert star['fwhm'] == pytest.approx(TRUE_FWHM, rel=0.1)


def test_find_stars_in_tile_reports_full_frame_positions():
    """Positions are given in full-frame pixels, not pixels within the tile.

    Tiles are cut from a large frame and the positions are used later to
    cut the same stars out of every frame, so a forgotten offset would
    send every later measurement to empty sky.
    """
    stars = find_stars_in_tile(star_image(seed=11), x_offset=1000,
                               y_offset=2000)
    found = _sorted_positions((s['x'], s['y']) for s in stars)
    assert found == _sorted_positions(
        (x + 1000, y + 2000) for x, y in STAR_POSITIONS)


def test_find_stars_in_tile_on_starless_image():
    """Pure noise yields no stars rather than an error or a false detection."""
    rng = np.random.default_rng(5)
    assert find_stars_in_tile(rng.normal(500, 8, (128, 128))) == []


def test_select_reference_stars_limits_number(tmp_path):
    """``n_stars`` caps how many reference stars are chosen.

    Every reference star is fitted on every frame, so the cap is what
    keeps opening the widget quick; without it, all the stars found are
    returned.
    """
    path = write_star_image(tmp_path / "one.fit", seed=3)
    assert len(select_reference_stars(path)) == len(STAR_POSITIONS)
    assert len(select_reference_stars(path, n_stars=2)) == 2


def test_select_reference_stars_spread_over_tiles(tmp_path):
    """Stars are collected from all the tiles, not just the central one.

    The tiles are made small enough here that each real star falls in a
    different one. Stars from across the frame are wanted so that tracking
    or focus problems at the edges show up.
    """
    # tiles small enough that the five of them really are different
    path = write_star_image(tmp_path / "one.fit", seed=3)
    stars = select_reference_stars(path, tile_size=160, corner_fraction=0.5)
    assert _sorted_positions(stars) == _sorted_positions(STAR_POSITIONS)


def test_measure_star_recovers_fwhm_and_flux():
    """A round Gaussian star gives back its true FWHM and flux, and no ellipticity.

    This is the basic check that the fit and the conversion from sigma to
    FWHM are right, since every flag shown to the user is built on them.
    """
    image = star_image(positions=[(60.0, 70.0)], fluxes=[5.0e4],
                       saturated=False, seed=4)
    result = measure_star(image[55:86, 45:76])
    assert result['fwhm'] == pytest.approx(TRUE_FWHM, rel=0.05)
    assert result['flux'] == pytest.approx(5.0e4, rel=0.05)
    assert result['ellipticity'] == pytest.approx(0.0, abs=0.05)


def test_measure_star_recovers_ellipticity():
    """A star twice as wide in y as in x is reported with ellipticity 0.5.

    Trailed stars are one of the faults the widget is meant to reveal.
    The FWHM of an elongated star is the geometric mean of its two
    widths.
    """
    image = star_image(positions=[(60.0, 70.0)], fluxes=[5.0e4],
                       sigma=2.0, sigma_y=4.0, saturated=False, seed=4)
    result = measure_star(image[55:86, 45:76])
    assert result['ellipticity'] == pytest.approx(0.5, abs=0.05)
    expected = FWHM_PER_SIGMA * np.sqrt(2.0 * 4.0)
    assert result['fwhm'] == pytest.approx(expected, rel=0.05)


def test_measure_star_on_blank_cutout_is_nan():
    """A cutout with no star in it gives NaN rather than a made-up width.

    A fit to noise always converges on something; reporting that number
    would put a believable but meaningless FWHM on the tile.
    """
    rng = np.random.default_rng(8)
    result = measure_star(rng.normal(500, 8, (31, 31)))
    assert np.isnan(result['fwhm'])


def test_measure_frame_matches_truth(tmp_path):
    """Each reference star on a frame is measured correctly and its cutout kept.

    The cutouts are what the widget shows under the viewer, so their
    shape and their float32 type (half the memory of float64, which
    matters on the hub) are checked too.
    """
    path = write_star_image(tmp_path / "one.fit", seed=6)
    measured = measure_frame(path, STAR_POSITIONS)
    assert len(measured) == len(STAR_POSITIONS)
    for star, flux in zip(measured, STAR_FLUXES):
        assert star['fwhm'] == pytest.approx(TRUE_FWHM, rel=0.05)
        assert star['flux'] == pytest.approx(flux, rel=0.05)
        assert star['cutout'].shape == (31, 31)
        assert star['cutout'].dtype == np.float32


def test_measure_frame_star_off_the_frame_is_nan(tmp_path):
    """Positions off the frame or on empty sky give NaN without spoiling the rest.

    Frames are not always perfectly registered, so a reference star can
    fall outside one of them. The real stars must still be measured, and
    nothing at all is read for a position that is not on the frame.
    """
    path = write_star_image(tmp_path / "one.fit", seed=6)
    # one position off the frame entirely, one on empty sky
    positions = list(STAR_POSITIONS) + [(400.0, 400.0), (1.0, 1.0)]
    measured = measure_frame(path, positions)
    for star in measured[:len(STAR_POSITIONS)]:
        assert np.isfinite(star['fwhm'])
    for star in measured[len(STAR_POSITIONS):]:
        assert np.isnan(star['fwhm'])
        assert np.isnan(star['flux'])
    # nothing was read for the position that is not on the frame
    assert measured[-2]['cutout'] is None


def test_measure_frame_truncated_cutout_is_nan(tmp_path):
    """A cutout trimmed by the edge to less than ``min_cutout_size`` is not fitted.

    Too little of the star is left for the fit to mean anything, so the
    result is NaN and no cutout is kept.
    """
    path = write_star_image(tmp_path / "one.fit", seed=6)
    measured = measure_frame(path, [(2.0, 2.0)], min_cutout_size=20)
    assert np.isnan(measured[0]['fwhm'])
    assert measured[0]['cutout'] is None


def test_measure_frame_star_near_the_edge(tmp_path):
    """A star whose cutout is only slightly trimmed by the edge is still measured.

    The trimmed cutout is no longer centred on the star, so this guards
    against the fit assuming that the star is in the middle.
    """
    # a star close enough to the edge that its cutout is trimmed
    near_edge = (8.0, 128.0)
    path = write_star_image(tmp_path / "edge.fit", positions=[near_edge],
                            fluxes=[5.0e4], saturated=False, seed=6)
    star = measure_frame(path, [near_edge])[0]
    assert star['cutout'].shape[1] < 31
    assert star['fwhm'] == pytest.approx(TRUE_FWHM, rel=0.05)


def _fake_stars(fwhms, fluxes):
    """Per-star measurements for one frame, as ``measure_frame`` returns them."""
    return [{'fwhm': f, 'ellipticity': 0.1, 'flux': flux}
            for f, flux in zip(fwhms, fluxes)]


def test_summarize_metrics_relative_flux_and_flags():
    """Flux is relative to the typical frame, and only the odd frames are flagged.

    Four ordinary frames, one with stars half as bright and one with
    stars half as wide again. The dim frame must be flagged for flux only
    and the broad one for FWHM only, which is what the user sees in red.
    """
    per_frame = {
        f'frame{i}.fit': _fake_stars([4.0, 4.0, 4.0], [100.0, 200.0, 300.0])
        for i in range(4)
    }
    per_frame['dim.fit'] = _fake_stars([4.0, 4.0, 4.0], [50.0, 100.0, 150.0])
    per_frame['fat.fit'] = _fake_stars([6.0, 6.0, 6.0], [100.0, 200.0, 300.0])

    summary = summarize_metrics(per_frame)

    assert summary['frame0.fit']['rel_flux'] == pytest.approx(1.0)
    assert summary['dim.fit']['rel_flux'] == pytest.approx(0.5)
    assert summary['fat.fit']['fwhm'] == pytest.approx(6.0)

    flagged_flux = {n for n, m in summary.items() if m['flux_flag']}
    flagged_fwhm = {n for n, m in summary.items() if m['fwhm_flag']}
    assert flagged_flux == {'dim.fit'}
    assert flagged_fwhm == {'fat.fit'}


def test_summarize_metrics_identical_frames_are_not_flagged():
    """Frames that differ only by measurement noise are never flagged.

    The FWHM cut is a number of median absolute deviations above the
    median, which would flag ordinary scatter on a very uniform night
    without the floor on the threshold.
    """
    rng = np.random.default_rng(2)
    per_frame = {
        f'frame{i}.fit': _fake_stars(list(rng.normal(4.0, 0.01, 3)),
                                     list(rng.normal(100.0, 0.5, 3)))
        for i in range(6)
    }
    summary = summarize_metrics(per_frame)
    assert not any(m['fwhm_flag'] or m['flux_flag'] for m in summary.values())


def test_summarize_metrics_drops_star_that_fails_often():
    """A star that cannot be measured on most frames is left out everywhere.

    Such a star says more about itself (too faint, too near an edge) than
    about the frames, and keeping it would make the frames' medians
    depend on which stars happened to be measured.
    """
    nan = float('nan')
    per_frame = {}
    for i in range(6):
        # the second star fails on four of the six frames
        fails = i < 4
        per_frame[f'frame{i}.fit'] = _fake_stars(
            [4.0, nan if fails else 4.0, 4.0],
            [100.0, nan if fails else 200.0, 300.0],
        )
    summary = summarize_metrics(per_frame)
    assert summary['frame0.fit']['n_stars'] == 2
    assert len(summary['frame0.fit']['star_fwhm']) == 2
    assert all(np.isfinite(v) for v in summary['frame0.fit']['star_fwhm'])


def test_summarize_metrics_keeps_star_that_fails_twice():
    """A star that fails on only ``max_failures`` frames is kept.

    Where it failed, its entry is ``None`` rather than NaN so that the
    summary can be written to the JSON cache, and the frame's own FWHM
    still comes from the other stars.
    """
    nan = float('nan')
    per_frame = {}
    for i in range(6):
        fails = i < 2
        per_frame[f'frame{i}.fit'] = _fake_stars(
            [4.0, nan if fails else 4.0, 4.0],
            [100.0, nan if fails else 200.0, 300.0],
        )
    summary = summarize_metrics(per_frame)
    assert summary['frame0.fit']['n_stars'] == 3
    # the failed star is None rather than NaN, so the result is JSON safe
    assert summary['frame0.fit']['star_fwhm'][1] is None
    assert np.isfinite(summary['frame0.fit']['fwhm'])


def test_summarize_metrics_frame_with_one_bad_star_is_still_measured():
    """One unmeasurable star on a frame does not spoil that frame's numbers.

    The frame's FWHM and relative flux come from the stars that were
    measured; in particular the relative flux must not drop just because
    one star is missing.
    """
    nan = float('nan')
    per_frame = {f'frame{i}.fit': _fake_stars([4.0, 4.0, 4.0],
                                              [100.0, 200.0, 300.0])
                 for i in range(4)}
    per_frame['frame0.fit'] = _fake_stars([nan, 4.0, 4.0],
                                          [nan, 200.0, 300.0])
    summary = summarize_metrics(per_frame)
    assert summary['frame0.fit']['fwhm'] == pytest.approx(4.0)
    assert summary['frame0.fit']['rel_flux'] == pytest.approx(1.0)


def test_summarize_metrics_of_nothing():
    """No frames gives an empty summary rather than an error.

    This is the case of a directory in which no stars were found.
    """
    assert summarize_metrics({}) == {}


# --- the measurements as the widget uses them, including the cache -------

def test_metrics_measured_and_flagged(star_fits_dir, viewer_factory):
    """The widget finds the stars, measures every frame and flags the bad ones.

    End to end through ``ImageSelect`` on five frames, of which one has
    broad stars and one has dim stars: the metrics are keyed by file name
    and exactly those two frames are flagged, each for the right reason.
    """
    isel = ImageSelect(directory=star_fits_dir,
                       viewer_factory=viewer_factory)
    assert len(isel.star_positions) == len(STAR_POSITIONS)
    assert _sorted_positions(isel.star_positions) == \
        _sorted_positions(STAR_POSITIONS)

    metrics = isel.metrics
    assert set(metrics) == set(isel._im_file_names)
    assert metrics['stars-000.fit']['fwhm'] == pytest.approx(TRUE_FWHM,
                                                             rel=0.05)
    assert metrics[DIM_FRAME]['rel_flux'] == pytest.approx(0.45, rel=0.1)
    assert metrics[BROAD_FRAME]['fwhm'] == pytest.approx(TRUE_FWHM * 1.7,
                                                         rel=0.05)

    assert {n for n, m in metrics.items() if m['fwhm_flag']} == {BROAD_FRAME}
    assert {n for n, m in metrics.items() if m['flux_flag']} == {DIM_FRAME}


def test_metrics_cached_and_reused(star_fits_dir, mocker, viewer_factory):
    """A second widget on the same data reads the cache and measures nothing.

    Measuring is the slow part of opening the notebook, so with the cache
    in ``thumbs/image_quality.json`` neither the star finder nor the
    frame measurement may run again, and the results must be the same.
    """
    isel = ImageSelect(directory=star_fits_dir, viewer_factory=viewer_factory)
    assert isel.quality_path == star_fits_dir / "thumbs" / "image_quality.json"
    assert isel.quality_path.exists()

    measure = mocker.patch("astro_notebooks.image_selector.measure_frame")
    select = mocker.patch(
        "astro_notebooks.image_selector.select_reference_stars")
    again = ImageSelect(directory=star_fits_dir,
                        viewer_factory=viewer_factory)
    assert measure.call_count == 0
    assert select.call_count == 0
    assert again.metrics == isel.metrics
    assert again.star_positions == isel.star_positions


def test_metrics_cache_invalidated_by_new_mtime(star_fits_dir, mocker,
                                               viewer_factory):
    """A frame that has changed on disk makes the whole cache stale.

    The flags compare frames with each other, so every frame is measured
    again, not just the one whose modification time changed.
    """
    ImageSelect(directory=star_fits_dir, viewer_factory=viewer_factory)

    frame = star_fits_dir / "stars-002.fit"
    bumped = frame.stat().st_mtime + 100
    os.utime(frame, (bumped, bumped))

    spy = mocker.patch("astro_notebooks.image_selector.measure_frame",
                       side_effect=measure_frame)
    ImageSelect(directory=star_fits_dir, viewer_factory=viewer_factory)
    # every frame is measured again, not just the one that changed
    assert spy.call_count == 5


def test_metrics_cache_invalidated_by_new_file(star_fits_dir, mocker,
                                              viewer_factory):
    """A frame added to the directory is measured, along with all the others.

    Without this a new frame would be shown with no measurements, and
    the other frames' flags would not take it into account.
    """
    ImageSelect(directory=star_fits_dir, viewer_factory=viewer_factory)
    write_star_image(star_fits_dir / "stars-005.fit", seed=9)

    spy = mocker.patch("astro_notebooks.image_selector.measure_frame",
                       side_effect=measure_frame)
    isel = ImageSelect(directory=star_fits_dir,
                       viewer_factory=viewer_factory)
    assert spy.call_count == 6
    assert "stars-005.fit" in isel.metrics


def test_star_cutouts_cached_as_pngs(star_fits_dir, viewer_factory):
    """A PNG of every reference star on every frame is cached beside the thumbnails.

    The tiles and the details panel show these, and they are named after
    the full file name so that ``x.fit`` and ``x.fits`` do not share
    them.
    """
    isel = ImageSelect(directory=star_fits_dir, viewer_factory=viewer_factory)
    for stem in isel._im_base_names:
        for star in range(len(isel.star_positions)):
            assert (isel.thumbs / f"{stem}_star{star}.png").exists()


def test_no_stars_means_no_metrics(fits_dir, viewer_factory):
    """Frames with no stars still open, with empty metrics and "n/a" on the tiles.

    The cache is written even so, so that the fruitless search for stars
    is not repeated every time the notebook is opened.
    """
    # the plain fixture frames are noise, with nothing to measure
    isel = ImageSelect(directory=fits_dir, viewer_factory=viewer_factory)
    assert isel.star_positions == []
    assert isel.metrics == {}
    assert isel.quality_path.exists()
    for tile in isel._selectors:
        assert tile._quality.value == "FWHM: n/a"

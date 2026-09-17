"""Simple image quality metrics for a night of light frames.

The point of this module is to answer, for each frame of a night, "are the
stars in this frame fatter, rounder or fainter than in the rest of the
night?", so that a student can spot the frames worth leaving out of a
combination.

The work is split into three steps:

1. :func:`select_reference_stars` picks a handful of well behaved stars on
   one frame. Because the frames of a night have already been registered,
   the same stars are in (very nearly) the same place on every frame.
2. :func:`measure_frame` fits a Gaussian to each of those stars on one
   frame and returns its FWHM, ellipticity and flux.
3. :func:`summarize_metrics` compares the frames with each other and flags
   the ones whose stars are broader or fainter than the night's median.

Everything that touches a file reads only small cutouts through
``hdu.section``: a whole 4096 x 4096 frame is never held in memory.
"""

import warnings

import numpy as np

from astropy.io import fits
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.psf import fit_fwhm

__all__ = [
    'find_stars_in_tile',
    'select_reference_stars',
    'measure_star',
    'measure_frame',
    'summarize_metrics',
]


# Sigma of a Gaussian to its full width at half maximum.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Size of the square regions searched for reference stars. Five of these
# cover enough of a 4096 x 4096 frame to find stars without reading, or
# searching, the whole thing.
DEFAULT_TILE_SIZE = 1024

# Size of the square cutout measured around each reference star. Big
# enough for a comfortably sampled star plus some sky, small enough that
# five of them are a rounding error in memory.
DEFAULT_CUTOUT_SIZE = 31

DEFAULT_N_STARS = 5

# Peak values at or above this are treated as saturated. Real detectors
# stop being linear well below full well, and a saturated star is a flat
# topped plateau whose width says nothing about the seeing.
DEFAULT_SATURATION = 5e4


def _image_hdu(hdul):
    """Primary HDU, or the first HDU that actually has data."""
    for hdu in hdul:
        if hdu.header.get('NAXIS', 0) > 0:
            return hdu
    raise ValueError('no image data found in FITS file')


def _tile_bounds(n_x, n_y, tile_size, corner_fraction):
    """Corners of the tiles searched for reference stars.

    One tile is centered on the image and four more sit part of the way
    (``corner_fraction`` of the distance) from the center toward each
    corner. Tiles are clipped to the image, so a frame smaller than
    ``tile_size`` gives one tile covering the whole frame.
    """
    center_x = n_x / 2.0
    center_y = n_y / 2.0
    offset_x = corner_fraction * n_x / 2.0
    offset_y = corner_fraction * n_y / 2.0

    centers = [(center_x, center_y)]
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            centers.append((center_x + sign_x * offset_x,
                            center_y + sign_y * offset_y))

    half = tile_size // 2
    bounds = []
    for c_x, c_y in centers:
        x0 = int(max(0, min(n_x - 1, round(c_x - half))))
        y0 = int(max(0, min(n_y - 1, round(c_y - half))))
        x1 = int(min(n_x, x0 + tile_size))
        y1 = int(min(n_y, y0 + tile_size))
        box = (x0, x1, y0, y1)
        if box not in bounds:
            bounds.append(box)
    return bounds


def find_stars_in_tile(tile, *,
                       n_stars=DEFAULT_N_STARS,
                       saturation=DEFAULT_SATURATION,
                       isolation=30.0,
                       threshold_sigma=50.0,
                       box_size=15,
                       border_width=20,
                       min_fwhm=1.5,
                       max_fwhm=15.0,
                       fit_shape=15,
                       x_offset=0,
                       y_offset=0):
    """Find isolated, unsaturated stars in one square piece of an image.

    Parameters
    ----------
    tile : array
        The image data to search. Non-finite pixels are ignored.
    n_stars : int
        Most stars to return; the brightest usable ones are kept.
    saturation : float
        Peaks at or above this value are rejected. A saturated star is a
        flat topped plateau, which both fits badly and shows up as a crowd
        of adjacent peaks.
    isolation : float
        Reject a peak that has another peak closer than this many pixels,
        so that blends and the edges of saturated plateaus are left out.
    threshold_sigma : float
        Peaks must be this many standard deviations above the background.
    box_size, border_width : int
        Passed to :func:`photutils.detection.find_peaks`.
    min_fwhm, max_fwhm : float
        Candidates whose fitted FWHM falls outside this range are dropped.
    fit_shape : int
        Size of the region fit when checking a candidate's FWHM.
    x_offset, y_offset : int
        Added to the returned positions, so that a tile taken out of a
        larger image can report positions in the full frame.

    Returns
    -------
    list of dict
        One entry per star, brightest first, with keys ``x``, ``y``,
        ``peak`` and ``fwhm``.
    """
    tile = np.asarray(tile, dtype=np.float32)
    bad = ~np.isfinite(tile)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, median, std = sigma_clipped_stats(tile, mask=bad)

    if not np.isfinite(std) or std <= 0:
        return []

    # A tile smaller than twice the border has nothing left to search.
    if min(tile.shape) <= 2 * border_width + box_size:
        border_width = max(0, (min(tile.shape) - box_size) // 4)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        peaks = find_peaks(tile, threshold=median + threshold_sigma * std,
                           box_size=box_size, border_width=border_width,
                           mask=bad)

    if peaks is None or len(peaks) == 0:
        return []

    all_x = np.asarray(peaks['x_peak'], dtype=float)
    all_y = np.asarray(peaks['y_peak'], dtype=float)
    values = np.asarray(peaks['peak_value'], dtype=float)

    # Brightest first, so that the candidates tried are the best measured.
    order = np.argsort(values)[::-1]

    # One background subtracted copy for all of the candidates, rather than
    # one per candidate.
    flat = tile - median

    found = []
    for index in order:
        peak_value = values[index]
        if peak_value >= saturation:
            continue

        x = all_x[index]
        y = all_y[index]

        # Distance to every *other* peak, including the saturated ones.
        distance = np.hypot(all_x - x, all_y - y)
        distance[index] = np.inf
        if distance.min() < isolation:
            continue

        fwhm = _check_fwhm(flat, x, y, fit_shape, min_fwhm, max_fwhm)
        if fwhm is None:
            continue

        found.append({'x': float(x + x_offset), 'y': float(y + y_offset),
                      'peak': float(peak_value), 'fwhm': float(fwhm)})
        if len(found) >= n_stars:
            break

    return found


def _check_fwhm(data, x, y, fit_shape, min_fwhm, max_fwhm):
    """Fitted FWHM of one candidate, or None if it is not usable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fwhm = float(fit_fwhm(data, xypos=[(x, y)],
                                  fit_shape=fit_shape)[0])
    except Exception:
        return None

    if not np.isfinite(fwhm) or fwhm < min_fwhm or fwhm > max_fwhm:
        return None
    return fwhm


def select_reference_stars(fits_path, *,
                           n_stars=DEFAULT_N_STARS,
                           tile_size=DEFAULT_TILE_SIZE,
                           corner_fraction=0.25,
                           isolation=30.0,
                           **kwargs):
    """Pick up to ``n_stars`` reference stars in one FITS image.

    Five square tiles of ``tile_size`` pixels (the center of the frame and
    four part of the way toward the corners) are read one at a time with
    ``hdu.section`` and searched with :func:`find_stars_in_tile`. The stars
    returned are taken from the tiles in turn, so they are spread over the
    frame rather than all coming from the busiest corner.

    Returns a list of ``(x, y)`` positions in full frame coordinates, which
    may be empty if the frame has no usable stars.
    """
    per_tile = []
    with fits.open(fits_path, memmap=True) as hdul:
        hdu = _image_hdu(hdul)
        n_y, n_x = hdu.shape[-2:]
        for x0, x1, y0, y1 in _tile_bounds(n_x, n_y, tile_size,
                                           corner_fraction):
            # hdu.section reads only this tile from disk
            tile = np.asarray(hdu.section[y0:y1, x0:x1], dtype=np.float32)
            per_tile.append(find_stars_in_tile(tile, n_stars=n_stars,
                                               isolation=isolation,
                                               x_offset=x0, y_offset=y0,
                                               **kwargs))

    # Take the best star from each tile, then the second best, and so on,
    # dropping stars that another tile has already contributed (the tiles
    # overlap a little).
    chosen = []
    for rank in range(n_stars):
        for stars in per_tile:
            if rank >= len(stars):
                continue
            star = stars[rank]
            if any(np.hypot(star['x'] - other['x'],
                            star['y'] - other['y']) < isolation
                   for other in chosen):
                continue
            chosen.append(star)
            if len(chosen) >= n_stars:
                return [(s['x'], s['y']) for s in chosen]

    return [(s['x'], s['y']) for s in chosen]


def _nan_metrics():
    return {'fwhm': float('nan'), 'ellipticity': float('nan'),
            'flux': float('nan')}


def measure_star(cutout, *, x_guess=None, y_guess=None, max_shift=3.0,
                 min_sigma=0.4, max_sigma=None, min_amplitude_sigma=5.0):
    """Fit a 2D Gaussian plus a constant background to one star cutout.

    ``x_guess`` and ``y_guess`` are the expected position of the star
    within ``cutout`` (its center by default). The fitted center may move
    up to ``max_shift`` pixels from there, which covers the pixel or two
    that registration leaves behind. A fitted star must stand at least
    ``min_amplitude_sigma`` times the pixel-to-pixel scatter above the
    background, so that a patch of empty sky is not measured as a star.

    Returns a dict with ``fwhm`` (the geometric mean of the x and y full
    widths at half maximum), ``ellipticity`` (0 for a round star) and
    ``flux``. All three are NaN if the fit fails or is unbelievable.
    """
    cutout = np.asarray(cutout, dtype=np.float32)
    if cutout.ndim != 2 or min(cutout.shape) < 5:
        return _nan_metrics()
    if not np.isfinite(cutout).all():
        return _nan_metrics()

    n_y, n_x = cutout.shape
    if x_guess is None:
        x_guess = (n_x - 1) / 2.0
    if y_guess is None:
        y_guess = (n_y - 1) / 2.0
    if max_sigma is None:
        max_sigma = min(n_x, n_y) / 4.0

    background = float(np.median(cutout))
    amplitude = float(cutout.max()) - background
    if amplitude <= 0:
        return _nan_metrics()

    model = (Gaussian2D(amplitude=amplitude,
                        x_mean=x_guess, y_mean=y_guess,
                        x_stddev=2.0, y_stddev=2.0)
             + Const2D(amplitude=background))
    model.x_mean_0.bounds = (x_guess - max_shift, x_guess + max_shift)
    model.y_mean_0.bounds = (y_guess - max_shift, y_guess + max_shift)
    model.x_stddev_0.bounds = (min_sigma, max_sigma)
    model.y_stddev_0.bounds = (min_sigma, max_sigma)

    y_grid, x_grid = np.mgrid[0:n_y, 0:n_x]
    try:
        with warnings.catch_warnings():
            # Fits that wander into a corner of parameter space are common
            # and harmless here; the results are checked below instead.
            warnings.simplefilter('ignore')
            fitter = TRFLSQFitter(calc_uncertainties=False)
            best = fitter(model, x_grid, y_grid, cutout)
    except Exception:
        return _nan_metrics()

    sigma_x = abs(float(best.x_stddev_0.value))
    sigma_y = abs(float(best.y_stddev_0.value))
    amplitude = float(best.amplitude_0.value)

    values = (sigma_x, sigma_y, amplitude)
    if not all(np.isfinite(v) for v in values) or amplitude <= 0:
        return _nan_metrics()
    if min(sigma_x, sigma_y) <= min_sigma or max(sigma_x, sigma_y) >= max_sigma:
        return _nan_metrics()
    # Scatter of the cutout, dominated by its (mostly empty) sky.
    noise = 1.4826 * float(np.median(np.abs(cutout - background)))
    if noise > 0 and amplitude < min_amplitude_sigma * noise:
        return _nan_metrics()

    return {
        'fwhm': float(FWHM_PER_SIGMA * np.sqrt(sigma_x * sigma_y)),
        'ellipticity': float(1.0 - min(sigma_x, sigma_y) / max(sigma_x,
                                                               sigma_y)),
        'flux': float(amplitude * 2.0 * np.pi * sigma_x * sigma_y),
    }


def measure_frame(fits_path, star_positions, *,
                  cutout_size=DEFAULT_CUTOUT_SIZE,
                  min_cutout_size=9,
                  keep_cutouts=True,
                  max_shift=3.0):
    """Measure every reference star on one frame.

    A square cutout of ``cutout_size`` pixels is read around each position
    with ``hdu.section``; nothing larger is ever in memory. A star that
    falls off the frame, or so close to an edge that less than
    ``min_cutout_size`` pixels of it are on the frame, gets NaN metrics.

    Returns one dict per position with the keys of :func:`measure_star`
    plus ``x``, ``y`` and ``cutout`` (a float32 array, or None when there
    was nothing to measure).
    """
    results = []
    half = cutout_size // 2

    with fits.open(fits_path, memmap=True) as hdul:
        hdu = _image_hdu(hdul)
        n_y, n_x = hdu.shape[-2:]

        for x, y in star_positions:
            entry = {'x': float(x), 'y': float(y), 'cutout': None}
            entry.update(_nan_metrics())
            results.append(entry)

            center_x = int(round(x))
            center_y = int(round(y))
            if not (0 <= center_x < n_x and 0 <= center_y < n_y):
                continue

            x0 = max(0, center_x - half)
            y0 = max(0, center_y - half)
            x1 = min(n_x, center_x + half + 1)
            y1 = min(n_y, center_y + half + 1)
            if (x1 - x0) < min_cutout_size or (y1 - y0) < min_cutout_size:
                continue

            # hdu.section reads only this cutout from disk
            cutout = np.asarray(hdu.section[y0:y1, x0:x1], dtype=np.float32)
            entry.update(measure_star(cutout,
                                      x_guess=x - x0, y_guess=y - y0,
                                      max_shift=max_shift))
            if keep_cutouts:
                entry['cutout'] = cutout

    return results


def _finite_or_none(value):
    """A plain float, or None when the value is NaN (JSON has no NaN)."""
    value = float(value)
    return value if np.isfinite(value) else None


def _nan_median(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float('nan')
    return float(np.median(values))


def summarize_metrics(per_frame, *,
                      max_failures=2,
                      n_mad=3.0,
                      min_fwhm_ratio=1.15,
                      min_rel_flux=0.7):
    """Turn per-star measurements into one verdict per frame.

    Parameters
    ----------
    per_frame : dict
        Maps file name to the list of per-star dicts returned by
        :func:`measure_frame`, in the same star order for every frame.
    max_failures : int
        A star that could not be measured on more than this many frames is
        left out of every frame's summary.
    n_mad : float
        A frame is flagged when its FWHM is more than this many (scaled)
        median absolute deviations above the night's median FWHM.
    min_fwhm_ratio : float
        ... and also at least this much larger than the median, so that a
        night of nearly identical frames does not flag its own noise.
    min_rel_flux : float
        A frame is flagged when its stars are fainter than this fraction of
        their median brightness over the night.

    Returns a dict keyed by file name whose values are JSON friendly:
    ``fwhm``, ``ellipticity`` and ``rel_flux`` (floats, or None when there
    is nothing to report), the per-star lists ``star_fwhm`` and
    ``star_rel_flux``, the booleans ``fwhm_flag`` and ``flux_flag``, and
    ``n_stars``.
    """
    names = list(per_frame)
    if not names:
        return {}

    n_stars = max((len(per_frame[name]) for name in names), default=0)

    def star_value(name, index, key):
        stars = per_frame[name]
        if index >= len(stars):
            return float('nan')
        return float(stars[index].get(key, float('nan')))

    # Stars that fail too often say more about the star than about the
    # frames, so drop them before comparing frames with each other.
    keep = []
    for index in range(n_stars):
        failures = sum(not np.isfinite(star_value(name, index, 'flux'))
                       or not np.isfinite(star_value(name, index, 'fwhm'))
                       for name in names)
        if failures <= max_failures:
            keep.append(index)

    median_flux = {
        index: _nan_median([star_value(name, index, 'flux')
                            for name in names])
        for index in keep
    }

    summary = {}
    for name in names:
        star_fwhm = [star_value(name, index, 'fwhm') for index in keep]
        star_ellipticity = [star_value(name, index, 'ellipticity')
                            for index in keep]
        star_rel_flux = []
        for index in keep:
            reference = median_flux[index]
            flux = star_value(name, index, 'flux')
            if np.isfinite(reference) and reference > 0:
                star_rel_flux.append(flux / reference)
            else:
                star_rel_flux.append(float('nan'))

        summary[name] = {
            'fwhm': _nan_median(star_fwhm),
            'ellipticity': _nan_median(star_ellipticity),
            'rel_flux': _nan_median(star_rel_flux),
            'star_fwhm': star_fwhm,
            'star_rel_flux': star_rel_flux,
            'n_stars': len(keep),
        }

    fwhms = [summary[name]['fwhm'] for name in names]
    median_fwhm = _nan_median(fwhms)
    finite_fwhm = [f for f in fwhms if np.isfinite(f)]
    if finite_fwhm and np.isfinite(median_fwhm):
        mad = 1.4826 * float(np.median([abs(f - median_fwhm)
                                        for f in finite_fwhm]))
        fwhm_limit = max(median_fwhm + n_mad * mad,
                         min_fwhm_ratio * median_fwhm)
    else:
        fwhm_limit = float('inf')

    for name in names:
        entry = summary[name]
        fwhm = entry['fwhm']
        rel_flux = entry['rel_flux']
        entry['fwhm_flag'] = bool(np.isfinite(fwhm) and fwhm > fwhm_limit)
        entry['flux_flag'] = bool(np.isfinite(rel_flux)
                                  and rel_flux < min_rel_flux)
        # JSON has no NaN, so a missing number is None from here on.
        entry['fwhm'] = _finite_or_none(fwhm)
        entry['ellipticity'] = _finite_or_none(entry['ellipticity'])
        entry['rel_flux'] = _finite_or_none(rel_flux)
        entry['star_fwhm'] = [_finite_or_none(v) for v in entry['star_fwhm']]
        entry['star_rel_flux'] = [_finite_or_none(v)
                                  for v in entry['star_rel_flux']]

    return summary

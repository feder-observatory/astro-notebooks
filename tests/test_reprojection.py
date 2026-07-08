import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.wcs import WCS
from reproject import reproject_interp

from astro_notebooks.reprojection import reproject_to_reference

SHAPE = (100, 100)
PIXEL_SCALE = 7.0 / 3600  # degrees / pixel


def _make_wcs(crpix_offset=(0.0, 0.0)):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [30.0, 45.0]
    wcs.wcs.crpix = [50.0 + crpix_offset[0], 50.0 + crpix_offset[1]]
    wcs.wcs.cdelt = [-PIXEL_SCALE, PIXEL_SCALE]
    return wcs


def _make_ccd(crpix_offset=(0.0, 0.0), seed=42):
    rng = np.random.default_rng(seed)
    data = rng.uniform(100.0, 2000.0, SHAPE)
    return CCDData(data, wcs=_make_wcs(crpix_offset=crpix_offset), unit="adu",
                   meta={"exposure": 30.0})


@pytest.fixture
def ccd_offset():
    """Image whose WCS is offset a few pixels from the reference WCS."""
    return _make_ccd(crpix_offset=(3.2, -2.7))


@pytest.fixture
def wcs_ref():
    return _make_wcs()


def test_output_dtype_float32(ccd_offset, wcs_ref):
    result = reproject_to_reference(ccd_offset, wcs_ref, SHAPE)
    assert result.data.dtype == np.float32
    assert isinstance(result, CCDData)
    # BITPIX is what actually lands on disk; -32 is FITS for 32-bit float.
    assert result.to_hdu()[0].header["BITPIX"] == -32


def test_output_shape_matches_shape_out(ccd_offset, wcs_ref):
    result = reproject_to_reference(ccd_offset, wcs_ref, SHAPE)
    assert result.data.shape == SHAPE

    shape_alt = (120, 110)
    result_alt = reproject_to_reference(ccd_offset, wcs_ref, shape_alt)
    assert result_alt.data.shape == shape_alt


def test_result_wcs_is_reference_wcs(ccd_offset, wcs_ref):
    result = reproject_to_reference(ccd_offset, wcs_ref, SHAPE)
    assert np.allclose(result.wcs.wcs.crpix, wcs_ref.wcs.crpix)
    assert np.allclose(result.wcs.wcs.crval, wcs_ref.wcs.crval)
    assert np.allclose(result.wcs.wcs.cdelt, wcs_ref.wcs.cdelt)
    assert list(result.wcs.wcs.ctype) == list(wcs_ref.wcs.ctype)


def test_unit_and_meta_preserved(ccd_offset, wcs_ref):
    result = reproject_to_reference(ccd_offset, wcs_ref, SHAPE)
    assert result.unit == ccd_offset.unit
    assert result.meta["exposure"] == 30.0


def test_median_subtraction(ccd_offset, wcs_ref):
    direct, _ = reproject_interp(ccd_offset, wcs_ref, shape_out=SHAPE, order=1,
                                 roundtrip_coords=False)
    median = np.nanmedian(ccd_offset.data)
    with_sub = reproject_to_reference(ccd_offset, wcs_ref, SHAPE, subtract_median=True)
    without_sub = reproject_to_reference(ccd_offset, wcs_ref, SHAPE, subtract_median=False)

    assert np.allclose(without_sub.data, direct.astype(np.float32), rtol=1e-6, equal_nan=True)
    assert np.allclose(with_sub.data, (direct - median).astype(np.float32), rtol=1e-6, atol=1e-3, equal_nan=True)


def test_identity_reprojection_preserves_values(wcs_ref):
    ccd = _make_ccd(crpix_offset=(0.0, 0.0))
    result = reproject_to_reference(ccd, wcs_ref, SHAPE, subtract_median=False)
    finite = np.isfinite(result.data)
    assert finite.sum() >= 0.99 * result.data.size
    expected = ccd.data.astype(np.float32)[finite]
    np.testing.assert_allclose(result.data[finite], expected, rtol=1e-5)
    # Also bound the worst-case absolute error: unlike the other tests, this
    # compares against the original input, so it is the one ground-truth
    # anchor that would catch a systematic shift/scale in the reproject path.
    max_abs_diff = np.max(np.abs(result.data[finite] - expected))
    assert max_abs_diff < 1e-2


def test_nan_outside_footprint(ccd_offset, wcs_ref):
    result = reproject_to_reference(ccd_offset, wcs_ref, SHAPE)
    assert np.isnan(result.data).any()
    assert np.isfinite(result.data).any()


def test_block_size_matches_unchunked(ccd_offset, wcs_ref):
    chunked = reproject_to_reference(ccd_offset, wcs_ref, SHAPE, block_size=(32, 32))
    unchunked = reproject_to_reference(ccd_offset, wcs_ref, SHAPE, block_size=None)
    assert chunked.data.dtype == np.float32
    assert np.allclose(chunked.data, unchunked.data, rtol=1e-5, atol=1e-3, equal_nan=True)

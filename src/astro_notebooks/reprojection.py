import numpy as np
from astropy.nddata import CCDData
from reproject import reproject_interp


def reproject_to_reference(ccd, wcs_ref, shape_out, order=1,
                           subtract_median=True, block_size=(512, 512)):
    """
    Reproject an image onto a reference WCS, returning a float32 result.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The image to reproject.
    wcs_ref : `~astropy.wcs.WCS`
        WCS of the reference image; the result is aligned to this.
    shape_out : tuple of int
        Shape of the output image, usually the reference image's shape.
    order : int, optional
        Interpolation order passed to `~reproject.reproject_interp`.
    subtract_median : bool, optional
        If ``True``, subtract the median of the *input* image from the
        result, removing most of the sky background.
    block_size : tuple of int or None, optional
        Chunk size for the reprojection, which limits peak memory use.
        ``None`` reprojects the whole image at once.

    Returns
    -------
    `~astropy.nddata.CCDData`
        The reprojected image, cast to float32 (plenty of precision for
        camera counts and half the file size of float64), with
        ``wcs_ref`` as its WCS and the input's unit and metadata.
        Pixels outside the input's footprint are NaN.
    """
    if subtract_median:
        median = np.nanmedian(ccd.data)
    else:
        median = 0.0

    # The round-trip coordinate check roughly doubles the cost of the
    # reprojection and is unnecessary for well-behaved shifted images.
    new_data, _ = reproject_interp(ccd, wcs_ref, shape_out=shape_out,
                                   order=order, roundtrip_coords=False,
                                   block_size=block_size)

    # Cast to float32 first, then subtract in place, so we never allocate a
    # full-frame float64 temporary (reproject_interp returns float64) and skip
    # the subtraction pass entirely when it is disabled.
    result = new_data.astype(np.float32)
    if subtract_median:
        result -= median
    return CCDData(result, wcs=wcs_ref, unit=ccd.unit, meta=ccd.meta)

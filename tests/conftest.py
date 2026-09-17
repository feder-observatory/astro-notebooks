import ipywidgets as ipw
import numpy as np
import pytest
from astropy.io import fits

N_IMAGES = 5
IMAGE_SHAPE = (128, 128)


@pytest.fixture
def fits_dir(tmp_path, monkeypatch):
    """Directory of small synthetic FITS images.

    ImageSelect caches its thumbnails in ``<data_dir>/thumbs``, so nothing
    should be written to the current working directory; cwd is still moved
    to tmp_path so that a test can check that.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(N_IMAGES):
        data = rng.uniform(100.0, 1000.0, size=IMAGE_SHAPE)
        # a few NaNs and one very large value to exercise cleaning/clamping
        data[0:2, 0:2] = np.nan
        data[5, 5] = 2e5
        hdu = fits.PrimaryHDU(data)
        hdu.header["IMAGETYP"] = "LIGHT"
        hdu.writeto(data_dir / f"image-{i:03d}.fit")
    monkeypatch.chdir(tmp_path)
    return data_dir


# Synthetic star fields used to exercise the quality metrics. The frames
# are registered, so the stars sit at the same place on every frame.
STAR_IMAGE_SHAPE = (256, 256)
STAR_SKY = 500.0
STAR_NOISE = 8.0
STAR_SIGMA = 2.5
STAR_FLUXES = (5.0e4, 4.0e4, 3.0e4)
STAR_POSITIONS = ((60.0, 70.0), (190.0, 80.0), (100.0, 190.0))
# A saturated star, which the star finder must refuse to use.
SATURATED_POSITION = (200.0, 200.0)
SATURATION_LEVEL = 65535.0

# Which frame of star_fits_dir is which.
BROAD_FRAME = "stars-003.fit"
DIM_FRAME = "stars-004.fit"


def star_image(shape=STAR_IMAGE_SHAPE, positions=STAR_POSITIONS,
               fluxes=STAR_FLUXES, sigma=STAR_SIGMA, sigma_y=None,
               sky=STAR_SKY, noise=STAR_NOISE, seed=0, saturated=True):
    """A synthetic frame: flat sky, Gaussian noise and Gaussian stars."""
    rng = np.random.default_rng(seed)
    image = rng.normal(sky, noise, shape).astype(np.float32)
    y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1]]

    sigma_y = sigma if sigma_y is None else sigma_y
    stars = list(zip(positions, fluxes))
    if saturated:
        # Bright enough that the middle of it runs into the "full well"
        # clip below, which is what makes it a flat topped plateau.
        stars.append((SATURATED_POSITION, 4.0e7))

    for (x, y), flux in stars:
        amplitude = flux / (2 * np.pi * sigma * sigma_y)
        image += amplitude * np.exp(
            -((x_grid - x) ** 2 / (2 * sigma ** 2)
              + (y_grid - y) ** 2 / (2 * sigma_y ** 2))
        )

    return np.minimum(image, SATURATION_LEVEL)


def write_star_image(path, **kwargs):
    """Write :func:`star_image` to ``path`` as a FITS file."""
    hdu = fits.PrimaryHDU(star_image(**kwargs))
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture
def star_fits_dir(tmp_path, monkeypatch):
    """Five registered frames of the same synthetic star field.

    Frames 0 to 2 are ordinary, frame 3 has visibly broader stars and
    frame 4 has stars half as bright; nothing else differs.
    """
    data_dir = tmp_path / "stars"
    data_dir.mkdir()
    for i in range(5):
        kwargs = dict(seed=i)
        if f"stars-{i:03d}.fit" == BROAD_FRAME:
            kwargs["sigma"] = STAR_SIGMA * 1.7
        if f"stars-{i:03d}.fit" == DIM_FRAME:
            kwargs["fluxes"] = tuple(0.45 * f for f in STAR_FLUXES)
        write_star_image(data_dir / f"stars-{i:03d}.fit", **kwargs)
    monkeypatch.chdir(tmp_path)
    return data_dir


@pytest.fixture
def mock_viewer(mocker):
    """Stand-in for the image viewer, which needs a browser to be real.

    It has to be a real widget, because ImageSelect puts it in a VBox, so
    it is an empty box with a mock in place of ``load_image``.
    """
    viewer = ipw.VBox()
    viewer.load_image = mocker.MagicMock()
    return viewer


@pytest.fixture
def viewer_factory(mock_viewer):
    """Hand ImageSelect the mock viewer instead of a real one."""
    return lambda: mock_viewer

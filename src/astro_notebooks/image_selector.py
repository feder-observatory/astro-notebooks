from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ipywidgets as ipw
import numpy as np

from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.visualization import simple_norm
from ccdproc import ImageFileCollection
from IPython.display import display
from PIL import Image
from reducer.image_browser import banded_block_reduce

try:
    from stellarphot.gui.custom_widgets import Spinner
except Exception:
    # stellarphot's GUI extras may be missing or incompatible; fall back to
    # a message-only stand-in with the same start/stop interface.
    Spinner = None


class _MessageSpinner(ipw.VBox):
    """Fallback for stellarphot's Spinner when it cannot be imported."""

    def __init__(self, *args, message="", **kwargs):
        super().__init__(*args, **kwargs)
        self._message = ipw.HTML(message)
        self.children = [self._message]
        self.layout.display = "none"

    def start(self):
        self.layout.display = "flex"

    def stop(self):
        self.layout.display = "none"


def _clamp(data):
    """Clamp very bright pixels, as a float32 copy of the input.

    The input is never modified. The result is float32, which keeps the
    memory used by a band of a big image small. This is used both on a
    whole frame and, as the ``preprocess`` argument of
    :func:`~reducer.image_browser.banded_block_reduce`, on one band at a
    time.
    """
    # float32 copy: small, short lived, and never a view on the caller's data
    scaled_data = np.asarray(data).astype(np.float32)
    scaled_data[scaled_data > 1e5] = 1e5
    return scaled_data


def _normalize(scaled_data, min_percent=20, max_percent=99.5):
    """Percentile-scale an already downsampled image and remove NaNs."""
    norm = simple_norm(scaled_data,
                       min_percent=min_percent,
                       max_percent=max_percent,
                       clip=True)

    # Replace all NaNs with 0
    normed_data = norm(scaled_data)
    normed_data[np.isnan(normed_data)] = 0

    return normed_data


def _scale_and_downsample(data, downsample=8,
                         min_percent=20,
                         max_percent=99.5):
    """Clamp, downsample and normalize an in-memory image.

    This is the whole-frame version of :func:`_thumbnail_data`; both clamp
    with :func:`_clamp` and normalize with :func:`_normalize`, so they
    return identical arrays for the same image.
    """
    scaled_data = _clamp(data)
    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
    return _normalize(scaled_data,
                      min_percent=min_percent,
                      max_percent=max_percent)


def _image_hdu(hdul):
    """Primary HDU, or the first HDU that actually has data."""
    for hdu in hdul:
        if hdu.header.get('NAXIS', 0) > 0:
            return hdu
    raise ValueError('no image data found in FITS file')


def _thumbnail_data(fits_path, downsample=8,
                    min_percent=20,
                    max_percent=99.5,
                    band_rows=None):
    """Downsampled, normalized image data read a band of rows at a time.

    The banded read itself is
    :func:`~reducer.image_browser.banded_block_reduce`, which reads only a
    band of rows at a time and clamps each band as it is read, so a full
    frame (and in particular a full float64 copy of one) is never made.
    Its result is identical to downsampling the whole frame, so this
    returns the same array as :func:`_scale_and_downsample` does for the
    same image.
    """
    with fits.open(fits_path, memmap=True) as hdul:
        small = banded_block_reduce(_image_hdu(hdul), downsample,
                                    band_rows=band_rows,
                                    preprocess=_clamp)

    return _normalize(small,
                      min_percent=min_percent,
                      max_percent=max_percent)


def _make_one_thumbnail(fits_path, dest_path, downsample):
    """Make a single uint8 grayscale PNG thumbnail for a FITS image.

    Runs in a worker thread; the FITS read, numpy work and PNG encode all
    release the GIL for most of their run time.
    """
    scaled = _thumbnail_data(fits_path, downsample=downsample)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(dest_path)


class ImageWithSelector(ipw.VBox):
    # value = tr.Bool(default_value=True).tag(sync=True)

    def __init__(self, image_png, *args, width="200px", fname="", **kwargs):
        super().__init__(*args, **kwargs)
        self._fname = fname
        img_layout = dict(
            object_fit='contain',
            width='100%'
        )
        self.image_display = ipw.Image(
            value=image_png,
            format='png',
            layout=img_layout
        )
        self._selector = ipw.Checkbox(
            description='Use image',
            value=True
        )
        self._valid_mark = ipw.Valid(
            description='',
            value=True
        )

        self._name = ipw.HTML(value=fname)

        ipw.link((self._selector, 'value'), (self._valid_mark, 'value'))
        # ipw.link((self, 'value'), (self._selector, 'value'))

        self.select_box = ipw.HBox(children=[self._selector, self._valid_mark])
        self.mobox = ipw.VBox(children=[self._name, self.select_box])
        self.children = [self.image_display, self.mobox]
        self.layout.width = width


class ImageSelect(ipw.VBox):
    # A small pool is both faster and roughly half the peak memory of the
    # default (one thread per CPU) pool, which matters on a JupyterHub with
    # a per-user memory cap.
    DEFAULT_MAX_WORKERS = 4

    def __init__(self, *args, directory=".", downsample=8,
                 max_workers=DEFAULT_MAX_WORKERS, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path(directory)
        self._downsample = downsample
        self._max_workers = max_workers
        self._collection = ImageFileCollection(self.path)
        self._move_rejects = ipw.Button(description='Move rejects')

        # Cache thumbnails next to the data rather than in the current
        # working directory, so that a cache is never reused for a
        # different directory of images.
        self.thumbs = self.path / 'thumbs'
        self.make_thumbnails(thumb_dir=self.thumbs)
        self.make_selectors(thumb_dir=self.thumbs)
        self.n_cols = 4
        gs = self._make_grid()
        self.children = [gs, self._move_rejects]
        # self.layout.max_height = "400px"
        # self.layout.overflow = "scroll hidden"
        self._move_rejects.on_click(self._move_rejects_clicked)

    def make_thumbnails(self, thumb_dir=None):
        self._images = []
        thumby = Path(thumb_dir) if thumb_dir is not None else self.thumbs
        thumby.mkdir(parents=True, exist_ok=True)
        self._collection.refresh()
        # Full file names (with extension) and, in the same order, the stems
        # used to name the thumbnail PNGs.
        self._im_file_names = []
        self._im_base_names = []
        todo = []
        for fname in self._collection.files_filtered(include_path=True):
            source = Path(fname)
            base = source.stem
            self._im_file_names.append(source.name)
            self._im_base_names.append(base)
            dest_path = thumby / (base + '.png')
            if dest_path.exists():
                continue
            todo.append((source, dest_path))

        if not todo:
            return

        spinner_cls = Spinner if Spinner is not None else _MessageSpinner
        spinner = spinner_cls(message="Generating image thumbnails...")
        progress = ipw.IntProgress(
            value=0, min=0, max=len(todo), description="Thumbnails"
        )
        progress_box = ipw.VBox(children=[spinner, progress])
        # Display right away so the user sees activity while the rest of
        # the widget is still being constructed.
        display(progress_box)
        spinner.start()

        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [
                    executor.submit(_make_one_thumbnail, src, dest, self._downsample)
                    for src, dest in todo
                ]
                for future in as_completed(futures):
                    future.result()
                    progress.value += 1
        finally:
            spinner.stop()
            progress_box.layout.display = "none"

    def make_selectors(self, thumb_dir=None):
        thumby = Path(thumb_dir) if thumb_dir is not None else self.thumbs
        pngs = list(thumby.glob('*.png'))

        for thumb in pngs:
            if thumb.stem not in self._im_base_names:
                thumb.unlink()
        pngs = list(thumby.glob('*.png'))

        png_dict = {p.stem: p for p in pngs}

        kiddos = {}
        for ims in self._im_base_names:
            image_png = png_dict[ims].read_bytes()
            iws = ImageWithSelector(image_png, fname=ims)
            kiddos[ims] = iws

        self._selectors = [kiddos[ims] for ims in self._im_base_names]

    def _move_rejects_clicked(self, _):
        reject_land = Path(self.path / 'rejects')
        for f, selector in zip(self._im_file_names, self._selectors):
            if not selector._valid_mark.value:
                reject_land.mkdir(exist_ok=True)
                source = self.path / f
                dest = reject_land / source.name
                source.rename(dest)

        self.make_thumbnails()
        self.make_selectors()
        gs = self._make_grid()
        self.children = [gs, self._move_rejects]

    def _make_grid(self):
        rows = len(self._selectors) // self.n_cols
        if len(self._selectors) % self.n_cols:
            rows += 1
        gs = ipw.GridspecLayout(rows, self.n_cols)
        for i in range(self.n_cols):
            for j in range(rows):
                index = i + j * self.n_cols
                if index >= len(self._selectors):
                    break
                gs[j, i] = self._selectors[index]

        return gs

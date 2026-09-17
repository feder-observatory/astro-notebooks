import json
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import ipywidgets as ipw
import numpy as np

from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.visualization import simple_norm
from ccdproc import ImageFileCollection
from IPython.display import display
from PIL import Image

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


# Name of the file, written beside the data, that remembers which frames
# the user has checked. It maps file name (with extension) -> bool.
SELECTION_FILE_NAME = 'image_selection.json'


def write_selection_manifest(isel, destination, run_label):
    """Record which frames went into a combination.

    Writes ``<destination>/<run_label>_manifest.json`` describing the
    selection held by ``isel`` (an :class:`ImageSelect`) at the moment of
    the call, and returns the path written.

    The manifest holds the run label, an ISO timestamp, the data directory
    the frames came from, and the ``included`` and ``excluded`` file names
    (relative to that data directory).
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    included = list(isel.selected_files)
    included_set = set(included)
    excluded = [f for f in isel._im_file_names if f not in included_set]

    manifest = {
        'run_label': run_label,
        'timestamp': datetime.now().astimezone().isoformat(),
        'data_dir': str(isel.path),
        'included': included,
        'excluded': excluded,
    }

    manifest_path = destination / f'{run_label}_manifest.json'
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def _atomic_write_json(path, contents):
    """Write ``contents`` as JSON to ``path`` without a partial file.

    The JSON goes to a temporary file in the same directory, which is then
    renamed over ``path``, so a reader never sees a half-written file.
    """
    path = Path(path)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent,
                                        prefix=path.name + '.',
                                        suffix='.tmp')
    try:
        with os.fdopen(handle, 'w') as f:
            json.dump(contents, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# Rough number of image rows to read from disk at a time. The real band
# height is rounded down to a multiple of the downsampling factor so that
# block_reduce only ever trims a partial block in the very last band, which
# is exactly what it would do for a whole-frame call.
_TARGET_BAND_ROWS = 256


def _band_height(downsample, target_rows=_TARGET_BAND_ROWS):
    """Number of rows to read at a time, a multiple of ``downsample``."""
    if downsample <= 1:
        return max(int(target_rows), 1)
    return max(downsample, (int(target_rows) // downsample) * downsample)


def _clamp_and_reduce(data, downsample):
    """Clamp bright pixels and block-average one band (or a whole frame).

    The input is never modified. The result is float32, which keeps the
    memory used by the band small; ``block_reduce`` trims any partial
    block at the end of each axis.
    """
    # float32 copy: small, short lived, and never a view on the caller's data
    scaled_data = np.asarray(data).astype(np.float32)
    scaled_data[scaled_data > 1e5] = 1e5
    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
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

    This is the whole-frame version of :func:`_thumbnail_data`; both share
    :func:`_clamp_and_reduce` and :func:`_normalize`, so they return
    identical arrays for the same image.
    """
    return _normalize(_clamp_and_reduce(data, downsample),
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

    Only ``band_rows`` rows of the image are in memory at once, so a full
    frame (and in particular a full float64 copy of one) is never made.
    """
    band = _band_height(downsample, band_rows or _TARGET_BAND_ROWS)

    reduced = []
    with fits.open(fits_path, memmap=True) as hdul:
        hdu = _image_hdu(hdul)
        n_rows = hdu.shape[0]
        for row0 in range(0, n_rows, band):
            row1 = min(row0 + band, n_rows)
            if downsample > 1 and (row1 - row0) < downsample:
                # block_reduce would trim these rows away anyway
                break
            # hdu.section reads only these rows from disk
            reduced.append(_clamp_and_reduce(hdu.section[row0:row1, :],
                                             downsample))

    small = np.concatenate(reduced, axis=0)
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

        # Cache thumbnails next to the data rather than in the current
        # working directory, so that a cache is never reused for a
        # different directory of images.
        self.thumbs = self.path / 'thumbs'
        self.make_thumbnails(thumb_dir=self.thumbs)
        self.make_selectors(thumb_dir=self.thumbs)
        # Restore first, then start watching the checkboxes, so that
        # restoring does not itself trigger a save.
        self._restore_selection()
        for selector in self._selectors:
            selector._selector.observe(self._selection_changed, names='value')
        # Save once now so that the selection file always exists, and so
        # that entries for files that have disappeared are dropped.
        self.save_selection()
        self.n_cols = 4
        gs = self._make_grid()
        self.children = [gs]
        # self.layout.max_height = "400px"
        # self.layout.overflow = "scroll hidden"

    @property
    def selection_path(self):
        """Path of the JSON file that remembers the current selection."""
        return self.path / SELECTION_FILE_NAME

    @property
    def selected_files(self):
        """Names of the checked files, in collection order.

        These are file names *relative to the data directory*, with the
        extension, so they can be handed straight to
        ``ImageFileCollection(location=data_dir, filenames=...)``.
        """
        return [fname
                for fname, selector in zip(self._im_file_names,
                                           self._selectors)
                if selector._selector.value]

    @property
    def selected_paths(self):
        """The checked files as full :class:`~pathlib.Path` objects."""
        return [self.path / fname for fname in self.selected_files]

    def _selection_changed(self, _change):
        self.save_selection()

    def save_selection(self):
        """Write the current checkbox state beside the data.

        Every file currently in the collection gets an entry, so entries
        for files that no longer exist are dropped.
        """
        selection = {fname: bool(selector._selector.value)
                     for fname, selector in zip(self._im_file_names,
                                                self._selectors)}
        _atomic_write_json(self.selection_path, selection)

    def _read_selection(self):
        """Saved selection, or an empty mapping if there is none to read."""
        try:
            with open(self.selection_path) as f:
                saved = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            warnings.warn(
                f'Ignoring unreadable image selection file '
                f'{self.selection_path}; starting with all images included.',
                stacklevel=2)
            return {}

        if not isinstance(saved, dict):
            warnings.warn(
                f'Ignoring image selection file {self.selection_path}, '
                f'which does not contain a mapping of file name to True or '
                f'False; starting with all images included.',
                stacklevel=2)
            return {}

        return saved

    def _restore_selection(self):
        """Set the checkboxes from the saved selection, if there is one.

        Files with no saved entry default to checked (included).
        """
        saved = self._read_selection()
        if not saved:
            return
        for fname, selector in zip(self._im_file_names, self._selectors):
            selector._selector.value = bool(saved.get(fname, True))

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

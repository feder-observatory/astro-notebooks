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

    included = isel.selected_files
    excluded = [f for f in isel._im_file_names if f not in included]

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


def _clamp(data):
    """
    Clamp very bright pixels in a float32 copy of the input.

    Parameters
    ----------
    data : array-like
        Image data, either a whole frame or one band of rows.

    Returns
    -------
    numpy.ndarray
        Copy of ``data`` as float32, with values above 1e5 set to 1e5. NaNs
        pass through unchanged. The input is never modified.

    Notes
    -----
    float32 keeps the memory used by a band of a big image small. This is
    called on a whole frame by `_scale_and_downsample` and, as the
    ``preprocess`` argument of `reducer.image_browser.banded_block_reduce`,
    on one band at a time by `_thumbnail_data`.
    """
    # float32 copy: small, short lived, and never a view on the caller's data
    scaled_data = np.asarray(data).astype(np.float32)
    scaled_data[scaled_data > 1e5] = 1e5
    return scaled_data


def _normalize(scaled_data, min_percent=20, max_percent=99.5):
    """
    Percentile-scale an already downsampled image to the range 0 to 1.

    Parameters
    ----------
    scaled_data : numpy.ndarray
        Clamped, downsampled image data.
    min_percent, max_percent : float, optional
        Percentiles of ``scaled_data`` that map to 0 and 1. Values outside
        them are clipped.

    Returns
    -------
    numpy.ma.MaskedArray
        Same shape as ``scaled_data``, values in [0, 1], NaNs replaced
        with 0. Nothing is masked; the masked array type is what
        `astropy.visualization.ImageNormalize` returns.
    """
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
    """
    Clamp, downsample and normalize an in-memory image.

    Parameters
    ----------
    data : array-like
        Full-resolution image data.
    downsample : int, optional
        Factor by which each axis is reduced with
        `astropy.nddata.block_reduce`. No reduction is done if this is 1.
    min_percent, max_percent : float, optional
        Percentiles of the downsampled image that map to 0 and 1.

    Returns
    -------
    numpy.ma.MaskedArray
        Downsampled image with values in [0, 1] and NaNs replaced with 0.

    Notes
    -----
    This is the whole-frame version of `_thumbnail_data`; both clamp with
    `_clamp` and normalize with `_normalize`, so they return identical
    arrays for the same image.
    """
    scaled_data = _clamp(data)
    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
    return _normalize(scaled_data,
                      min_percent=min_percent,
                      max_percent=max_percent)


def _image_hdu(hdul):
    """
    Find the HDU that holds the image in an open FITS file.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The open FITS file.

    Returns
    -------
    HDU object from `astropy.io.fits`
        The primary HDU if it has data, otherwise the first HDU that does.

    Raises
    ------
    ValueError
        If no HDU in the file has data.
    """
    for hdu in hdul:
        if hdu.header.get('NAXIS', 0) > 0:
            return hdu
    raise ValueError('no image data found in FITS file')


def _thumbnail_data(fits_path, downsample=8,
                    min_percent=20,
                    max_percent=99.5,
                    band_rows=None):
    """
    Downsampled, normalized image data read a band of rows at a time.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        FITS file to read. It is opened memory-mapped.
    downsample : int, optional
        Factor by which each axis is reduced.
    min_percent, max_percent : float, optional
        Percentiles of the downsampled image that map to 0 and 1.
    band_rows : int or None, optional
        Approximate number of image rows to read at a time. It is passed on
        to `reducer.image_browser.banded_block_reduce`, which rounds it to
        a whole number of blocks and uses roughly 256 rows if this is
        ``None``.

    Returns
    -------
    numpy.ma.MaskedArray
        Downsampled image with values in [0, 1] and NaNs replaced with 0.

    Notes
    -----
    The banded read itself is
    `reducer.image_browser.banded_block_reduce`, which reads only a band of
    rows at a time and clamps each band as it is read, so a full frame (and
    in particular a full float64 copy of one) is never made. Its result is
    identical to downsampling the whole frame, so this returns the same
    array as `_scale_and_downsample` does for the same image.
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
    """
    Grid of image thumbnails, each with a checkbox for keeping the image.

    The checkbox state is written to ``<directory>/image_selection.json``
    every time a checkbox changes and is restored from that file when the
    widget is made again for the same directory. No files are moved.

    Parameters
    ----------
    *args
        Passed on to `ipywidgets.VBox`.
    directory : str or pathlib.Path, optional
        Directory containing the FITS images. Thumbnails are cached in
        ``<directory>/thumbs`` rather than in the current working
        directory, so a cache is never reused for a different directory of
        images.
    downsample : int, optional
        Factor by which each image axis is reduced to make a thumbnail.
    max_workers : int, optional
        Number of threads used to make thumbnails. The default, 4, is both
        faster and roughly half the peak memory of one thread per CPU,
        which matters on a JupyterHub with a per-user memory cap.
    **kwargs
        Passed on to `ipywidgets.VBox`.
    """

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
        """
        Save the selection whenever a "Use image" checkbox changes.

        Observer for the ``value`` trait of every tile's checkbox, attached
        in ``__init__`` after the saved selection has been restored so that
        restoring does not trigger a save.
        """
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
        """
        Read the saved selection.

        Returns
        -------
        dict
            Mapping of file name to ``True`` (included) or ``False``. It is
            empty if there is no selection file, or if the file cannot be
            read or does not hold a mapping.

        Warns
        -----
        UserWarning
            If the file cannot be read or does not hold a mapping, in which
            case all of it is ignored, or if some of its values are not
            ``true`` or ``false``, in which case only those entries are
            ignored.

        Notes
        -----
        Entries with a value that is not a real boolean are dropped one at
        a time rather than rejecting the whole file. ``__init__`` saves the
        selection right after restoring it, so rejecting the whole file
        would throw away every other choice in it. A dropped entry means
        that image is included, which is the default for any image with no
        entry.
        """
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

        # bool("false") is True, so anything that is not a real boolean
        # must not be allowed through to the checkboxes.
        bad = [name for name, value in saved.items()
               if not isinstance(value, bool)]
        if bad:
            warnings.warn(
                f'Ignoring entries in {self.selection_path} whose value is '
                f'not true or false, and including those images: {bad}',
                stacklevel=2)
            saved = {name: value for name, value in saved.items()
                     if name not in bad}

        return saved

    def _restore_selection(self):
        """
        Set the checkboxes from the saved selection, if there is one.

        Notes
        -----
        Images with no saved entry are left checked (included).
        """
        saved = self._read_selection()
        if not saved:
            return
        for fname, selector in zip(self._im_file_names, self._selectors):
            selector._selector.value = saved.get(fname, True)

    def make_thumbnails(self, thumb_dir=None):
        """
        Make a PNG thumbnail for each image that does not have one yet.

        The image collection is refreshed first. A progress bar is displayed
        while thumbnails are being made, and nothing is displayed if all of
        them already exist.

        Parameters
        ----------
        thumb_dir : str, pathlib.Path or None, optional
            Directory the thumbnails are written to; it is created if
            needed. ``None`` means ``<directory>/thumbs``.
        """
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
        """
        Make one thumbnail-with-checkbox widget for each image.

        Thumbnails that no longer match an image in the directory are
        deleted first.

        This is called once, from ``__init__``, which afterwards restores
        the saved selection, starts watching the checkboxes and lays the
        widgets out. Calling it again replaces the widgets without doing
        any of that: the new ones are not shown, but they are what gets
        saved, so the saved selection stops matching what is on screen.

        Parameters
        ----------
        thumb_dir : str, pathlib.Path or None, optional
            Directory the thumbnails are read from. ``None`` means
            ``<directory>/thumbs``.
        """
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

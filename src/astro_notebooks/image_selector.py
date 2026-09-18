import html
import io
import json
import os
import re
import secrets
import shutil
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path

import ipyevents
import ipywidgets as ipw
import numpy as np

from astropy.io import fits
from astropy.nddata import CCDData, block_reduce
from astropy.visualization import simple_norm
from ccdproc import ImageFileCollection
from IPython.display import display
from PIL import Image
from reducer.astro_gui import Combiner
from reducer.image_browser import banded_block_reduce

from .image_quality import (
    DEFAULT_CUTOUT_SIZE,
    _image_hdu,
    measure_frame,
    select_reference_stars,
    summarize_metrics,
)

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

# Name of the file, in the thumbnail cache, that holds the star measurements
# so that they survive from one session to the next.
QUALITY_FILE_NAME = 'image_quality.json'

# Cutout PNGs live in the thumbnail directory beside the thumbnails, named
# <stem of the FITS file>_star<n>.png.
_CUTOUT_PNG_PATTERN = re.compile(r'^(?P<stem>.+)_star\d+\.png$')


def write_selection_manifest(isel, destination, run_label, included=None):
    """
    Record which frames went into a combination.

    Parameters
    ----------
    isel : ImageSelect
        The selector the frames were chosen in.
    destination : str or pathlib.Path
        Directory the manifest is written to. It is created if needed.
    run_label : str
        Base of the manifest's file name,
        ``<destination>/<run_label>_manifest.json``.
    included : list of str, optional
        File names, relative to the data directory, of the frames that
        were combined. ``None`` means the frames checked in ``isel`` at the
        moment of the call.

    Returns
    -------
    pathlib.Path
        Path of the manifest that was written.

    Notes
    -----
    The manifest holds the run label, an ISO timestamp, the data directory
    the frames came from, and the ``included`` and ``excluded`` file names
    (relative to that data directory).

    Pass ``included`` when the combination has already happened, as
    :class:`SelectedCombiner` does, so that the manifest lists the frames
    that were actually combined even if a checkbox has changed since.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if included is None:
        included = isel.selected_files
    else:
        included = list(included)
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
    """
    Write ``contents`` as JSON to ``path`` without a partial file.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write. Its directory must already exist.
    contents : object
        Anything `json.dump` can serialize.

    Notes
    -----
    The JSON goes to a temporary file in the same directory, which is
    flushed to disk and then renamed over ``path``, so a reader never sees
    a half-written file. The temporary file is removed if anything fails.

    A new file gets the permissions any ordinary new file would (set by
    the process umask). Rewriting an existing file keeps that file's
    permissions and adds any that a new file would get, so a file left
    readable by its owner only by an older version becomes readable like
    any other file.

    Syncing to disk and setting the permissions are best effort: some
    network and FUSE file systems refuse them, and that must not stop the
    selection being saved.
    """
    path = Path(path)
    tmp_name = path.parent / f'{path.name}.{secrets.token_hex(4)}.tmp'
    # Mode 0o666 lets the kernel apply the umask, unlike tempfile.mkstemp,
    # which always makes the file readable by its owner only.
    handle = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(handle, 'w') as f:
            json.dump(contents, f, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        try:
            # The temporary file's own mode is what the umask allows.
            new_mode = tmp_name.stat().st_mode | path.stat().st_mode
            os.chmod(tmp_name, new_mode & 0o7777)
        except OSError:
            # No existing file, or a file system that refuses chmod; the
            # umask-derived mode stands.
            pass
        os.replace(tmp_name, path)
    except BaseException:
        tmp_name.unlink(missing_ok=True)
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


def _cutout_png_path(thumb_dir, stem, index):
    """Path of the cached PNG of one star's cutout on one frame."""
    return Path(thumb_dir) / f'{stem}_star{index}.png'


def _remove_cutout_pngs(thumb_dir, keep_stems=()):
    """Delete the cached star cutouts, which are about to be remade.

    ``keep_stems`` are the stems of the thumbnails themselves, so that a
    frame whose own name ends in ``_star3`` keeps its thumbnail.
    """
    for png in Path(thumb_dir).glob('*_star*.png'):
        if png.stem not in keep_stems and _CUTOUT_PNG_PATTERN.match(png.name):
            png.unlink()


def _save_cutout_png(cutout, dest_path):
    """Save one star cutout as a small grayscale PNG."""
    scaled = _normalize(np.asarray(cutout, dtype=np.float32),
                        min_percent=1, max_percent=99.5)
    Image.fromarray((scaled * 255).astype(np.uint8), mode='L').save(dest_path)


def _enlarged_png(png_path, size):
    """Bytes of a cutout PNG blown up to ``size`` pixels across.

    The cutouts are only a few tens of pixels on a side, so they are
    enlarged with nearest-neighbor sampling: the point is to see the shape
    of the star, not to make a pretty picture of it.
    """
    with Image.open(png_path) as img:
        big = img.resize((size, size), Image.NEAREST)
        buffer = io.BytesIO()
        big.save(buffer, format='png')
    return buffer.getvalue()


def _measure_one_frame(fits_path, star_positions, thumb_dir, cutout_size):
    """Measure the reference stars on one frame and cache their cutouts.

    Runs in a worker thread. Only one small cutout at a time is read from
    the frame, so the memory this costs is negligible even for a big
    image. Returns the per-star measurements without the cutout arrays,
    which have been written to the thumbnail directory as PNGs instead.
    """
    try:
        measured = measure_frame(fits_path, star_positions,
                                 cutout_size=cutout_size)
    except Exception as error:
        warnings.warn(f'Could not measure stars in {Path(fits_path).name}: '
                      f'{error}', stacklevel=2)
        return []

    # Keyed by the full file name, like the thumbnails, so that x.fit and
    # x.fits do not share their cutouts.
    stem = Path(fits_path).name
    stars = []
    for index, star in enumerate(measured):
        cutout = star.pop('cutout')
        if cutout is not None:
            _save_cutout_png(cutout, _cutout_png_path(thumb_dir, stem, index))
        stars.append(star)
    return stars


def _default_viewer():
    """The image viewer used unless the caller supplies another."""
    # Imported here because pulling in bqplot takes a moment, and because
    # a test can replace the viewer entirely with viewer_factory.
    from astrowidgets.bqplot import ImageWidget

    return ImageWidget(display_width=400)


def _metrics_summary_html(fname, metrics):
    """Description of one frame's measurements for the details panel."""
    lines = [f'<b>{fname}</b>']
    if not metrics or metrics.get('fwhm') is None:
        lines.append('No star measurements are available for this frame.')
    else:
        fwhm = _flag_span('{:.2f} px'.format(metrics['fwhm']),
                          metrics.get('fwhm_flag'))
        lines.append(f'FWHM: {fwhm}')
        ellipticity = metrics.get('ellipticity')
        if ellipticity is not None:
            lines.append('Ellipticity: {:.2f}'.format(ellipticity))
        rel_flux = metrics.get('rel_flux')
        if rel_flux is not None:
            brightness = _flag_span('{:.2f}&times;'.format(rel_flux),
                                    metrics.get('flux_flag'))
            lines.append(f'Brightness vs. the night: {brightness}')
        lines.append('Stars measured: {}'.format(metrics.get('n_stars', 0)))
        if metrics.get('fwhm_flag'):
            lines.append('<i>Stars are broader here than in most frames.</i>')
        if metrics.get('flux_flag'):
            lines.append('<i>Stars are fainter here than in most frames.</i>')
    return '<br>'.join(lines)


def _flag_span(text, flagged):
    """``text``, in red when ``flagged``."""
    if not flagged:
        return text
    return f'<span style="color: #c62828; font-weight: bold;">{text}</span>'


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
        # Filled in by set_metrics once the stars have been measured.
        self._quality = ipw.HTML(value='FWHM: n/a')
        self._star_cutout = ipw.Image(
            format='png',
            layout=dict(width='64px', height='64px', object_fit='contain',
                        display='none')
        )

        ipw.link((self._selector, 'value'), (self._valid_mark, 'value'))
        # ipw.link((self, 'value'), (self._selector, 'value'))

        self.select_box = ipw.HBox(children=[self._selector, self._valid_mark])
        self.mobox = ipw.VBox(children=[self._name, self._quality,
                                        self._star_cutout, self.select_box])
        self.children = [self.image_display, self.mobox]
        self.layout.width = width

    def set_metrics(self, metrics, cutout_png=None):
        """Show this frame's star measurements on the tile.

        ``metrics`` is one frame's entry from
        :func:`~astro_notebooks.image_quality.summarize_metrics`, or None
        when there is nothing to show, and ``cutout_png`` is the PNG of the
        brightest reference star on this frame.
        """
        if not metrics or metrics.get('fwhm') is None:
            self._quality.value = 'FWHM: n/a'
        else:
            flagged = bool(metrics.get('fwhm_flag') or metrics.get('flux_flag'))
            text = f'FWHM: {metrics["fwhm"]:.2f} px'
            rel_flux = metrics.get('rel_flux')
            if rel_flux is not None:
                text += f' &middot; {rel_flux:.2f}&times;'
            self._quality.value = _flag_span(text, flagged)

        if cutout_png:
            self._star_cutout.value = cutout_png
            self._star_cutout.layout.display = None


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

    # How tall the scrolling panel of thumbnails is, and how wide the
    # tiles and the panel that holds them are.
    TILES_HEIGHT = '600px'
    TILES_WIDTH = '460px'
    TILE_WIDTH = '200px'

    def __init__(self, *args, directory=".", downsample=8,
                 max_workers=DEFAULT_MAX_WORKERS,
                 cutout_size=DEFAULT_CUTOUT_SIZE,
                 viewer_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path(directory)
        self._downsample = downsample
        self._max_workers = max_workers
        self._cutout_size = cutout_size
        self._viewer_factory = viewer_factory or _default_viewer
        self._collection = ImageFileCollection(self.path)

        # Cache thumbnails next to the data rather than in the current
        # working directory, so that a cache is never reused for a
        # different directory of images.
        self.thumbs = self.path / 'thumbs'
        self.star_positions = []
        self.metrics = {}
        self.make_thumbnails_and_metrics(thumb_dir=self.thumbs)
        self.make_selectors(thumb_dir=self.thumbs)
        self._apply_metrics()
        # Saving is switched off, for the life of this widget, if the
        # selection file cannot be read or the directory cannot be
        # written, so that a selection saved earlier is never overwritten
        # by one that was not restored from it.
        self._can_save = True
        # What this widget last wrote to the selection file, or None.
        # SelectedCombiner uses it to tell a failed save (the file still
        # holds this) from a file that something else has written since.
        self._last_saved = None
        self._save_problem = ''
        self._message = ipw.HTML()
        self._message.layout.display = 'none'
        # Restore first, then start watching the checkboxes, so that
        # restoring does not itself trigger a save.
        self._restore_selection()
        for selector in self._selectors:
            selector._selector.observe(self._selection_changed, names='value')
        # Save once now so that the selection file always exists, and so
        # that entries for files that have disappeared are dropped.
        if self._can_save:
            try:
                self.save_selection()
            except OSError as err:
                self._can_save = False
                self._save_problem = repr(err)
                warnings.warn(
                    f'The image selection cannot be saved to '
                    f'{self.selection_path}: {err!r}. Choices made in this '
                    f'widget will not be remembered.',
                    stacklevel=2)
        if not self._can_save:
            self._show_not_saving_message()

        # One frame at a time is shown at full resolution on the right;
        # nothing is loaded into the viewer until a thumbnail is clicked.
        self.viewer = self._viewer_factory()
        self.details = ipw.VBox(children=[
            ipw.HTML('Click a thumbnail to see the frame full size.')
        ])
        self._connect_clicks()

        self.tiles_box = ipw.Box(
            children=self._selectors,
            layout=ipw.Layout(flex_flow='row wrap',
                              overflow='hidden auto',
                              height=self.TILES_HEIGHT,
                              width=self.TILES_WIDTH)
        )
        right_panel = ipw.VBox(children=[self.viewer, self.details])
        self.children = [self._message,
                         ipw.HBox(children=[self.tiles_box, right_panel])]

    @property
    def selection_path(self):
        """Path of the JSON file that remembers the current selection."""
        return self.path / SELECTION_FILE_NAME

    @property
    def quality_path(self):
        """Path of the JSON file that caches the star measurements."""
        return self.thumbs / QUALITY_FILE_NAME

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

        Notes
        -----
        An exception raised in a widget observer is logged by the kernel
        and never reaches the notebook, so a failed save is reported in
        the widget's own message instead; that message is the only sign
        the user gets. Saving is tried again on the next change, and
        since each save writes the whole selection a later success loses
        nothing.
        """
        if not self._can_save:
            self._show_not_saving_message()
            return
        try:
            self.save_selection()
        except OSError as err:
            self._show_message(
                f'This change could NOT be saved: '
                f'{html.escape(repr(err))}. The selection '
                f'file {self.selection_path} does not match the checkboxes. '
                f'Saving will be tried again at the next change.',
                error=True)
        else:
            self._show_message('')

    @property
    def message(self):
        """Text currently shown above the grid of images."""
        return self._message.value

    def _show_message(self, text, error=False):
        """
        Show a message above the grid of images, or hide it.

        Parameters
        ----------
        text : str
            Message to show. An empty string hides the message.
        error : bool, optional
            If ``True`` the message is shown in bold red.
        """
        if error:
            text = f'<b style="color: #b00020">{text}</b>'
        self._message.value = text
        self._message.layout.display = 'flex' if text else 'none'

    def _show_not_saving_message(self):
        """Tell the user that this widget is not saving the selection."""
        self._show_message(
            f'Selections are NOT being saved in this session: '
            f'{html.escape(self._save_problem)}. Fix the problem and run '
            f'this cell again.', error=True)

    def save_selection(self):
        """Write the current checkbox state beside the data.

        Every file currently in the collection gets an entry, so entries
        for files that no longer exist are dropped.

        Nothing is written if saving has been switched off because the
        selection file could not be read, or the directory could not be
        written, when the widget was made.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        if not self._can_save:
            return
        selection = {fname: bool(selector._selector.value)
                     for fname, selector in zip(self._im_file_names,
                                                self._selectors)}
        _atomic_write_json(self.selection_path, selection)
        self._last_saved = selection

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
        A file that exists but cannot be read (a permission problem, or a
        passing fault on a network disk) may well hold a good selection,
        so it is left alone and saving is switched off for this widget
        (``_can_save``) rather than letting the next save overwrite it. A
        file that can be read but does not hold a JSON mapping is moved
        aside to ``image_selection.json.bak`` before a new one is written.

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
        except OSError as err:
            self._stop_saving(err)
            return {}
        except ValueError:
            # Not JSON at all; handled with "not a mapping" below.
            saved = None

        if not isinstance(saved, dict):
            backup = self.selection_path.with_name(
                self.selection_path.name + '.bak')
            try:
                os.replace(self.selection_path, backup)
            except OSError as err:
                self._stop_saving(err)
                return {}
            warnings.warn(
                f'Ignoring image selection file {self.selection_path}, '
                f'which does not contain a mapping of file name to True or '
                f'False; it has been moved to {backup.name}. Starting with '
                f'all images included.',
                stacklevel=2)
            return {}

        # bool("false") is True, so anything that is not a real boolean
        # must not be allowed through to the checkboxes.
        bad = [name for name, value in saved.items()
               if not isinstance(value, bool)]
        if bad:
            # The save that follows rewrites these entries as true, so keep
            # what the file said.
            backup = self.selection_path.with_name(
                self.selection_path.name + '.bak')
            try:
                shutil.copyfile(self.selection_path, backup)
            except OSError as err:
                self._stop_saving(err)
                return {}
            warnings.warn(
                f'Ignoring entries in {self.selection_path} whose value is '
                f'not true or false, and including those images: {bad}. '
                f'The file as it was has been copied to {backup.name}.',
                stacklevel=2)
            saved = {name: value for name, value in saved.items()
                     if name not in bad}

        return saved

    def _stop_saving(self, err):
        """
        Switch saving off because the selection file could not be read.

        Parameters
        ----------
        err : OSError
            The error raised while reading the file or moving it aside.
        """
        self._can_save = False
        self._save_problem = repr(err)
        warnings.warn(
            f'Could not read the image selection file '
            f'{self.selection_path}: {err!r}. It has been left alone, all '
            f'images start out included, and choices made in this widget '
            f'will not be saved.',
            stacklevel=3)

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

    def make_thumbnails_and_metrics(self, thumb_dir=None):
        """Build whatever the cache in ``thumb_dir`` is missing.

        Thumbnails and star measurements are made in the same pass, by the
        same small pool of threads, behind a single progress bar. Anything
        already cached is left alone, so opening the notebook a second time
        on the same data does no work at all.
        """
        thumby = Path(thumb_dir) if thumb_dir is not None else self.thumbs
        thumby.mkdir(parents=True, exist_ok=True)
        self._collection.refresh()
        # Full file names (with extension) and, in the same order, the names
        # used for the thumbnail PNGs. Those are the full file names too,
        # not the stems, so that x.fit and x.fits do not share a thumbnail
        # or a checkbox.
        self._im_file_names = []
        self._im_base_names = []
        todo = []
        for fname in self._collection.files_filtered(include_path=True):
            source = Path(fname)
            base = source.name
            self._im_file_names.append(source.name)
            self._im_base_names.append(base)
            dest_path = thumby / (base + '.png')
            if dest_path.exists():
                continue
            todo.append((source, dest_path))

        cached = self._read_quality_cache()
        if cached is None:
            # Picking the reference stars reads a few tiles of one frame,
            # and every frame is then measured at those same positions.
            _remove_cutout_pngs(thumby, self._im_base_names)
            self.star_positions = self._find_reference_stars()
            measure_todo = (list(self._im_file_names) if self.star_positions
                            else [])
        else:
            self.star_positions = cached['stars']
            self.metrics = cached['metrics']
            measure_todo = []

        if todo or measure_todo:
            measured = self._run_jobs(todo, measure_todo, thumby)
        else:
            measured = {}

        if cached is None:
            self.metrics = summarize_metrics(measured) if measured else {}
            self._write_quality_cache()

    def _run_jobs(self, thumbnail_todo, measure_todo, thumb_dir):
        """Run the thumbnail and measurement jobs behind a progress bar."""
        spinner_cls = Spinner if Spinner is not None else _MessageSpinner
        spinner = spinner_cls(message="Preparing images...")
        progress = ipw.IntProgress(
            value=0, min=0,
            max=len(thumbnail_todo) + len(measure_todo),
            description="Preparing"
        )
        progress_box = ipw.VBox(children=[spinner, progress])
        # Display right away so the user sees activity while the rest of
        # the widget is still being constructed.
        display(progress_box)
        spinner.start()

        measured = {}
        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(_make_one_thumbnail, src, dest,
                                    self._downsample): None
                    for src, dest in thumbnail_todo
                }
                for fname in measure_todo:
                    future = executor.submit(_measure_one_frame,
                                             self.path / fname,
                                             self.star_positions,
                                             thumb_dir,
                                             self._cutout_size)
                    futures[future] = fname

                for future in as_completed(futures):
                    result = future.result()
                    fname = futures[future]
                    if fname is not None:
                        measured[fname] = result
                    progress.value += 1
        finally:
            spinner.stop()
            progress_box.layout.display = "none"

        return measured

    def _find_reference_stars(self, max_frames=3):
        """Positions of the stars measured on every frame.

        The search stops at the first frame that yields stars; a frame or
        two after that are tried in case the first one is unusable. An
        empty list means this data set has no measurable stars, in which
        case the widget simply shows no quality numbers.
        """
        for fname in self._im_file_names[:max_frames]:
            try:
                stars = select_reference_stars(self.path / fname)
            except Exception as error:
                warnings.warn(f'Could not look for stars in {fname}: {error}',
                              stacklevel=2)
                continue
            if stars:
                return stars
        return []

    def _read_quality_cache(self):
        """Cached measurements, or None if they need to be made again.

        The cache is thrown away if the set of frames has changed or if any
        frame has been written since it was measured.
        """
        try:
            with open(self.quality_path) as f:
                cached = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None

        if not isinstance(cached, dict):
            return None
        if cached.get('cutout_size') != self._cutout_size:
            return None

        mtimes = cached.get('mtimes')
        if not isinstance(mtimes, dict):
            return None
        if set(mtimes) != set(self._im_file_names):
            return None
        for fname, recorded in mtimes.items():
            try:
                if os.path.getmtime(self.path / fname) > recorded:
                    return None
            except OSError:
                return None

        metrics = cached.get('metrics')
        stars = cached.get('stars')
        if not isinstance(metrics, dict) or not isinstance(stars, list):
            return None

        return {'stars': [tuple(star) for star in stars], 'metrics': metrics}

    def _write_quality_cache(self):
        """Save the measurements so the next session can skip the work."""
        mtimes = {}
        for fname in self._im_file_names:
            try:
                mtimes[fname] = os.path.getmtime(self.path / fname)
            except OSError:
                return
        try:
            _atomic_write_json(self.quality_path, {
                'cutout_size': self._cutout_size,
                'stars': [list(star) for star in self.star_positions],
                'mtimes': mtimes,
                'metrics': self.metrics,
            })
        except OSError as err:
            # Only a cache: the measurements are still shown, and are made
            # again next time. The widget must open regardless.
            warnings.warn(
                f'Could not save the star measurements to '
                f'{self.quality_path}: {err!r}. They will be made again '
                f'next time.', stacklevel=2)

    def make_selectors(self, thumb_dir=None):
        """
        Make one thumbnail-with-checkbox widget for each image.

        Thumbnails that no longer match an image in the directory are
        deleted first. That includes thumbnails named after the stem of an
        image (``x.png``) by earlier versions; they are now named after the
        full file name (``x.fit.png``).

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

        # Drop the thumbnails, and the star cutouts, of files that are no
        # longer in the data directory.
        thumbnails = {}
        for png in thumby.glob('*.png'):
            if png.stem in self._im_base_names:
                thumbnails[png.stem] = png
                continue
            cutout = _CUTOUT_PNG_PATTERN.match(png.name)
            if cutout and cutout.group('stem') in self._im_base_names:
                continue
            png.unlink()

        kiddos = {}
        for ims in self._im_base_names:
            image_png = thumbnails[ims].read_bytes()
            iws = ImageWithSelector(image_png, fname=ims,
                                    width=self.TILE_WIDTH)
            kiddos[ims] = iws

        self._selectors = [kiddos[ims] for ims in self._im_base_names]

    def _apply_metrics(self):
        """Put the star measurements on the tiles."""
        for fname, stem, tile in zip(self._im_file_names,
                                     self._im_base_names,
                                     self._selectors):
            cutout_path = _cutout_png_path(self.thumbs, stem, 0)
            cutout_png = (_enlarged_png(cutout_path, 64)
                          if cutout_path.exists() else None)
            tile.set_metrics(self.metrics.get(fname), cutout_png=cutout_png)

    def _connect_clicks(self):
        """Make a click on a thumbnail show that frame in the viewer."""
        # The Event objects have to outlive this method, or the clicks stop
        # being reported.
        self._click_events = []
        for index, tile in enumerate(self._selectors):
            event = ipyevents.Event(source=tile.image_display,
                                    watched_events=['click'])
            event.on_dom_event(partial(self._clicked, index))
            self._click_events.append(event)

    def _clicked(self, index, _event):
        """Show the frame whose thumbnail was clicked; called by ipyevents."""
        self._show_frame(index)

    def show_frame(self, name):
        """Show the named frame, as if its thumbnail had been clicked.

        ``name`` is a file name with its extension, as it appears in
        :attr:`selected_files`.
        """
        self._show_frame(self._im_file_names.index(name))

    def _show_frame(self, index):
        """Load frame ``index`` into the viewer and describe it."""
        fname = self._im_file_names[index]
        path = self.path / fname
        try:
            # No image label, so each frame replaces the one before it
            # rather than piling up in the viewer.
            self.viewer.load_image(str(path))
        except ValueError:
            # Frames with no BUNIT keyword cannot be read straight from a
            # file name, but they can be read with a unit supplied.
            self.viewer.load_image(CCDData.read(path, unit='adu'))
        self.details.children = self._details(index)

    def _details(self, index):
        """Widgets describing one frame for the panel under the viewer."""
        fname = self._im_file_names[index]
        stem = self._im_base_names[index]
        summary = ipw.HTML(_metrics_summary_html(fname,
                                                 self.metrics.get(fname)))

        cutouts = []
        for star in range(len(self.star_positions)):
            cutout_path = _cutout_png_path(self.thumbs, stem, star)
            if not cutout_path.exists():
                continue
            cutouts.append(ipw.Image(
                value=_enlarged_png(cutout_path, 100),
                format='png',
                layout=dict(width='100px', height='100px',
                            object_fit='contain')
            ))

        if not cutouts:
            return [summary]
        return [summary,
                ipw.HTML('The stars measured on this frame:'),
                ipw.Box(children=cutouts,
                        layout=ipw.Layout(flex_flow='row wrap'))]


class SelectedCombiner(Combiner):
    """
    Combine the frames that are checked in an `ImageSelect`.

    Parameters
    ----------
    *args
        Passed on to `reducer.astro_gui.Combiner`.
    image_select : ImageSelect
        The selector whose checked frames are combined.
    run_label : str
        Base of the name of the combined image(s) and of the manifest that
        lists the frames they were made from.
    **kwargs
        Passed on to `reducer.astro_gui.Combiner`, for example
        ``description``, ``toggle_type``, ``group_by``, ``apply_to`` and
        ``destination``. Do not pass ``image_source`` or
        ``file_name_base``; they are set from ``image_select`` and
        ``run_label``.

    Attributes
    ----------
    manifest_path : pathlib.Path or None
        Manifest written by the most recent combination, or ``None`` if
        there has not been a successful one.
    message : str
        Text shown below the combine button: why nothing was combined,
        why the combination failed, or what was combined.
    last_error : Exception or None
        What went wrong the last time the button was pressed, or ``None``
        if nothing did.
    last_traceback : str or None
        Formatted traceback of ``last_error``; ``print`` it to read it.

    Notes
    -----
    The frames that are combined are the ones checked *when the combine
    button is pressed*, not the ones checked when this widget was made. A
    plain ``Combiner`` given
    ``ImageFileCollection(filenames=isel.selected_files)`` combines a
    snapshot taken when that collection was made, because refreshing a
    collection keeps its list of file names, so a frame unchecked later
    would still be combined.

    Nothing is combined if no frame is checked. This is refused here
    because an `~ccdproc.ImageFileCollection` given an empty list of file
    names uses every file in the directory.

    Only the checked frames that match ``apply_to`` are combined, and only
    those are listed as ``included`` in the manifest; a checked dark in a
    directory of lights is listed as ``excluded``. Nothing is combined if
    no checked frame matches.

    Nothing is combined either if the selection file beside the data does
    not match the checkboxes of ``image_select``. That happens when the
    cell that makes the selector is run again without re-running the cell
    that makes this widget, which would otherwise silently combine the
    frames checked in the old, discarded selector.

    When a combination finishes, `write_selection_manifest` records the
    frames that went into it, from the same snapshot, next to the result.
    No manifest is written if the combination fails.

    No exception is allowed to escape from `action`. reducer only shows
    the "Unlock settings" button once `action` has returned, so an
    exception would leave the widget locked with no way to try again.
    Failures are shown in the widget instead and kept in ``last_error``
    and ``last_traceback``.

    Two attributes that are private to reducer are set when the button is
    pressed: ``Combiner._image_source`` and the ``_image_source`` of the
    combiner's group-by widget, which keeps its own reference to the
    collection and would otherwise group the frames using the old one.
    """

    def __init__(self, *args, image_select, run_label, **kwargs):
        """Make the widget; the parameters are in the class docstring."""
        for reserved in ('image_source', 'file_name_base'):
            if reserved in kwargs:
                raise TypeError(
                    f'{reserved} cannot be given to SelectedCombiner; it '
                    f'is set from image_select and run_label.')
        self._image_select = image_select
        self._run_label = run_label
        self.manifest_path = None
        self.last_error = None
        self.last_traceback = None
        super().__init__(*args, file_name_base=run_label, **kwargs)

        self._message = ipw.HTML()
        self._message.layout.display = 'none'
        self.children = list(self.children) + [self._message]

    @property
    def message(self):
        """Text currently shown below the combine button."""
        return self._message.value

    def _show_message(self, text, error=False, detail=''):
        """
        Show a message below the combine button, or hide it.

        Parameters
        ----------
        text : str
            Message to show. An empty string hides the message.
        error : bool, optional
            If ``True`` the message is shown in bold red.
        detail : str, optional
            Plain text, such as a traceback, shown folded up below the
            message.
        """
        if error:
            text = f'<b style="color: #b00020">{text}</b>'
        if detail:
            text += (f'<details><summary>Details</summary>'
                     f'<pre>{html.escape(detail)}</pre></details>')
        self._message.value = text
        self._message.layout.display = 'flex' if text else 'none'

    def _show_failure(self, text, err):
        """
        Show a failure in the widget and remember the exception.

        Must be called while ``err`` is being handled, so that its
        traceback can be recorded.

        Parameters
        ----------
        text : str
            What failed, in words. The exception is added to it.
        err : Exception
            The exception that was caught.
        """
        self.last_error = err
        self.last_traceback = traceback.format_exc()
        self._show_message(
            f'{text} {html.escape(repr(err))}. Press "Unlock settings" to '
            f'try again.', error=True, detail=self.last_traceback)

    def _selection_on_disk(self):
        """
        Read the selector's selection file without changing anything.

        Returns
        -------
        dict or None
            Contents of the file, or ``None`` if it cannot be read or does
            not hold a mapping, in which case it cannot be compared with
            the checkboxes.
        """
        try:
            with open(self._image_select.selection_path) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            return None
        return saved if isinstance(saved, dict) else None

    def _selector_is_stale(self):
        """
        Whether the saved selection disagrees with the selector's checkboxes.

        Returns
        -------
        bool
            ``True`` if the selection file beside the data differs from the
            state of the checkboxes in the selector this widget was made
            with, and also from what that selector last wrote. Always
            ``False`` if the selector cannot save its selection or the
            file cannot be read, because then the two are not expected to
            match; `_combine_selected` says so in its message instead.

        Notes
        -----
        Every change of a checkbox is saved at once, so the two only differ
        if something else has written the file since: usually a newer
        selector made by re-running the selector's cell, which is the one
        the user is looking at. If the selector's latest save failed, the
        file still holds what the selector wrote before that, which is not
        stale: the selector has already told the user about the failed
        save, and the checkboxes are what the user sees.
        """
        isel = self._image_select
        if not getattr(isel, '_can_save', True):
            return False
        on_disk = self._selection_on_disk()
        if on_disk is None:
            return False
        current = {fname: bool(selector._selector.value)
                   for fname, selector in zip(isel._im_file_names,
                                              isel._selectors)}
        return on_disk not in (current, getattr(isel, '_last_saved', None))

    def action(self):
        """
        Combine the frames that are checked right now.

        This runs when the combine button is pressed. It never raises; see
        the Notes of the class. A failure is shown in the widget and kept
        in ``last_error`` and ``last_traceback``.
        """
        self.manifest_path = None
        self.last_error = None
        self.last_traceback = None
        try:
            self._combine_selected()
        except (Exception, KeyboardInterrupt) as err:
            self._show_failure('Something went wrong before the images '
                               'were combined, and no manifest was '
                               'written:', err)

    def _combine_selected(self):
        """
        Do the work of `action`: check, combine, then write the manifest.

        Failures of the combination and of writing the manifest are shown
        in the widget here. Anything else that goes wrong is raised, and
        `action` shows it.
        """
        selected = self._image_select.selected_files

        if not selected:
            self._show_message(
                'No images are checked, so nothing was combined. Check at '
                'least one image, press "Unlock settings" and try again.',
                error=True)
            return

        if self._selector_is_stale():
            self._show_message(
                f'Nothing was combined. The selection saved in '
                f'{self._image_select.selection_path} does not match the '
                f'image selector this combiner was made with. Most likely '
                f'the cell that makes the image selector was run again (or '
                f'the selection was changed from another notebook): re-run '
                f'this cell too, then try again.', error=True)
            return

        missing = [fname for fname in selected
                   if not (Path(self._image_select.path) / fname).exists()]
        if missing:
            self._show_message(
                f'Nothing was combined. These checked images are no longer '
                f'in {self._image_select.path}: '
                f'{html.escape(", ".join(missing))}. Run the cell that '
                f'makes the image selector again, then this cell, and try '
                f'again.', error=True)
            return

        self._show_message('')
        collection = ImageFileCollection(location=self._image_select.path,
                                         filenames=selected)
        # Only the checked frames that match apply_to get combined, so only
        # those may be recorded as included. With no apply_to this is every
        # checked frame the collection could read.
        apply_to = self.apply_to if self._apply_to else {}
        to_combine = list(collection.files_filtered(**apply_to))
        if not to_combine:
            wanted = ', '.join(f'{k}={v}' for k, v in apply_to.items())
            self._show_message(
                f'None of the checked images match {wanted or "the settings"}, '
                f'so nothing was combined. Check at least one such image, '
                f'press "Unlock settings" and try again.', error=True)
            return
        if len(to_combine) != len(selected):
            collection = ImageFileCollection(
                location=self._image_select.path, filenames=to_combine)
        # reducer offers no public way to replace the collection, and the
        # group-by widget holds its own reference to it; see Notes above.
        self._image_source = collection
        self._group_by._image_source = collection

        try:
            super().action()
        except (Exception, KeyboardInterrupt) as err:
            self._show_failure('The combination failed, and no manifest '
                               'was written:', err)
            return

        try:
            self.manifest_path = write_selection_manifest(
                self._image_select, self.destination, self._run_label,
                included=to_combine)
        except Exception as err:
            self._show_failure(
                f'The images WERE combined and written to '
                f'{self.destination}, but the manifest that lists them '
                f'could not be written:', err)
            return

        done = (f'Done. {len(to_combine)} of the {len(selected)} checked '
                f'images were combined; they are listed in '
                f'{self.manifest_path}.')
        n_skipped = len(selected) - len(to_combine)
        if n_skipped:
            done += (f' {n_skipped} checked image(s) did not match '
                     f'apply_to and were left out.')
        if not getattr(self._image_select, '_can_save', True):
            done += (f' <b>Note:</b> the image selector could not read or '
                     f'save {self._image_select.selection_path}, so the '
                     f'checkboxes as shown were used; they may differ from '
                     f'a selection saved earlier.')
        self._show_message(done)

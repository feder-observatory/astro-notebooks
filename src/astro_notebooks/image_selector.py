import html
import json
import os
import secrets
import traceback
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
from reducer.astro_gui import Combiner
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
    the process umask); rewriting an existing file keeps that file's
    permissions.
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
            os.fsync(f.fileno())
        try:
            os.chmod(tmp_name, path.stat().st_mode & 0o7777)
        except FileNotFoundError:
            # No existing file, so the umask-derived mode stands.
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
        # Saving is switched off, for the life of this widget, if the
        # selection file cannot be read or the directory cannot be
        # written, so that a selection saved earlier is never overwritten
        # by one that was not restored from it.
        self._can_save = True
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
        self.n_cols = 4
        gs = self._make_grid()
        self.children = [self._message, gs]
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
                f'This change could NOT be saved: {err!r}. The selection '
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
            f'{self._save_problem}. Fix the problem and run this cell '
            f'again.', error=True)

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
            warnings.warn(
                f'Ignoring entries in {self.selection_path} whose value is '
                f'not true or false, and including those images: {bad}',
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
            with. Always ``False`` if the selector cannot save its
            selection or the file cannot be read, because then the two are
            not expected to match.

        Notes
        -----
        Every change of a checkbox is saved at once, so the two only differ
        if something else has written the file since: usually a newer
        selector made by re-running the selector's cell, which is the one
        the user is looking at.
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
        return on_disk != current

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
        except Exception as err:
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

        self._show_message('')
        collection = ImageFileCollection(location=self._image_select.path,
                                         filenames=selected)
        # Only the checked frames that match apply_to get combined, so only
        # those may be recorded as included.
        apply_to = self.apply_to if self._apply_to else {}
        to_combine = list(selected)
        if apply_to:
            to_combine = list(collection.files_filtered(**apply_to))
            if not to_combine:
                wanted = ', '.join(f'{k}={v}' for k, v in apply_to.items())
                self._show_message(
                    f'None of the checked images match {wanted}, so '
                    f'nothing was combined. Check at least one such image, '
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
        except Exception as err:
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

        n_total = len(self._image_select._im_file_names)
        done = (f'Done. {len(to_combine)} of {n_total} images were combined; '
                f'they are listed in {self.manifest_path}.')
        n_skipped = len(selected) - len(to_combine)
        if n_skipped:
            done += (f' {n_skipped} checked image(s) did not match '
                     f'apply_to and were left out.')
        self._show_message(done)

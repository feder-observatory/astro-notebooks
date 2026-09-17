import io
import json
import os
import re
import tempfile
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

    stem = Path(fits_path).stem
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
        # Restore first, then start watching the checkboxes, so that
        # restoring does not itself trigger a save.
        self._restore_selection()
        for selector in self._selectors:
            selector._selector.observe(self._selection_changed, names='value')
        # Save once now so that the selection file always exists, and so
        # that entries for files that have disappeared are dropped.
        self.save_selection()

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
        self.children = [ipw.HBox(children=[self.tiles_box, right_panel])]

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
        _atomic_write_json(self.quality_path, {
            'cutout_size': self._cutout_size,
            'stars': [list(star) for star in self.star_positions],
            'mtimes': mtimes,
            'metrics': self.metrics,
        })

    def make_selectors(self, thumb_dir=None):
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

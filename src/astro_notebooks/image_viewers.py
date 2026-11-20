"""Viewers for a folder of FITS images and for a single FITS image."""

import os
import warnings

import ipywidgets as ipw
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.visualization import (
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)
from astropy.wcs import FITSFixedWarning
from astrowidgets.bqplot import ImageWidget
from ccdproc import ImageFileCollection
from ipyfilechooser import FileChooser
from reducer.image_browser import ImageBrowser

#: Header keywords the folder viewer groups the images by, outermost first.
TREE_KEYS = ("imagetyp", "object")

#: The stretches the single image viewer offers, the same ones as the
#: colour image maker.
STRETCHES = {
    "linear": LinearStretch(),
    "log": LogStretch(),
    "sqrt": SqrtStretch(),
}

#: File names the single image chooser lists.
FITS_PATTERNS = ["*.fit*", "*.fit*.[bg]z"]

#: Percentiles of the image the ends of the black and white slider are set
#: to. Not the minimum and maximum: a few hot pixels or one saturated star
#: would stretch the slider until the sky was a sliver at one end of it.
SLIDER_PERCENTILES = (0.1, 99.9)

#: Percentiles of the image the black and white points start at. These are
#: the cuts astrowidgets itself starts an image with.
DEFAULT_CUT_PERCENTILES = (1, 95)

#: How many steps the black and white slider has from end to end.
SLIDER_STEPS = 1000

#: About how many pixels the percentiles are worked out from. A whole
#: frame takes around a second; a sample this size gives the same answer.
PERCENTILE_SAMPLE = 1_000_000


def read_image(path):
    """
    Read a FITS image, whether or not its header gives a unit.

    Parameters
    ----------
    path : str or `os.PathLike`
        A FITS file.

    Returns
    -------
    `astropy.nddata.CCDData`
        The image, in the unit its header gives, or in adu if it gives
        none. Raw camera images give none, and ``CCDData.read`` refuses
        to read them without being told one.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
        try:
            return CCDData.read(path)
        except ValueError as error:
            if "unit" not in str(error):
                raise
            return CCDData.read(path, unit="adu")


def level_range(data):
    """
    Work out the range and starting value of the black and white slider.

    Parameters
    ----------
    data : array-like
        The image.

    Returns
    -------
    limits : tuple of float
        The ends of the slider, the `SLIDER_PERCENTILES` of the pixels
        that are finite. The upper is always above the lower, so that
        they can be the ends of a slider, even if the image is one value.
    cuts : tuple of float
        The black and white points to start at, the
        `DEFAULT_CUT_PERCENTILES`, which are inside the limits.
    """
    data = np.asarray(data)
    # Every nth pixel of every nth row, so that the sample is spread over
    # the whole image.
    stride = max(1, int(np.sqrt(data.size / PERCENTILE_SAMPLE)))
    sample = data[::stride, ::stride]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return (0.0, 1.0), (0.0, 1.0)

    low, black, white, high = (
        float(value)
        for value in np.percentile(
            sample, [SLIDER_PERCENTILES[0], *DEFAULT_CUT_PERCENTILES, SLIDER_PERCENTILES[1]]
        )
    )
    if high <= low:
        return (low, low + 1.0), (low, low + 1.0)
    return (low, high), (black, white)


class FolderViewer(ipw.VBox):
    """
    Browse the FITS images in a folder that is picked with a folder chooser.

    Parameters
    ----------
    folder : str or `os.PathLike`, optional
        The folder shown to begin with. The default is the current
        directory.
    keys : sequence of str, optional
        Header keywords the images are grouped by, outermost first. A
        keyword none of the images in a folder has is left out for that
        folder.

    Attributes
    ----------
    tree_keys : list of str
        The header keywords the images are grouped by.
    chooser : `ipyfilechooser.FileChooser`
        Picks the folder.
    note : `ipywidgets.HTML`
        Says that a folder is being read, or why there is nothing to show.
    browser : `reducer.image_browser.ImageBrowser` or None
        The browser of the folder that is shown, or None if the folder
        has no FITS images in it.
    folder : str
        The folder that is shown.
    """

    def __init__(self, folder=".", keys=TREE_KEYS):
        super().__init__()
        # Not ``keys``: that is the widget's own list of what it sends to
        # the page.
        self.tree_keys = list(keys)
        self.folder = os.path.abspath(folder)
        self.browser = None

        self.chooser = FileChooser(
            self.folder,
            title="Folder of images",
            select_default=True,
            show_only_dirs=True,
        )
        self.chooser.register_callback(self._folder_chosen)
        self.note = ipw.HTML()
        self._browser_box = ipw.Box()
        self.children = [self.chooser, self.note, self._browser_box]

        self.show_folder(self.folder)

    def _folder_chosen(self, chooser):
        self.show_folder(chooser.selected_path)

    def show_folder(self, folder):
        """
        Show the images in a folder in place of the ones shown now.

        Parameters
        ----------
        folder : str or `os.PathLike`
            The folder to show. If it has no FITS images in it, or cannot
            be read, the note says so and no browser is shown.
        """
        self.folder = os.path.abspath(folder)
        self.browser = None
        self._browser_box.children = []
        # Every header in the folder is read, which takes a while for a
        # big folder.
        self.note.value = "Reading the images in this folder…"

        try:
            with warnings.catch_warnings():
                # The collection warns when a folder has no FITS files;
                # the note says so instead.
                warnings.simplefilter("ignore")
                collection = ImageFileCollection(self.folder)
            summary = collection.summary
            if summary is None or len(summary) == 0:
                self.note.value = "There are no FITS images in this folder."
                return
            keys = [key for key in self.tree_keys if key in summary.colnames]
            # The browser fills in a missing value of a key only if it is
            # given a key to do it for.
            self.browser = ImageBrowser(collection, keys=keys, allow_missing=bool(keys))
        except Exception as error:  # noqa: BLE001
            # An error raised in a widget callback is not shown in the
            # notebook, so without this a folder that cannot be read
            # would just leave the viewer empty.
            self.note.value = f"Could not read this folder: {error}"
            return

        self._browser_box.children = [self.browser]
        self.note.value = ""


class SingleImageViewer(ipw.HBox):
    """
    View one FITS image, with a stretch and black and white points to set.

    The stretch chooser and the black and white slider are the ones the
    colour image maker has for each of its colour channels.

    Parameters
    ----------
    path : str or `os.PathLike`, optional
        An image to show to begin with. The default is to show nothing
        until an image is picked with the file chooser.

    Attributes
    ----------
    image_widget : `astrowidgets.bqplot.ImageWidget`
        Shows the image.
    chooser : `ipyfilechooser.FileChooser`
        Picks the image.
    stretch_chooser : `ipywidgets.Dropdown`
        Picks the stretch, one of `STRETCHES`.
    level_slider : `ipywidgets.FloatRangeSlider`
        Sets the black and white points. Its ends are set from each image
        as it is loaded.
    note : `ipywidgets.HTML`
        Says why an image could not be shown.
    """

    def __init__(self, path=None):
        super().__init__()
        self.image_widget = ImageWidget()
        self.chooser = FileChooser(title="Choose an image", filter_pattern=FITS_PATTERNS)
        self.chooser.register_callback(self._image_chosen)

        self.stretch_chooser = ipw.Dropdown(options=list(STRETCHES), description="Stretch")
        self.level_slider = ipw.FloatRangeSlider(
            value=(0, 1),
            min=0,
            max=1,
            step=1 / SLIDER_STEPS,
            description="Set black and white",
            style={"description_width": "initial"},
            continuous_update=False,
            layout={"width": "100%"},
            # Nothing to set until there is an image.
            disabled=True,
        )
        self.note = ipw.HTML()
        self.stretch_chooser.observe(self._stretch_changed, names="value")
        self.level_slider.observe(self._levels_changed, names="value")

        self.viewer = ipw.VBox([self.stretch_chooser, self.level_slider, self.image_widget])
        self.children = [self.viewer, ipw.VBox([self.chooser, self.note])]
        # The file chooser drops below the image in a narrow window.
        self.layout.flex_flow = "row wrap"

        self._loaded = False
        if path is not None:
            self.load(path)

    def _image_chosen(self, chooser):
        self.load(chooser.selected)

    def load(self, path):
        """
        Show an image in place of the one shown now.

        The ends of the black and white slider are set from the image and
        the black and white points go back to where they start. The
        stretch stays as it is chosen. If the image cannot be read, the
        note says so and the image shown before stays.

        Parameters
        ----------
        path : str or `os.PathLike`
            A FITS file.
        """
        try:
            image = read_image(path)
            limits, cuts = level_range(image.data)
            self.image_widget.load_image(image)
        except Exception as error:  # noqa: BLE001
            # An error raised in a widget callback is not shown in the
            # notebook, so without this nothing at all would happen.
            self.note.value = f"Could not show {os.path.basename(str(path))}: {error}"
            return
        self.note.value = ""
        self._loaded = True

        # All at once: one at a time, a new minimum above the old maximum
        # is refused.
        with self.level_slider.hold_trait_notifications():
            self.level_slider.min, self.level_slider.max = limits
            self.level_slider.step = (limits[1] - limits[0]) / SLIDER_STEPS
            self.level_slider.value = cuts
        self.level_slider.disabled = False

        # The viewer keeps the cuts and stretch of the image it showed
        # before, and the slider's observer does not run if its value did
        # not change, so give the viewer both.
        self._apply_levels()
        self._apply_stretch()

    def _apply_levels(self):
        self.image_widget.set_cuts(ManualInterval(*self.level_slider.value))

    def _apply_stretch(self):
        self.image_widget.set_stretch(STRETCHES[self.stretch_chooser.value])

    def _levels_changed(self, change):
        if self._loaded:
            self._apply_levels()

    def _stretch_changed(self, change):
        if self._loaded:
            self._apply_stretch()

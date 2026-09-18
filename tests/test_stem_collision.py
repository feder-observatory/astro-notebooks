import json

import numpy as np
import pytest
from astropy.io import fits

from astro_notebooks import image_selector
from astro_notebooks.image_selector import SELECTION_FILE_NAME, ImageSelect

# Two of these differ only in their extension, so they share a stem.
COLLIDING_NAMES = ["frame-1.fit", "frame-1.fits", "frame-2.fit"]


@pytest.fixture
def colliding_dir(tmp_path):
    """Directory of small FITS images, two of which share a stem.

    ``frame-1.fit`` and ``frame-1.fits`` are different images with
    different pixel values, so a thumbnail made from one cannot be mistaken
    for a thumbnail made from the other.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(11)
    for name in COLLIDING_NAMES:
        hdu = fits.PrimaryHDU(rng.uniform(100.0, 1000.0, size=(64, 64)))
        hdu.header["IMAGETYP"] = "LIGHT"
        hdu.writeto(data_dir / name)
    return data_dir


def _selector_for(widget, fname):
    """Return the thumbnail-with-checkbox widget for the file ``fname``."""
    return widget._selectors[widget._im_file_names.index(fname)]


def test_files_sharing_a_stem_get_separate_selectors(colliding_dir):
    """``x.fit`` and ``x.fits`` each get their own widget and checkbox.

    Keying the selectors by stem made the two files share one object, so
    there must be three distinct selectors and three distinct checkboxes
    for the three files, each labelled with its full file name.
    """
    w = ImageSelect(directory=colliding_dir)
    assert sorted(w._im_file_names) == sorted(COLLIDING_NAMES)
    assert len(w._selectors) == 3
    assert len({id(sel) for sel in w._selectors}) == 3
    assert len({id(sel._selector) for sel in w._selectors}) == 3
    assert [sel._fname for sel in w._selectors] == w._im_file_names


def test_unticking_one_file_leaves_its_stem_twin_checked(colliding_dir):
    """Rejecting ``frame-1.fit`` does not reject ``frame-1.fits``.

    The twin must stay checked on screen, stay in ``selected_files`` (and
    so in the combination), and be recorded as kept in the selection file.
    """
    w = ImageSelect(directory=colliding_dir)
    _selector_for(w, "frame-1.fit")._selector.value = False

    assert _selector_for(w, "frame-1.fits")._selector.value is True
    assert sorted(w.selected_files) == ["frame-1.fits", "frame-2.fit"]
    saved = json.loads((colliding_dir / SELECTION_FILE_NAME).read_text())
    assert saved == {"frame-1.fit": False,
                     "frame-1.fits": True,
                     "frame-2.fit": True}


def test_restore_keeps_stem_twins_apart(colliding_dir):
    """A saved selection that treats stem twins differently is restored.

    With one shared checkbox the second entry overwrote the first, and the
    save made when the widget is built then wrote both back as kept, which
    erased the rejection for good. Both the widget and the file on disk
    must still hold the saved state afterwards.
    """
    saved = {"frame-1.fit": False, "frame-1.fits": True, "frame-2.fit": True}
    (colliding_dir / SELECTION_FILE_NAME).write_text(json.dumps(saved))

    w = ImageSelect(directory=colliding_dir)

    assert _selector_for(w, "frame-1.fit")._selector.value is False
    assert _selector_for(w, "frame-1.fits")._selector.value is True
    assert sorted(w.selected_files) == ["frame-1.fits", "frame-2.fit"]
    on_disk = json.loads((colliding_dir / SELECTION_FILE_NAME).read_text())
    assert on_disk == saved


def test_files_sharing_a_stem_get_separate_thumbnails(colliding_dir):
    """Each file has its own thumbnail, named after the full file name.

    The thumbnails of the stem twins must be different files with
    different contents, and each selector must show the thumbnail of its
    own file rather than its twin's.
    """
    w = ImageSelect(directory=colliding_dir)
    names = {p.name for p in w.thumbs.glob("*.png")}
    assert names == {name + ".png" for name in COLLIDING_NAMES}

    fit_png = (w.thumbs / "frame-1.fit.png").read_bytes()
    fits_png = (w.thumbs / "frame-1.fits.png").read_bytes()
    assert fit_png != fits_png
    shown_fit = bytes(_selector_for(w, "frame-1.fit").image_display.value)
    shown_fits = bytes(_selector_for(w, "frame-1.fits").image_display.value)
    assert shown_fit == fit_png
    assert shown_fits == fits_png


def test_old_stem_named_thumbnail_is_removed(colliding_dir):
    """A thumbnail cached under the old stem-based name is deleted.

    Earlier versions wrote ``thumbs/frame-1.png``. It no longer matches
    any image, so it must not be left in the cache for ever, and the
    thumbnails under the new names must still be made.
    """
    thumbs = colliding_dir / "thumbs"
    thumbs.mkdir()
    old_thumbnail = thumbs / "frame-1.png"
    old_thumbnail.write_bytes(b"not really a png")

    w = ImageSelect(directory=colliding_dir)

    assert not old_thumbnail.exists()
    names = {p.name for p in w.thumbs.glob("*.png")}
    assert names == {name + ".png" for name in COLLIDING_NAMES}

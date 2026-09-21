"""Shared utilities: font loading, image orientation, Tk helpers."""

import contextlib
import os
import sys

from PIL import Image

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(name):
    """True for a real image by extension; false for a dotfile like macOS's
    ._name.
    """
    return not name.startswith(".") and name.lower().endswith(IMAGE_EXTENSIONS)


_CUSTOM_FONT_LOADED = False


@contextlib.contextmanager
def suppress_tk_mac_warnings():
    if sys.platform == "darwin":
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    else:
        yield


_ARCHIVO_FILES = (
    "Archivo-Regular.ttf",
    "Archivo-Bold.ttf",
    "Archivo-Medium.ttf",
    "Archivo-SemiBold.ttf",
)


def _existing_font_paths():
    return [
        p
        for p in (os.path.join(ASSETS_DIR, n) for n in _ARCHIVO_FILES)
        if os.path.exists(p)
    ]


def _load_custom_fonts():
    """Register the bundled Archivo fonts with the OS for this process.

    Returns True only when at least one font file was registered, so
    _get_font_family never names a family that is not available.
    """
    global _CUSTOM_FONT_LOADED
    if _CUSTOM_FONT_LOADED:
        return True
    paths = _existing_font_paths()
    if not paths:
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes

            FR_PRIVATE = 0x10
            gdi32 = ctypes.windll.gdi32
            added = [
                gdi32.AddFontResourceExW(path, FR_PRIVATE, 0) for path in paths
            ]
            _CUSTOM_FONT_LOADED = any(added)
            return _CUSTOM_FONT_LOADED
        except Exception:
            return False
    elif sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            ct_path = ctypes.util.find_library("CoreText")
            if ct_path:
                ct = ctypes.cdll.LoadLibrary(ct_path)
                cf_path = ctypes.util.find_library("CoreFoundation")
                cf = ctypes.cdll.LoadLibrary(cf_path)
                # Undeclared, ctypes truncates the 64-bit CFURLRef to int and
                # CoreText segfaults.
                cf.CFURLCreateFromFileSystemRepresentation.restype = (
                    ctypes.c_void_p
                )
                cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_long,
                    ctypes.c_bool,
                ]
                cf.CFRelease.argtypes = [ctypes.c_void_p]
                ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
                ct.CTFontManagerRegisterFontsForURL.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                ]
                registered = False
                for path in paths:
                    encoded = path.encode("utf-8")
                    url_ref = cf.CFURLCreateFromFileSystemRepresentation(
                        None, encoded, len(encoded), False
                    )
                    if url_ref:
                        if ct.CTFontManagerRegisterFontsForURL(
                            url_ref, 1, None
                        ):
                            registered = True
                        cf.CFRelease(url_ref)
                _CUSTOM_FONT_LOADED = registered
                return registered
        except Exception:
            return False
    return False


def _get_font_family():
    """Pick the UI font family: bundled Archivo if registered, else a system
    sans.
    """
    import tkinter.font as tkFont

    if _CUSTOM_FONT_LOADED:
        return "Archivo"
    try:
        available = set(tkFont.families())
    except Exception:
        available = set()
    for family in (
        "Segoe UI",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ):
        if family in available:
            return family
    return "TkDefaultFont"


# TIFF/EXIF tag id for Orientation; ExifTags.Base needs Pillow 9.4, so use the
# number.
_EXIF_ORIENTATION_TAG = 0x0112

_ORIENTATION_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def oriented_size(path):
    """(width, height, orientation) of an image file after EXIF rotation,
    without decoding.
    """
    with Image.open(path) as img:
        width, height = img.size
        try:
            orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1))
        except Exception:
            orientation = 1
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return width, height, orientation


def auto_orient_image(img):
    """Return *img* rotated/flipped upright per its EXIF Orientation tag.

    Images without a usable tag are returned as the same object.
    """
    try:
        orientation = img.getexif().get(_EXIF_ORIENTATION_TAG)
    except Exception as e:
        print(f"Warning: Could not read EXIF orientation: {e}")
        return img
    method = _ORIENTATION_TRANSPOSE.get(orientation)
    if method is None:
        return img
    return img.transpose(method)

"""Shared utilities — font loading, image orientation, Tk helpers."""

import os
import sys
import contextlib

from PIL import Image, ExifTags

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

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


def _load_custom_fonts():
    global _CUSTOM_FONT_LOADED
    if not os.path.isdir(ASSETS_DIR):
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes
            FR_PRIVATE = 0x10
            gdi32 = ctypes.windll.gdi32
            for name in ("Archivo-Regular.ttf", "Archivo-Bold.ttf",
                         "Archivo-Medium.ttf", "Archivo-SemiBold.ttf"):
                path = os.path.join(ASSETS_DIR, name)
                if os.path.exists(path):
                    gdi32.AddFontResourceExW(path, FR_PRIVATE, 0)
            _CUSTOM_FONT_LOADED = True
            return True
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
                for name in ("Archivo-Regular.ttf", "Archivo-Bold.ttf",
                             "Archivo-Medium.ttf", "Archivo-SemiBold.ttf"):
                    path = os.path.join(ASSETS_DIR, name)
                    if os.path.exists(path):
                        url_ref = cf.CFURLCreateFromFileSystemRepresentation(
                            None, path.encode("utf-8"), len(path.encode("utf-8")), False)
                        if url_ref:
                            ct.CTFontManagerRegisterFontsForURL(url_ref, 1, None)
                _CUSTOM_FONT_LOADED = True
                return True
        except Exception:
            return False
    return False


def _get_font_family():
    import tkinter.font as tkFont
    if _CUSTOM_FONT_LOADED:
        return "Archivo"
    try:
        available = set(tkFont.families())
    except Exception:
        available = set()
    for family in ("Segoe UI", "Helvetica Neue", "Helvetica",
                   "Arial", "DejaVu Sans", "sans-serif"):
        if family in available:
            return family
    return "TkDefaultFont"


def auto_orient_image(img):
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        ops = {
            2: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda i: i.rotate(180, expand=True),
            4: lambda i: i.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda i: i.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT).rotate(270, expand=True),
            6: lambda i: i.rotate(270, expand=True),
            7: lambda i: i.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=True),
            8: lambda i: i.rotate(90, expand=True),
        }
        if orientation in ops:
            img = ops[orientation](img)
    except Exception as e:
        print(f"Warning: Could not auto-orient image: {e}")
    return img

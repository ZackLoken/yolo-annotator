"""YoloLabeler - lightweight YOLO annotation tool (detection + segmentation)."""

__version__ = "1.0.0"

__all__ = ["YoloLabeler", "main"]


def __getattr__(name):
    """Import gui.py only when the GUI is actually asked for.

    Importing it eagerly pulls Tk and CustomTkinter into every entry point under
    this package, including the headless yololabeler-import.
    """
    if name in __all__:
        from . import gui
        return getattr(gui, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)

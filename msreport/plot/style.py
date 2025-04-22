"""Manage and apply custom plotting styles for the `msreport.plot` module.

Active styles in msreport are predefined or customizable Matplotlib style sheets that
are automatically applied to all plots generated within the library. By changing the
active style, users can define the rcParams used for styling the plots, such as color
and font settings.

The `set_active_style` function allows users to select style sheets from the msreport
library or any style sheets available in Matplotlib. Additionally, it supports passing
a dictionary of rcParams to further customize the active style. The additional
parameters are applied after the style sheet, potentially overriding settings from the
style sheet.

Available msreport style sheets:
- "msreport-notebook"
- "seaborn-whitegrid
"""

import functools
import pathlib
from contextlib import contextmanager
from typing import Any

import matplotlib.style

__all__ = ["set_active_style"]


_STYLE_DIR: str = (pathlib.Path(__file__).parent / "style_sheets").resolve().as_posix()
_DEFAULT_STYLE: str = "msreport-notebook"
_active_style_name: str | None = _DEFAULT_STYLE
_active_style_rc_override: dict[str, Any] | None = None


def _get_library_styles() -> dict[str, str]:
    """Scan the style directory and returns a dict of available library styles.

    Returns:
        A dictionary mapping style names (without extension) to their full paths.
        Returns an empty dictionary if the style directory doesn't exist or is empty.
    """
    styles = {}
    try:
        for filepath in pathlib.Path(_STYLE_DIR).iterdir():
            if filepath.suffix == ".mplstyle":
                styles[filepath.stem] = filepath.resolve().as_posix()
    except OSError as err:
        raise OSError(
            f"Could not read 'msreport.plot' style directory {_STYLE_DIR}: {err}. "
            "Please check if the directory exists and is accessible."
        ) from err

    return styles


def _get_available_styles() -> list[str]:
    """Get a list of all available style names from library and matplotlib."""
    lib_styles = _get_library_styles().keys()
    mpl_styles = matplotlib.style.available
    return list(set(lib_styles) | set(mpl_styles))


_AVAILABLE_STYLES: list[str] = _get_available_styles()
_LIBRARY_STYLE_PATHS: dict[str, str] = _get_library_styles()


@contextmanager
def use_active_style():
    """Context manager to temporarily apply the active style for plotting."""
    active_style_context_arg = _get_active_style_context_arg()
    with matplotlib.rc_context():
        matplotlib.style.use(active_style_context_arg)
        yield


def with_active_style(func):
    """Decorator to apply the active style context to a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        active_style_context_arg = _get_active_style_context_arg()
        with matplotlib.rc_context():
            matplotlib.style.use(active_style_context_arg)
            return func(*args, **kwargs)

    return wrapper


def set_active_style(style: str | None, rc: dict[str, Any] | None = None):
    """Set the active plotting style for the msreport.plot submodule.

    The chosen style, potentially modified by the rc dictionary, will be
    applied temporarily using a context manager within the library's
    plotting functions. This does not modify the global matplotlib rcParams
    permanently.

    Args:
        style: The name of the base style to activate. This can be one of the
            built-in msreport styles (e.g., 'notebook', 'powerpoint'),
            a standard matplotlib style, or a style registered by another
            library like Seaborn (if available).
        rc: An optional dictionary mapping matplotlib rcParams names (strings)
            to their desired values. These settings will be applied *after*
            the base style, overriding any conflicting parameters from the
            base style for the duration of the plot context.

    Raises:
        ValueError: If the specified base style name is not found among the
            library's styles or the available matplotlib styles.
        TypeError: If rc is not a dictionary or None.
    """
    global _active_style_name, _active_style_rc_override

    if style is not None and style not in _AVAILABLE_STYLES:
        current_available = _get_available_styles()
        if style not in current_available:
            raise ValueError(
                f"Style '{style}' not found. Available styles are: "
                f"{', '.join(current_available)}"
            )

    if rc is not None and not isinstance(rc, dict):
        raise TypeError(f"rc argument must be a dictionary or None, got {type(rc)}")

    _active_style_name = style
    _active_style_rc_override = rc.copy() if rc is not None else None


def get_active_style() -> str | None:
    """Return the name of the currently active 'msreport.plot' plotting style."""
    return _active_style_name


def get_active_override() -> dict[str, Any] | None:
    """Return the currently active rcParam overrides for the 'msreport.plot' style."""
    return _active_style_rc_override


def _get_active_style_context_arg() -> list[str | dict[str, Any]]:
    """Get the argument needed for matplotlib.style.context for the active style.

    This combines the base style name/path with any active rcParam overrides.
    Matplotlib's style context manager can accept a list where later elements
    override earlier ones.

    Returns:
        A list containing the style name or path and any active rcParam overrides.
    """
    context_args: list[str | dict[str, Any]] = []

    active_style_name = get_active_style()
    if active_style_name is None:
        ...
    elif active_style_name in _LIBRARY_STYLE_PATHS:
        context_args.append(_LIBRARY_STYLE_PATHS[active_style_name])
    else:
        context_args.append(active_style_name)

    active_override = get_active_override()
    if active_override is not None:
        context_args.append(active_override)
    return context_args

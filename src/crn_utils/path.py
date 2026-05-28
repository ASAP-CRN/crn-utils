"""
Path utilities for ASAP CRN repositories.
"""
from pathlib import Path

__all__ = ["get_repo_root"]


def get_repo_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` until a directory containing `.git` is found.

    Parameters
    ----------
    start : Path or None
        Starting path to search upward from. Typically `Path(__file__)` of
        the calling module, so the function resolves the root of whichever
        repo the caller lives in. Defaults to `Path.cwd()` when None.

    Returns
    -------
    Path
        The nearest ancestor directory (inclusive of `start` itself if it is
        a directory) that contains a `.git` entry.

    Raises
    ------
    RuntimeError
        If no `.git` directory is found between `start` and the filesystem
        root.

    Examples
    --------
    Locate the root of the repo that contains the calling script, even if it's a different repo:

    >>> from pathlib import Path
    >>> from crn_utils.path import get_repo_root
    >>> root = get_repo_root(Path(__file__))

    Locate the root of the repo matching the current working directory:

    >>> root = get_repo_root()
    """

    anchor = (start if start is not None else Path.cwd()).resolve()
    for candidate in [anchor, *anchor.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate a git repository root above '{anchor}'. "
        "Pass an explicit `start` path if running outside a git repository."
    )

"""Functions to deal with different common pathname manipulations.

As Lezargus is going to be cross platform, this is a nice abstraction.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import glob
import os

from lezargus.library import logging


def get_directory(pathname: str) -> str:
    """Get the directory from the pathname without the file or the extension.

    Parameters
    ----------
    pathname : str
        The pathname which the directory will be extracted.

    Returns
    -------
    directory : str
        The directory which belongs to the pathname.

    """
    directory = os.path.dirname(pathname)
    return directory


def get_most_recent_filename_in_directory(
    directory: str,
    basename: str | list | None = None,
    extension: str | list | None = None,
    recursive: bool = False,
    recency_function: hint.Callable[[str], float] | None = None,
) -> str | None:
    """Get the most recent filename from a directory.

    Because of issues with different operating systems having differing
    issues with storing the creation time of a file, this function sorts based
    off of modification time unless a custom function is provided.

    Parameters
    ----------
    directory : str or list
        The directory by which the most recent file will be derived from.
    basename : str or list, default = None
        The basename filter which we will use to weed out the files. Wildcard
        expansion is supported. A list may be provided to cover multiple cases
        to include. If not provided, we default to all files.
    extension : str or list, default = None
        The extension by which to filter for. It is often the case that some
        files are created but the most recent file of some type is desired.
        Only files which match the included extensions will be considered.
    recursive : bool, default = False
        If True, the directory is searched recursively for the most recent file
        based on the recency function.
    recency_function : callable, default = None
        A function which, when provided, provides a sorting index for a given
        filename. This is used when the default sorting method (modification
        time) is not desired and a custom function can be provided here. The
        larger the value returned by this function, the more "recent" a
        given file will be considered to be.

    Returns
    -------
    recent_filename : str
        The filename of the most recent file, by modification time, in the
        directory. If no recent file is found, we return None.

    """
    # Check if the directory provided actually exists.
    if not os.path.isdir(directory):
        logging.critical(
            critical_type=logging.InputError,
            message=f"The provided directory does not exist: {directory}",
        )
    # The default basename filter and extension.
    basename = "*" if basename is None else basename
    extension = "*" if extension is None else extension

    # The default recency function, if not provided, is the modification times
    # of the files themselves.
    recency_function = (
        os.path.getmtime if recency_function is None else recency_function
    )

    # We need to get all of the valid pathnames which we can use to glob
    # to see the available files. We account for all permutations of the dial;g
    # different cases.
    # We also need to check if we accept recursive directories.
    directory = os.path.join(*[directory, "**"]) if recursive else directory
    basename_list = [basename] if isinstance(basename, str) else basename
    extension_list = [extension] if isinstance(extension, str) else extension
    # Finding the permutations.
    search_pathnames = [
        merge_pathname(directory=directory, filename=namedex, extension=extdex)
        for namedex in basename_list
        for extdex in extension_list
    ]

    # Now, based on the permutations, we try and find all of the valid
    # entries.
    matching_filenames = []
    for searchdex in search_pathnames:
        files = glob.glob(pathname=searchdex, recursive=recursive)
        matching_filenames = matching_filenames + files

    # We ought to check if there are any files which were even found in the
    # first place.
    if len(matching_filenames) == 0:
        # No files.
        logging.warning(
            warning_type=logging.FileWarning,
            message=(
                "No recent file found. No matching files found in directory:"
                f" {directory}"
            ),
        )
        return None

    # For all of the matching filenames, we need to find the most recent via
    # the modification time. Given that the modification times are a UNIX time,
    # the largest is the most recent.
    recent_filename = max(matching_filenames, key=recency_function)
    # Just a quick check to make sure the file exists.
    if not os.path.isfile(recent_filename):
        logging.error(
            error_type=logging.FileError,
            message=(
                "For some reason, the detected most recent file"
                f" `{recent_filename}` is not actually a typical file."
            ),
        )
    # All done.
    return recent_filename


def get_filename_without_extension(pathname: str) -> str:
    """Get the filename from the pathname without the file extension.

    Parameters
    ----------
    pathname : str
        The pathname which the filename will be extracted.

    Returns
    -------
    filename : str
        The filename without the file extension.

    """
    # In the event that there are more than one period in the full filename.
    # We only remove last one as is the conventions for extensions.
    file_components = os.path.basename(pathname).split(".")[:-1]
    filename = ".".join(file_components)
    return filename


def get_filename_with_extension(pathname: str) -> str:
    """Get the filename from the pathname with the file extension.

    Parameters
    ----------
    pathname : str
        The pathname which the filename will be extracted.

    Returns
    -------
    filename : str
        The filename with the file extension.

    """
    return os.path.basename(pathname)


def get_file_extension(pathname: str) -> str:
    """Get the file extension only from the pathname.

    Parameters
    ----------
    pathname : str
        The pathname which the file extension will be extracted.

    Returns
    -------
    extension : str
        The file extension only.

    """
    extension = os.path.basename(pathname).split(".")[-1]
    return extension


def merge_pathname(
    directory: hint.Union[str, list] | None = None,
    filename: str | None = None,
    extension: str | None = None,
) -> str:
    """Join the directories, filenames, and file extensions into one pathname.

    Parameters
    ----------
    directory : str or list, default = None
        The directory(s) which is going to be used. If it is a list,
        then the paths within it are combined.
    filename : str, default = None
        The filename that is going to be used for path construction.
    extension : str, default = None
        The filename extension that is going to be used.

    Returns
    -------
    pathname : str
        The combined pathname.

    """
    # Combine the directories if it is a list.
    directory = directory if directory is not None else ""
    directory = (
        directory if isinstance(directory, list | tuple) else [str(directory)]
    )
    total_directory = os.path.join(*directory)
    # Filename.
    filename = filename if filename is not None else ""
    # File extension.
    extension = extension if extension is not None else ""
    # Combining them into one path.
    if not extension:
        filename_extension = filename
    else:
        filename_extension = filename + "." + extension
    pathname = os.path.join(total_directory, filename_extension)
    return pathname


def split_pathname(pathname: str) -> tuple[str, str, str]:
    """Return a pathname split into its components.

    This is a wrapper function around the more elementary functions
    `get_directory`, `get_filename_without_extension`, and
    `get_file_extension`.

    Parameters
    ----------
    pathname : str
        The combined pathname which to be split.

    Returns
    -------
    directory : str
        The directory which was split from the pathname.
    filename : str
        The filename which was split from the pathname.
    extension : str
        The filename extension which was split from the pathname.

    """
    directory = get_directory(pathname=pathname)
    filename = get_filename_without_extension(pathname=pathname)
    extension = get_file_extension(pathname=pathname)
    return directory, filename, extension

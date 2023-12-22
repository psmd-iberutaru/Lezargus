"""Functions to deal with different common pathname manipulations.

As Lezargus is going to be cross platform, this is a nice abstraction.
"""

import glob
import os

from lezargus.library import hint
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
    extension: hint.Union[str, list] = None,
    recursive: bool = False,
    recency_function: hint.Callable[[str], float] = None,
) -> str:
    """Get the most recent filename from a directory.

    Because of issues with different operating systems having differing
    issues with storing the creation time of a file, this function sorts based
    off of modification time.

    Parameters
    ----------
    directory : str
        The directory by which the most recent file will be derived from.
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
        directory.
    """
    # Check if the directory provided actually exists.
    if not os.path.isdir(directory):
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The directory provided `{directory}` does not exist. A most"
                " recent file cannot be obtained."
            ),
        )

    # The default recency function, if not provided, is the modification times
    # of the files themselves.
    recency_function = (
        os.path.getmtime if recency_function is None else recency_function
    )

    # We need to check all of the files matching the provided extension. If
    # none was provided, we use all.
    extension = "*" if extension is None else extension
    extension_list = (
        (extension,) if isinstance(extension, str) else tuple(extension)
    )
    matching_filenames = []
    for extensiondex in extension_list:
        # If the extension has a leading dot, then we remove it as it
        # is already assumed.
        if extensiondex.startswith("."):
            clean_extension = extensiondex[1:]
        else:
            clean_extension = extensiondex
        # Fetch all of the matching files within the directory. We only want
        # files within the directory, not above or below unless recursive is
        # set
        directory_list = [directory, "**"] if recursive else [directory]
        pathname_glob_filter = merge_pathname(
            directory=directory_list,
            filename="*",
            extension=clean_extension,
        )
        extension_matching_files = glob.glob(
            pathname_glob_filter,
            recursive=recursive,
        )
        matching_filenames += extension_matching_files

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
    directory: hint.Union[str, list] = None,
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

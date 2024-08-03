"""Functions to handle the management of a temporary directory for files.

Some of the procedures that are done require the use of a temporary location
to write, read, and store files. The temporary directory is generally created
on initialization of this module. The user rarely would need access to this
function.
"""

import os
import shutil

import lezargus
from lezargus.library import logging


def create_temporary_directory(
    directory: str,
    flag_filename: str | None = None,
    overwrite: bool = False,
) -> None:
    """Create the temporary directory based on the directory path.

    We attempt to create the temporary directory. We check to make sure that
    it can be made and it does not already exist with files inside and we
    also add a file description to notify any passerby.

    Parameters
    ----------
    directory : str
        The directory path to make the temporary directory.
    flag_filename : str, default = None
        This is the filename of the file which serves the the description to
        inform people that the directory is temporary. If None, the file is not
        written.
    overwrite : bool, default = None
        If True, we "overwrite" the temporary directory. We do not actually
        clear any files, but we do not error when it exists.

    Returns
    -------
    None

    """
    # We determine the absolute path, just to be clear.
    temporary_directory = os.path.abspath(directory)

    # Next, we check if the directory exists. If there are any files, then
    # we warn about them not being safe.
    directory_exists = os.path.exists(temporary_directory)
    files_exist = (
        os.listdir(temporary_directory) != 0 if directory_exists else False
    )
    # We now work with the validity of the directories. If there are files
    # the temporary directory is not safe.
    if directory_exists and files_exist:
        if overwrite:
            # The user allows us to overwrite it, but we still give a warning.
            logging.warning(
                warning_type=logging.DataLossWarning,
                message=(
                    f"Proposed temporary directory {temporary_directory} is not"
                    " empty; files within it are not safe."
                ),
            )
        else:
            # Temporary directory has files and no instruction to overwrite,
            # continuing is a no-go to preserve the directory.
            logging.critical(
                critical_type=logging.DirectoryError,
                message=(
                    "The proposed temporary directory"
                    f" {temporary_directory} exists and is not empty, cannot"
                    " make one."
                ),
            )

    # Now we make the directory. We already made the checks above for an
    # already existing directory.
    os.makedirs(temporary_directory, exist_ok=True)

    # We add a file in there just to note that it is a temporary directory.
    if flag_filename is not None:
        flag_pathname = lezargus.library.path.merge_pathname(
            directory=temporary_directory,
            filename=flag_filename,
            extension="txt",
        )
        create_temporary_directory_file(filename=flag_pathname)

    # All done.


def delete_temporary_directory(
    directory: str,
    flag_filename: str | None = None,
    force: bool = False,
) -> None:
    """Delete the temporary directory based on the directory path.

    We attempt to delete the temporary directory. We make sure that it is a
    temporary directory based on the expected presence of the flag file. We
    stop if the flag file does not exist.

    Parameters
    ----------
    directory : str
        The directory path to make the temporary directory.
    flag_filename : str, default = None
        This is the filename of the file which serves the the description to
        inform people that the directory is temporary. If None, this check is
        skipped.
    force : bool, default = None
        Force the deletion of the directory and its contents regardless of
        the presence of the flag file.

    Returns
    -------
    None

    """
    # We determine the absolute path, just to be clear.
    temporary_directory = os.path.abspath(directory)

    # We do all of the checks here first, then sort through them later. This
    # is so we can sort through them for both warnings and the force flag.
    # ...checking if the directory already exists.
    directory_exists = os.path.isdir(temporary_directory)
    # ...checking the directory for the temporary directory flag file.
    if flag_filename is not None:
        flag_pathname = lezargus.library.path.merge_pathname(
            directory=temporary_directory,
            filename=flag_filename,
            extension="txt",
        )
        flag_exists = os.path.exists(flag_pathname)
    else:
        # We skip the check.
        flag_exists = True

    # Grouping the checks as a single flag.
    clean_delete = directory_exists and flag_exists

    # All of the checks are done computationally, now we just make sure we
    # can remove the directory. If we are forced to remove the directory, we
    # will but we will still warn.
    if clean_delete:
        # The deletion is considered clean.
        shutil.rmtree(temporary_directory)
    elif not clean_delete and force:
        # The deletion is not going to be clean, there are some issues
        # present. We still proceed anyways.
        logging.warning(
            warning_type=logging.DataLossWarning,
            message=(
                f"Temporary directory {temporary_directory} forcibly"
                " deleted, unclean removal."
            ),
        )
        shutil.rmtree(temporary_directory, ignore_errors=True)
    else:
        # The deletion is not clean and without an override we cannot proceed.
        logging.critical(
            critical_type=logging.DirectoryError,
            message=(
                f"Temporary directory {temporary_directory} cannot be"
                " cleanly deleted."
            ),
        )

    # All done.


def create_temporary_directory_file(filename: str) -> None:
    """Write the information for the temporary directory file.

    We just write a few lines informing the user of the temporary directory.
    Though we could store the raw text in data, it might complicate
    initialization so we just have a hard copy here.

    Parameters
    ----------
    filename : str
        The full filename of the temporary directory file which will be
        written. If the file already exists, we will overwrite it.

    Returns
    -------
    None

    """
    # The file contents.
    file_content = [
        R"This directory is a temporary directory for the Lezargus program.",
        R"Any manual modifications to this directory may be deleted without",
        R"warning at any time. To change the temporary directory path, see",
        R"the `LEZARGUS_TEMPORARY_DIRECTORY` configuration parameter.",
    ]
    # The line separators are not added by the write lines, so we add them
    # here.
    file_lines = [linedex + "\n" for linedex in file_content]
    # And writing the file. We default to UTF 8 encoding as it is becoming
    # more universal and it is ASCII compatible.
    with open(filename, "w+", encoding="utf_8") as file:
        file.writelines(file_lines)
    # All done.

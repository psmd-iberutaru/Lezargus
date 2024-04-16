"""This file contains tests test the Lezargus config handling code."""

import os

import conftest

import lezargus


def test_read_configuration_file():
    """Test the read_configuration_file function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We first test a failure of the file extension. It should not actually
    # fail, but a message should pop up.
    bad_filename = "config_filename_without_correct_extension.txt"
    try:
        __ = lezargus.library.config.read_configuration_file(
            filename=bad_filename,
        )
    except lezargus.library.logging.ElevatedError:
        # We ignore the elevated file error.
        pass

    # We next test if we can detect files which do not exist.
    no_filename = "config_filename_does_not_exist.yaml"
    try:
        __ = lezargus.library.config.read_configuration_file(
            filename=no_filename,
        )
    except lezargus.library.logging.FileError:
        # We ignore the elevated file error.
        pass

    # Finally we test a configuration file which does exist, but is not flat.
    nonflat_filename = "config_filename_not_flat_configuration.yaml"
    nonflat_filename = conftest.fetch_test_filename(basename=nonflat_filename)
    try:
        __ = lezargus.library.config.read_configuration_file(
            filename=nonflat_filename,
        )
    except lezargus.library.logging.ElevatedError:
        # We ignore the elevated file error.
        pass


def test_load_configuration_file() -> None:
    """Test the load_configuration_file function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We test to see that the casefold check works.
    # Not sure how to test this.

    # We test differing capitalization on a configuration file.
    lowercase_filename = "config_filename_with_lowercase_keys.yaml"
    lowercase_filename = conftest.fetch_test_filename(
        basename=lowercase_filename,
    )
    try:
        __ = lezargus.library.config.load_configuration_file(
            filename=lowercase_filename,
        )
    except lezargus.library.logging.ElevatedError:
        # We ignore the elevated file error.
        pass


def test_create_configuration_file() -> None:
    """Test the create_configuration_file function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We test the overwriting functionality. Borrowing a filename from other
    # tests.
    same_filename = "config_filename_with_lowercase_keys.yaml"
    same_filename = conftest.fetch_test_filename(basename=same_filename)
    try:
        __ = lezargus.library.config.create_configuration_file(
            filename=same_filename,
            overwrite=False,
        )
    except lezargus.library.logging.ElevatedError:
        # We ignore the elevated error of the file already existing.
        pass

    # We create two files with both a correct extension and one without.
    noext_filename = "config_filename_without_ext"
    ext_filename = noext_filename + ".yaml"
    # Fixing the path.
    noext_filename = conftest.fetch_test_filename(basename=noext_filename)
    ext_filename = conftest.fetch_test_filename(basename=ext_filename)
    __ = lezargus.library.config.create_configuration_file(
        filename=noext_filename,
        overwrite=True,
    )
    __ = lezargus.library.config.create_configuration_file(
        filename=ext_filename,
        overwrite=True,
    )
    # We don't want both files to actually stay so we can delete them.
    os.remove(ext_filename)

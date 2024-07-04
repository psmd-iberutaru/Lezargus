"""We set up a few things for the testing environment."""

import os

import pytest

import lezargus


def pytest_sessionstart(session: pytest.Session) -> None:
    """Called after the Session object has been created and
    before performing collection and entering the run test loop.

    Parameters
    ----------
    session : Session

    Returns
    -------
    None

    """
    # We need to load the new configuration file.
    test_directory = os.path.dirname(
        os.path.realpath(os.path.join(os.path.realpath(__file__))),
    )
    test_configuration_file = lezargus.library.path.merge_pathname(
        directory=test_directory,
        filename="test_configuration_overrides",
        extension="yaml",
    )
    lezargus.library.configuration.load_configuration_file(
        filename=test_configuration_file,
    )


# Here and below are some convenience functions.


def fetch_test_filename(basename: str) -> str:
    """Fetch the filename/pathname of a file which is needed for some tests.

    Parameters
    ----------
    basename : str
        The basename of the file, plus the extension to grab.

    Returns
    -------
    test_filename : str
        The test filename.

    """
    # We need to load the new configuration file.
    test_directory = os.path.dirname(
        os.path.realpath(os.path.join(os.path.realpath(__file__))),
    )
    test_file_directory = os.path.join(test_directory, "test_files", "")
    test_filename = test_file_directory + basename
    return test_filename

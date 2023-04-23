"""This file contains tests which are global to the project as a whole."""

import lezargus


def test_true() -> None:
    """This is a test that should always pass. This is just a default test
    to make sure tests runs.
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    # Always true test.
    assert_message = "This test should always pass."
    assert False, assert_message
    return None

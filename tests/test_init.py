"""This file contains tests which are global to the project as a whole."""

import lezargus

lezargus.library.logging.info(
    message=(
        "We are testing; the log messages within this file are often expected."
    ),
)


def test_true() -> None:
    """This is a test that should always pass.

    This is just a default test to make sure tests runs.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Always true test.
    assert_message = "This test should always pass."
    assert True, assert_message


def test_debug_flags() -> None:
    """This is a test to make sure that we do not ship with debug flags on.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # All of the debug flags must be off, no debug flag should be enabled
    # during formal packaging and shipping of a new version.

    # We get all of the names of the cache attributes to then clear.
    debug_prefix = "INTERNAL_DEBUG_"
    config_attributes = dir(lezargus.config)
    debug_attributes = [
        keydex
        for keydex in config_attributes
        if keydex.startswith(debug_prefix)
    ]

    # Checking to see if the debug flags are all off.
    for flagdex in debug_attributes:
        debug_flag_state = getattr(lezargus.config, flagdex)
        assert_message = f"Debug flag {flagdex} is True."
        assert not debug_flag_state, assert_message

    return None

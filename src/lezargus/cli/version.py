"""Command line Help function.

Execute: version ... ... ... ...

We just print version number.
"""

import lezargus
from lezargus.library import hint
from lezargus.library import logging

# isort: split

# We need to import the version number anyways.
from lezargus.__version__ import __version__


def version(parser: hint.ArgumentParser, arguments: dict) -> None:
    """Execute: `version ... ... ... ...`; the version number.

    Parameters
    ----------
    parser : ArgumentParser
        The argument parser which we are using.
    arguments : dict
        The parsed arguments from which the interpreted action will use. Note
        though that these arguments also has the interpreted actions.

    Returns
    -------
    None

    """
    # We still need the parser and arguments.
    lezargus.library.wrapper.do_nothing(parser, arguments)

    # Just print it.
    version_string = f"lezargus-{__version__}"
    logging.info(message=f"Version {version_string}.")

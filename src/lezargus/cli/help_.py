"""Command line Help function.

Execute: help ... ... ... ...

We just print the help function really.
"""

import lezargus
from lezargus.library import hint


def help_(parser: hint.ArgumentParser, arguments: dict) -> None:
    """Execute: `help ... ... ... ...`; the help function.

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
    # The arguments are needed, just in case.
    lezargus.library.wrapper.do_nothing(parser, arguments)
    # Doing the job.
    parser.print_help()

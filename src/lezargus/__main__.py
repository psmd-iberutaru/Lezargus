"""Just a small hook for the main execution.

This section parses arguments which is then passed to execution to do exactly
as expected by the commands.
"""

import argparse

import lezargus
import lezargus.terminate
from lezargus.library import hint


def parse_arguments() -> tuple[hint.ArgumentParser, dict]:
    """Parse all CLI arguments supplied.

    Parameters
    ----------
    None

    Returns
    -------
    parser : ArgumentParser
        The parser itself. This may not be needed for a lot of things, but
        it is still helpful for command processing.
    arguments : dict
        The arguments as parsed by the parser, we converted it to a dictionary.

    """
    # General description.
    parser = argparse.ArgumentParser(
        description=(
            "This is the command-line interface for Lezargus, see `lezargus"
            " help` for more information. For information on the available"
            " options for the CLI action chain, please see `list` and the main"
            " documentation."
        ),
        prefix_chars="-+",
    )

    # Adding positional arguments.
    parser.add_argument(
        "primary",
        action="store",
        nargs="?",
        default="help",
        help="The primary action in the CLI action chain.",
    )
    parser.add_argument(
        "secondary",
        action="store",
        nargs="?",
        default="",
        help="The secondary action in the CLI action chain.",
    )
    parser.add_argument(
        "tertiary",
        action="store",
        nargs="?",
        default="",
        help="The tertiary action in the CLI action chain.",
    )
    parser.add_argument(
        "quaternary",
        action="store",
        nargs="?",
        default="",
        help="The quaternary action in the CLI action chain.",
    )
    parser.add_argument(
        "quinary",
        action="store",
        nargs="?",
        default="",
        help="The quinary action in the CLI action chain.",
    )

    # Any and all optional arguments should be overrides to the configuration.
    parser.add_argument(
        "--options",
        action="store",
        required=False,
        default=None,
        help="Options?",
    )

    # Configuration override arguments are defined here. However, to make them
    # exactly special, we require the use of "+" as the prefix instead. This
    # allows us to not interfere with other optional parameters defined later.
    # We add the variables on-the-fly for ease.
    default_configuration = (
        lezargus.library.configuration.get_default_configuration()
    )
    for keydex, valuedex in default_configuration.items():
        # We allow both all upper and lowercase versions of the flag, just
        # for convenience.
        parser.add_argument(
            f"++{keydex.upper()}",
            f"++{keydex.lower()}",
            action="store",
            default=valuedex,
            required=False,
            help=argparse.SUPPRESS,
        )

    # Parsing the arguments.
    arguments = vars(parser.parse_args())
    # All done.
    return parser, arguments


def main() -> None:
    """Execute Lezargus.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Parse the arguments.
    parser, arguments = parse_arguments()

    # We apply any configuration overrides before executing any actions.
    # But only the configurations.
    default_configuration = (
        lezargus.library.configuration.get_default_configuration()
    )
    override_configuration = {
        keydex: arguments.get(keydex, valuedex)
        for keydex, valuedex in default_configuration.items()
    }
    lezargus.library.configuration.apply_configuration(
        configuration=override_configuration,
    )

    # We want to make sure we properly clean up after ourselves, even if
    # there is any errors.
    try:
        # Before we do anything, we need to initialize the module.
        lezargus.initialize.initialize()
        # And executing the arguments, starting with the primary action
        lezargus.cli.execute_primary_action(parser=parser, arguments=arguments)
    finally:
        lezargus.terminate.terminate()


if __name__ == "__main__":
    # Executing the actual functionality of this file.
    main()

"""Execute the primary CLI action here, going down the CLI chain where needed.

The CLI is built on a chain system, and we decide how to go down the chain
regarding the primary action here.

"""

import lezargus
from lezargus.library import hint
from lezargus.library import logging


def execute_primary_action(
    parser: hint.ArgumentParser,
    arguments: dict,
) -> None:
    """Execute the primary action.

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
    # We get the primary action.
    primary_action = str(arguments["primary"])
    primary_action = primary_action.casefold().strip()

    # We need to find the correct corse of action to execute for the primary
    # action.
    match primary_action:
        # Help function.
        case "help":
            lezargus.cli.help_.help_(parser=parser, arguments=arguments)
        case "version":
            lezargus.cli.version.version(parser=parser, arguments=arguments)

        # The user wants a list of available actions.
        case "list":
            available_actions = [
                "help",
                "version",
                # Other commands go above, in order.
                "list",
            ]
            logging.info(message=f"Available actions: {available_actions}")
        # No matching action.
        case _:
            logging.critical(
                critical_type=logging.CommandLineError,
                message=(
                    f"Unknown primary action: {primary_action}. Use `list` to"
                    " list available actions."
                ),
            )

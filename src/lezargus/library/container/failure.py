"""A container to use for when a bad or failed result is to be the return.

This container does not contain much data, but it represents an errored
return. Or, more specifically, a bad return for functions.
"""


class LezargusFailure:
    """Failure class to use when there is a failed return or similar.

    A function failure may result in an error, but it does not need to
    stop the program. Unfortunately, the usage of "None" as the failure state
    of programs conflicts with the usage of "None" as a the return for a void
    function.

    This class should therefore be used when there should be a return which
    indicates that a function, algorithm, or other routine failed. It serves
    as an indication for it in the program itself. The user should be notified
    by proper error and warning loggings.
    """

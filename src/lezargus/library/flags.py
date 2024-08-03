"""Functions to deal with quality flag tracking, masks, and other information.

Flags and masks is the primary backbone of how the quality of data can be
communicated. Here, we package all of the different functions regarding
flags and masks.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import numpy as np

import lezargus


def combine_masks(*masks: hint.NDArray) -> hint.NDArray:
    """Combine two or more masks.

    The masks follow the Numpy convention; a True value means that the data
    is considered masked.

    Parameters
    ----------
    *masks : ndarray
        The set of masks to combine.

    Returns
    -------
    combined_mask : ndarray
        The combined mask.

    """
    # We assume the shape of the first mask.
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    # Combining all of the masks.
    for maskdex in masks:
        combined_mask |= maskdex
    # All done.
    combined_mask = np.array(combined_mask, dtype=bool)
    return combined_mask


def combine_flags(*flags: hint.NDArray) -> hint.NDArray:
    """Combine two or more flag arrays.

    The flag values here follow the Lezargus convention, see [[TODO]].

    Parameters
    ----------
    *flags : ndarray
        The set of flags to combine.

    Returns
    -------
    combined_flags : ndarray
        The combined mask.

    """
    # We assume the shape of the first mask.
    combined_flags = np.zeros_like(flags[0], dtype=np.uint)
    # Combining all of the flags together.
    for maskdex in flags:
        combined_flags *= maskdex
    # All done.
    combined_flags = np.array(combined_flags, dtype=np.uint)
    return combined_flags


def reduce_flags(flag_array: hint.NDArray) -> hint.NDArray:
    """Reduce the flag value to the minimum it can be.

    Flags, based on the Lezargus convention (see [[TODO]]),
    rely on the prime factors to determine the total flags present. As
    multiplication is how flags propagate, the value can get big quickly.
    We reduce the values within a flag array to the lowest it can be.

    Parameters
    ----------
    flag_array : ndarray
        The flag array to be reduced into its lowest form.

    Returns
    -------
    lowest_flag_array : ndarray
        The flags, reduced to the lowest value.

    """
    # The lowest flag array.
    lowest_flag_array = np.ones_like(flag_array, dtype=np.uint)

    # If the flag value is wholely divisible by a prime number, then they have
    # that flag and we should record it.
    for primedex in lezargus.library.data.PRIME_FLAGS:
        flag_presence = (flag_array % primedex) == 0
        # For where it is present, we apply the flag.
        lowest_flag_array[flag_presence] *= primedex

    # All done.
    return lowest_flag_array

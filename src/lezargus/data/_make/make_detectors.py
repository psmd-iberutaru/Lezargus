"""Make functions to create the detectors and detector simulators.

This module contains functions which, provided a specific detector
specification, outputs a detector simulator.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import lezargus
from lezargus.library import logging


def make_detector() -> hint.DetectorSimulator:
    """Load a detector generator file to make the detector object.

    Note, the format of the detector generator file is
    very specific. User usage of the this function is discouraged.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the detector
        generator. The paths are handled automatically.

    Returns
    -------
    detector : DetectorSimulator
        The detector simulation object.

    """
    # TODO
    logging.error(
        error_type=logging.ToDoError,
        message=(
            "Detector creation in the data module not currently done. Default"
            " dummy detectors are used instead."
        ),
    )

    # We have not done anything here, so dummy detectors are used instead.
    detector = lezargus.library.container.DetectorArray(
        detector_shape=(2048, 2048),
        pixel_size=18 * 1e-6,
    )
    detector.recalculate_detector()
    return detector

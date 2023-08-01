"""Function wrappers.

We borrow a lot of functions from different packages; however, for a lot of 
them, we build wrappers around them to better integrate them into our 
package provided its own idiosyncrasies. 
"""

import numpy as np
import scipy.interpolate

from lezargus import library
from lezargus.library import logging
from lezargus.library import hint


def cubic_interpolate_1d_function(
    x: hint.Array, y: hint.Array
) -> hint.Callable[[hint.Array], hint.Array]:
    """Wrapper around Scipy's Cubic interpolation.

    Parameters
    ----------
    x : Array
        The x data to interpolate over.
    y : Array
        The y data to interpolate over.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # Create a cubic spline.
    cubic_interpolate_function = scipy.interpolate.CubicSpline(
        x=x, y=y, bc_type="not-a-knot", extrapolate=True
    )

    # Defining the wrapper function.
    def interpolate_1d_wrapper(input_data: hint.Array) -> hint.Array:
        """A cubic interpolator wrapper.

        Parameters
        ----------
        input_data : Array
            The input data.

        Returns
        -------
        output_data : Array
            The output data.
        """
        # We need to check if there is any interpolation.
        original_x = cubic_interpolate_function.x
        if not (
            (min(original_x) <= input_data) & (input_data <= max(original_x))
        ).all():
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Interpolating beyond original input domain, extrapolation"
                    " is used."
                ),
            )
        # Computing the interpolation.
        output_data = cubic_interpolate_function(input_data, nu=0)
        return output_data

    # All done, return the function itself.
    return interpolate_1d_wrapper

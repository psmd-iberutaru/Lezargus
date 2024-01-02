"""Interpolation routines, across both multi-dimensional and multi-mode.

We have many interpolation functions for a wide variety of use cases. We store
all of them here. We usually derive the more specialty interpolation functions
from a set of base functions.
"""

import numpy as np
import scipy.interpolate

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


def cubic_1d_interpolate_factory(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a wrapper function around Scipy's Cubic interpolation.

    We ignore NaN values for interpolation. Moreover, this function allows
    for extrapolation outside of the normal domain, though we still log it
    for debugging purposes.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # Clean up the data, removing anything that is not usable.
    clean_x, clean_y = library.array.clean_finite_arrays(x, y)

    # Create a cubic spline.
    cubic_interpolate_function = scipy.interpolate.CubicSpline(
        x=clean_x,
        y=clean_y,
        bc_type="not-a-knot",
        extrapolate=True,
    )

    # Defining the wrapper function.
    def interpolate_wrapper(input_data: hint.ndarray) -> hint.ndarray:
        """Cubic interpolator wrapper.

        Parameters
        ----------
        input_data : ndarray
            The input data.

        Returns
        -------
        output_data : ndarray
            The output data.
        """
        # We need to check if there is any extrapolation and just log it.
        original_x = cubic_interpolate_function.x
        if not (
            (min(original_x) <= input_data) & (input_data <= max(original_x))
        ).all():
            logging.debug(
                message=(
                    "Extrapolation of a cubic interpolator extrapolation"
                    " function attempted."
                ),
            )
        # Computing the interpolation.
        output_data = cubic_interpolate_function(input_data, nu=0)
        return output_data

    # All done, return the function itself.
    return interpolate_wrapper


def cubic_1d_interpolate_bounds_factory(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a wrapper function around Scipy's Cubic interpolation.

    We ignore NaN values for interpolation.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # We use the custom implementation, returning the cubic form of it.
    interpolate_function = custom_1d_interpolate_bounds_factory(
        interpolation=cubic_1d_interpolate_factory,
        x=x,
        y=y,
    )

    # All done, return the function itself.
    return interpolate_function


def cubic_1d_interpolate_gap_factory(
    x: hint.ndarray,
    y: hint.ndarray,
    gap_size: float | None = None,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around Scipy's Cubic interpolation, accounting for gaps.

    Regions which are considered to have a gap are not interpolated. Should a
    request for data within a gap region be called, we return NaN.
    We also ignore NaN values for interpolation.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.
    gap_size : float, default = None
        The maximum difference between two ordered x-coordinates before the
        region within the difference is considered to be a gap. If None,
        we assume that there are no gaps.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # We use the custom implementation, returning the cubic form of it.
    interpolate_wrapper = custom_1d_interpolate_gap_factory(
        interpolation=cubic_1d_interpolate_factory,
        x=x,
        y=y,
        gap_size=gap_size,
    )

    # All done, return the function itself.
    return interpolate_wrapper


def cubic_1d_interpolate_bounds_gap_factory(
    x: hint.ndarray,
    y: hint.ndarray,
    gap_size: float | None = None,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around Scipy's Cubic interpolation, accounting for gaps, bounds.

    Regions which are considered to have a gap are not interpolated. Should
    a request for data be outside of the input domain, we return NaN.
    Should a request for data within a gap region be called, we return NaN.
    We also ignore NaN values for interpolation.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.
    gap_size : float, default = None
        The maximum difference between two ordered x-coordinates before the
        region within the difference is considered to be a gap. If None,
        we assume that there are no gaps.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """

    # We use the custom implementation, returning the cubic form of it.
    # First the bounds wrapper, which we will need to wrap later.
    def bounds_wrapper(x: hint.ndarray, y: hint.ndarray) -> hint.ndarray:
        return custom_1d_interpolate_bounds_factory(
            interpolation=cubic_1d_interpolate_factory,
            x=x,
            y=y,
        )

    # Finally, the gap wrapper.
    interpolate_wrapper = custom_1d_interpolate_gap_factory(
        interpolation=bounds_wrapper,
        x=x,
        y=y,
        gap_size=gap_size,
    )

    # All done, return the function itself.
    return interpolate_wrapper


def nearest_neighbor_1d_interpolate_factory(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around Scipy's interp1d interpolation.

    This function exists so that in the event of the removal of Scipy's
    interp1d function, we only need to fix it once here.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # Clean up the data, removing anything that is not usable.
    clean_x, clean_y = library.array.clean_finite_arrays(x, y)
    # Create a cubic spline.
    nearest_neighbor_function = scipy.interpolate.interp1d(
        x=clean_x,
        y=clean_y,
        kind="nearest",
        bounds_error=False,
        fill_value=(clean_y[0], clean_y[-1]),
    )

    # Defining the wrapper function.
    def interpolate_wrapper(input_data: hint.ndarray) -> hint.ndarray:
        """Cubic interpolator wrapper.

        Parameters
        ----------
        input_data : ndarray
            The input data.

        Returns
        -------
        output_data : ndarray
            The output data.
        """
        # We need to check if there is any interpolation.
        original_x = nearest_neighbor_function.x
        if not (
            (min(original_x) <= input_data) & (input_data <= max(original_x))
        ).all():
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Interpolating beyond original input domain, padding"
                    " extrapolation is used."
                ),
            )
        # Computing the interpolation.
        output_data = nearest_neighbor_function(input_data)
        return output_data

    # All done, return the function itself.
    return interpolate_wrapper


def nearest_neighbor_1d_interpolate_bounds_factory(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around Scipy's interp1d interpolation.

    This function exists so that in the event of the removal of Scipy's
    interp1d function, we only need to fix it once here.

    Parameters
    ----------
    x : ndarray
        The x data to interpolate over.
    y : ndarray
        The y data to interpolate over.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # We use the custom implementation, returning the nearest neighbor form.
    interpolate_wrapper = custom_1d_interpolate_bounds_factory(
        interpolation=nearest_neighbor_1d_interpolate_factory,
        x=x,
        y=y,
    )

    # All done, return the function itself.
    return interpolate_wrapper


def custom_1d_interpolate_gap_factory(
    interpolation: hint.Callable[
        [hint.ndarray, hint.ndarray],
        hint.Callable[[hint.ndarray], hint.ndarray],
    ],
    x: hint.ndarray,
    y: hint.ndarray,
    gap_size: float | None = None,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around a custom interpolation, accounting for gaps.

    Regions which are considered to have a gap are not interpolated. Should a
    request for data within a gap region be called, we return NaN.
    We also ignore NaN values for interpolation.

    Parameters
    ----------
    interpolation : Callable
        The interpolation factory function which we are wrapping around.
    x : ndarray
        The x data to interpolate over. NaN values are ignored.
    y : ndarray
        The y data to interpolate over. NaN values are ignored.
    gap_size : float, default = None
        The maximum difference between two ordered x-coordinates before the
        region within the difference is considered to be a gap. If None,
        we assume that there are no gaps.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data.
    """
    # Defaults for the gap spacing limit. Note, if no gap is provided, there
    # really is no reason to be using this function.
    if gap_size is None:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Gap interpolation delta is None; consider using normal"
                " interpolation, it is strictly better."
            ),
        )
        gap_size = +np.inf
    else:
        gap_size = float(gap_size)

    # Clean up the data, removing anything that is not usable.
    clean_x, clean_y = library.array.clean_finite_arrays(x, y)
    sort_index = np.argsort(clean_x)
    sort_x = clean_x[sort_index]
    sort_y = clean_y[sort_index]

    # We next need to find where the bounds of the gap regions are, measuring
    # based on the gap delta criteria.
    x_delta = sort_x[1:] - sort_x[:-1]
    is_gap = x_delta > gap_size
    # And the bounds of each of the gaps.
    upper_gap = sort_x[1:][is_gap]
    lower_gap = clean_x[:-1][is_gap]

    # The basic cubic interpolator function.
    custom_interpolate_function = interpolation(sort_x, sort_y)
    # And we attach the gap limits to it so it can carry it. We use our
    # module name to avoid name conflicts with anything the Scipy project may
    # add in the future.
    custom_interpolate_function.lezargus_upper_gap = upper_gap
    custom_interpolate_function.lezargus_lower_gap = lower_gap

    # Defining the wrapper function.
    def interpolate_wrapper(input_data: hint.ndarray) -> hint.ndarray:
        """Cubic gap interpolator wrapper.

        Parameters
        ----------
        input_data : ndarray
            The input data.

        Returns
        -------
        output_data : ndarray
            The output data.
        """
        # We first interpolate the data.
        output_data = custom_interpolate_function(input_data)
        # And, we NaN out any points within the gaps of the domain of the data.
        for lowerdex, upperdex in zip(
            custom_interpolate_function.lezargus_lower_gap,
            custom_interpolate_function.lezargus_upper_gap,
            strict=True,
        ):
            # We NaN out points based on the input. We do not want to NaN the
            # actual bounds themselves however.
            output_data[(lowerdex < input_data) & (input_data < upperdex)] = (
                np.nan
            )
        # All done.
        return output_data

    # All done, return the function itself.
    return interpolate_wrapper


def custom_1d_interpolate_bounds_factory(
    interpolation: hint.Callable[
        [hint.ndarray, hint.ndarray],
        hint.Callable[[hint.ndarray], hint.ndarray],
    ],
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Wrap around a custom interpolation, accounting for bounds.

    We interpolate, based on the provided interpolation function. However,
    for any interpolation which is outside of the original domain boundary of
    the input data, we return NaN instead.

    Parameters
    ----------
    interpolation : Callable
        The interpolation factory function which we are wrapping around.
    x : ndarray
        The x data to interpolate over. NaN values are ignored.
    y : ndarray
        The y data to interpolate over. NaN values are ignored.

    Returns
    -------
    interpolate_function : Callable
        The interpolation function of the data, with the bounds handled.
    """
    # Clean up the data, removing anything that is not usable.
    clean_x, clean_y = library.array.clean_finite_arrays(x, y)

    # Create a cubic spline.
    custom_interpolate_function = interpolation(clean_x, clean_y)
    # The limits of the actual domain of the data, attached to the function.
    custom_interpolate_function.lezargus_lower_domain = np.nanmin(clean_x)
    custom_interpolate_function.lezargus_upper_domain = np.nanmax(clean_x)

    # Defining the wrapper function.
    def interpolate_wrapper(input_data: hint.ndarray) -> hint.ndarray:
        """Cubic interpolator wrapper.

        Parameters
        ----------
        input_data : ndarray
            The input data.

        Returns
        -------
        output_data : ndarray
            The output data.
        """
        # We need to check if where any of the input data exceeds the domain.
        min_domain = custom_interpolate_function.lezargus_lower_domain
        max_domain = custom_interpolate_function.lezargus_upper_domain
        in_bounds = (min_domain <= input_data) & (input_data <= max_domain)

        # Everything within the domain should be processed normally. Otherwise
        # we return NaN.
        output_data = np.empty_like(input_data, dtype=float)
        output_data[in_bounds] = custom_interpolate_function(
            input_data[in_bounds],
        )
        output_data[~in_bounds] = np.nan
        # All done.
        return output_data

    # All done, return the function itself.
    return interpolate_wrapper

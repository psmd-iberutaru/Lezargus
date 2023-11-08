"""Function wrappers.

We borrow a lot of functions from different packages; however, for a lot of
them, we build wrappers around them to better integrate them into our
package provided its own idiosyncrasies. Moreover, a lot of these wrapper
functions are either legacy or depreciated or otherwise overly-complex; and
as such, they may be changed in future builds so we unify all changes.
"""

import collections.abc

import astropy.modeling.models
import astropy.units
import numpy as np
import scipy.interpolate

from lezargus.library import hint
from lezargus.library import logging


def cubic_interpolate_1d_function(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a wrapper around Scipy's Cubic interpolation.

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
    # Clean up the data, removing anything that is not usable.
    clean_index = np.isfinite(x) & np.isfinite(y)
    clean_x = x[clean_index]
    clean_y = y[clean_index]
    # Create a cubic spline.
    cubic_interpolate_function = scipy.interpolate.CubicSpline(
        x=clean_x,
        y=clean_y,
        bc_type="not-a-knot",
        extrapolate=True,
    )

    # Defining the wrapper function.
    def interpolate_1d_wrapper(input_data: hint.ndarray) -> hint.ndarray:
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


def nearest_neighbor_interpolate_1d_function(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a wrapper around Scipy's interp1d interpolation.

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
    clean_index = np.isfinite(x) & np.isfinite(y)
    clean_x = x[clean_index]
    clean_y = y[clean_index]
    # Create a cubic spline.
    nearest_neighbor_function = scipy.interpolate.interp1d(
        x=clean_x,
        y=clean_y,
        kind="nearest",
        fill_value="extrapolate",
    )

    # Defining the wrapper function.
    def interpolate_1d_wrapper(input_data: hint.ndarray) -> hint.ndarray:
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
                    "Interpolating beyond original input domain, extrapolation"
                    " is used."
                ),
            )
        # Computing the interpolation.
        output_data = nearest_neighbor_function(input_data)
        return output_data

    # All done, return the function itself.
    return interpolate_1d_wrapper


def blackbody_function(
    temperature: float,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a callable blackbody function for a given temperature.

    This function is a wrapper around the Astropy blackbody model. This wrapper
    exists to remove the unit baggage of the original Astropy blackbody
    model so that we can stick to the convention of Lezargus.

    Parameters
    ----------
    temperature : float
        The blackbody temperature, in Kelvin.

    Returns
    -------
    blackbody : Callable
        The blackbody function, the wavelength callable is in microns. The
        return units are in FLAM/sr.
    """
    # The temperature, assigning units to them because that is what Astropy
    # wants.
    temperature_qty = astropy.units.Quantity(temperature, unit="Kelvin")
    flam_scale = astropy.units.Quantity(
        1,
        unit=astropy.units.erg
        / astropy.units.s
        / astropy.units.cm**2
        / astropy.units.AA
        / astropy.units.sr,
    )
    blackbody_instance = astropy.modeling.models.BlackBody(
        temperature=temperature_qty,
        scale=flam_scale,
    )

    def blackbody(wave: hint.ndarray) -> hint.ndarray:
        """Blackbody function.

        Parameters
        ----------
        wave : ndarray
            The wavelength of the input, in microns.

        Returns
        -------
        flux : ndarray
            The blackbody flux, as returned by a blackbody, in units of FLAM/sr.
        """
        wave = astropy.units.Quantity(wave, unit="micron")
        flux = blackbody_instance(wave)
        return flux.value

    # All done.
    return blackbody


def wavelength_overlap_fraction(
    base: hint.ndarray,
    contain: hint.ndarray,
) -> str:
    """Check if two wavelengths, defined as arrays, overlap.

    This is a function to check if the wavelength arrays overlap each other.
    Specifically, this checks if the contain wavelength array is within the
    base wavelength array, and if so, how much.

    Parameters
    ----------
    base : ndarray
        The base wavelength array which we are comparing the contain
        array against.
    contain : ndarray
        The wavelength array that we are seeing if it is within the base
        wavelength array.

    Returns
    -------
    fraction : float
        The fraction percent the two wavelength regions overlap with each
        other. This value may be larger than 1 for large overlaps.
    """
    # Getting the endpoints of the arrays.
    base_min = base.min()
    base_max = base.max()
    contain_min = contain.min()
    contain_max = contain.max()

    # First off, if the contain array is larger than the base array, by
    # default, it covers the base array, but, this sort of comparison does not
    # make much sense so we warn the user.
    if contain_min < base_min and base_max < contain_max:
        fraction = 1
        logging.warning(
            warning_type=logging.InputWarning,
            message=(
                "The contain array fully exceeds the base array, which is not"
                " the intention of the inputs. The inputs may need to be"
                " reversed."
            ),
        )
    # Second, we check if the contain array is fully within the base array.
    elif base_min <= contain_min and contain_max <= base_max:
        fraction = 1
    # Third, we check if the contain array is outside of the array on the
    # lower section. And, we check if the contain array is outside of the
    # array on the upper section.
    elif (contain_min <= base_min and contain_max <= base_min) or (
        base_max <= contain_min and base_max <= contain_max
    ):
        fraction = 0
    # Fourth, we check the case if the contain array exceeds the base array on
    # the lower section.
    elif contain_min <= base_min and contain_max <= base_max:
        # We compute the fractional percentage.
        fraction = (contain_max - base_min) / (base_max - base_min)
    # Fifth, we check the case if the contain array exceeds the base array on
    # the upper section.
    elif base_min <= contain_min and base_max <= contain_max:
        # We again compute the fractional percentage.
        fraction = (base_max - contain_min) / (base_max - base_min)
    # Whatever the case is here, is unknown.
    else:
        logging.error(
            error_type=logging.UndiscoveredError,
            message=(
                "This cases for the wavelength overlap fraction is not covered."
                " The domain of the base array is [{bl}, {bu}] and the contain"
                " array domain is [{cl}, {cu}.".format(
                    bl=base_min,
                    bu=base_max,
                    cl=contain_min,
                    cu=contain_max,
                )
            ),
        )
        fraction = 0

    return fraction


def combine_overlap_wavelength_array(
    *wavelengths: hint.ndarray,
) -> hint.ndarray:
    """Combine overlapping wavelengths, building on earlier bands.

    For more information, see [[TODO]].

    Parameters
    ----------
    *wavelengths : ndarray
        Positional arguments for the wavelength arrays we are combining. We
        use the first wavelength array provided and add points to it from
        areas in subsequent wavelength arrays that the original one does not
        cover.

    Returns
    -------
    combine_wavelength : ndarray
        The combined wavelength.
    """
    # We the first wavelength array is where we start from.
    combine_wavelength = wavelengths[0].tolist()
    for wavedex in wavelengths[1:]:
        # We add any points which falls outside of the current combine
        # wavelength range.
        min_wave = np.nanmin(combine_wavelength)
        max_wave = np.nanmax(combine_wavelength)
        # Adding points which are not within the current combine region.
        add_index = ~((min_wave <= wavedex) & (wavedex <= max_wave))
        combine_wavelength = combine_wavelength + wavedex[add_index].tolist()
    # All done.
    return np.sort(combine_wavelength)


def flatten_list_recursively(
    object_list: list[hint.ndarray | list],
) -> list[float]:
    """Flatten a list containing different sized numerical data.

    Parameters
    ----------
    object_list : list
        The object to flatten. Note, it must contain numerical data only.

    Returns
    -------
    flattened_list : list
        The list object, flattened.
    """
    # We do this recursively because how else to implement it is not really
    # known to Sparrow.
    flattened_list = []
    # Checking each entry of the input data.
    for itemdex in object_list:
        # If the entry is a number.
        if isinstance(itemdex, int | float | np.number):
            # Add the entry to the flattened list.
            flattened_list.append(float(itemdex))
            continue
        # We do a quick check if the object is iterable. We check using
        # this method first as it is likely quicker than catching errors.
        if isinstance(itemdex, collections.abc.Iterable):
            # Flatten out the subentry.
            inner_flat_list = flatten_list_recursively(object_list=itemdex)
            flattened_list = flattened_list + inner_flat_list
            continue
        # Sometimes the instance check is not enough. We use the expensive
        # iterable check.
        try:
            __ = iter(itemdex)
        except ValueError:
            # The object is not an iterable.
            flattened_list.append(float(itemdex))
            continue
        else:
            # The object is likely an iterable.
            inner_flat_list = flatten_list_recursively(object_list=itemdex)
            flattened_list = flattened_list + inner_flat_list
            continue

    # All done.
    return flattened_list

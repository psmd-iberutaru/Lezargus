"""Function wrappers.

We borrow a lot of functions from different packages; however, for a lot of
them, we build wrappers around them to better integrate them into our
package provided its own idiosyncrasies. Moreover, a lot of these wrapper
functions are either legacy or depreciated or otherwise overly-complex; and
as such, they may be changed in future builds so we unify all changes.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import collections.abc

import astropy.modeling.models
import astropy.units
import numpy as np

from lezargus.library import logging


def do_nothing(*args: hint.Any, **kwargs: hint.Any) -> None:
    """Do nothing, accepts arguments to prevent unused argument lint error.

    This function is just a fancy way of doing absolutely nothing. It serves
    as a way to "use" arguments for stub functions, templates, etc., so that
    the linter does not complain about such semantics.

    Parameters
    ----------
    *args : Any
        Positional arguments, which nothing will be done to them.
    **kwargs : Any
        The keyword arguments, which nothing will be done to them.

    Returns
    -------
    None

    """
    # We just do nothing.
    args = (None,)
    kwargs = {"None": None}
    __ = type(args)()
    __ = type(kwargs)()


def blackbody_function(
    temperature: float,
) -> hint.Callable[[hint.NDArray], hint.NDArray]:
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
        The blackbody function, the wavelength callable is in meters. The
        return units are in W m^-2 m^-1 sr^-1.

    """
    # The temperature, assigning units to them because that is what Astropy
    # wants.
    temperature_qty = astropy.units.Quantity(temperature, unit="Kelvin")
    si_scale = astropy.units.Quantity(
        1,
        unit=astropy.units.Unit("W m^-2 m^-1 sr^-1"),
    )

    blackbody_instance = astropy.modeling.models.BlackBody(
        temperature=temperature_qty,
        scale=si_scale,
    )

    def blackbody(wave: hint.NDArray) -> hint.NDArray:
        """Blackbody function.

        Parameters
        ----------
        wave : ndarray
            The wavelength of the input, in meters.

        Returns
        -------
        flux : ndarray
            The blackbody flux, as returned by a blackbody, in units of
            W m^-2 m^-1/sr.

        """
        wave = astropy.units.Quantity(wave, unit="meter")
        flux = blackbody_instance(wave).value
        return flux

    # All done.
    return blackbody


def wavelength_overlap_fraction(
    base: hint.NDArray,
    contain: hint.NDArray,
) -> float:
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
                "This cases for the wavelength overlap fraction is not"
                f" covered. The domain of the base array is [{base_min},"
                f" {base_max}] and the contain array domain is [{contain_min},"
                f" {contain_max}."
            ),
        )
        fraction = 0

    return fraction


def flatten_list_recursively(
    object_list: list[hint.NDArray | list],
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

"""Function wrappers.

We borrow a lot of functions from different packages; however, for a lot of
them, we build wrappers around them to better integrate them into our
package provided its own idiosyncrasies.
"""

import astropy.modeling.models
import astropy.units
import scipy.interpolate

from lezargus.library import hint
from lezargus.library import logging


def cubic_interpolate_1d_function(
    x: hint.ndarray,
    y: hint.ndarray,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Return a wrapper around Scipy's Cubic interpolation.

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
        x=x,
        y=y,
        bc_type="not-a-knot",
        extrapolate=True,
    )

    # Defining the wrapper function.
    def interpolate_1d_wrapper(input_data: hint.ndarray) -> hint.ndarray:
        """Cubic interpolator wrapper.

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
        wave : Array
            The wavelength of the input, in microns.

        Returns
        -------
        flux : Array
            The blackbody flux, as returned by a blackbody, in units of FLAM/sr.
        """
        wave = astropy.units.Quantity(wave, unit="micron")
        flux = blackbody_instance(wave)
        return flux.value

    # All done.
    return blackbody

"""Atmospheric functions and other operations.

This file keeps track of all of the functions and computations which deal
with the atmosphere.
"""

import astropy.modeling
import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


def airmass(zenith_angle: float | hint.ndarray) -> float | hint.ndarray:
    """Calculate the airmass from the zenith angle.

    This function calculates the airmass provided a zenith angle. For most
    cases the plane-parallel atmosphere method works, and it is what this
    function uses. However, we also use a more accurate formula for airmass at
    higher zenith angles (>80 degree), namely from DOI:10.1364/AO.28.004735.
    We use a weighted average between 75 < z < 80 degrees to allow for a
    smooth transition.

    Parameters
    ----------
    zenith_angle : float or ndarray
        The zenith angle, in radians.

    Returns
    -------
    airmass_ : float or ndarray
        The airmass. The variable name is to avoid name conflicts.
    """
    # The bounds of the spline region.
    low_spline_deg = 75
    high_spline_deg = 80

    # For the Kasten Young 1989 equation, we need the zenith angle in degrees.
    zenith_angle_degree = np.rad2deg(zenith_angle)

    # We either use the faster secant version for zenith angles.
    secant_airmass = 1 / np.cos(zenith_angle)
    kasten_young_airmass = 1 / (
        np.cos(zenith_angle)
        + 0.50572 * (6.07995 + 90 - zenith_angle_degree) ** (-1.6364)
    )
    # The two modes of calculation.
    airmass_ = np.where(
        zenith_angle_degree <= high_spline_deg,
        secant_airmass,
        kasten_young_airmass,
    )
    # Creating the average splice between the two regions.
    splice_index = (zenith_angle_degree >= low_spline_deg) & (
        zenith_angle_degree <= high_spline_deg
    )
    kasten_young_weights = (
        zenith_angle_degree[splice_index] - low_spline_deg
    ) / 5.0
    secant_weights = 1 - kasten_young_weights
    airmass_[splice_index] = (secant_airmass[splice_index] * secant_weights) + (
        kasten_young_airmass[splice_index] * kasten_young_weights
    )
    # All done.
    return airmass_


def index_of_refraction_ideal_air(wavelength: hint.ndarray) -> hint.ndarray:
    """Calculate the ideal refraction of air over wavelength.

    The index of refraction of air depends slightly on wavelength, we use
    the updated Edlen equations found in DOI: 10.1088/0026-1394/30/3/004.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in microns.

    Returns
    -------
    ior_ideal_air : ndarray
        The ideal air index of refraction.
    """
    # The wave number is actually used more in these equations.
    wavenumber = 1 / wavelength
    # Calculating the index of refraction, left hand then right hand side of
    # the equation.
    ior_ideal_air = (
        8342.54
        + 2406147 / (130 - wavenumber**2)
        + 15998 / (38.9 - wavenumber**2)
    )
    ior_ideal_air = ior_ideal_air / 1e8 + 1
    return ior_ideal_air


def index_of_refraction_dry_air(
    wavelength: hint.ndarray,
    pressure: float,
    temperature: float,
) -> hint.ndarray:
    """Calculate the refraction of air of pressured warm dry air.

    The index of refraction depends on wavelength, pressure and temperature, we
    use the updated Edlén equations found in DOI: 10.1088/0026-1394/30/3/004.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in microns.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    temperature : float
        The temperature of the atmosphere, in Kelvin.

    Returns
    -------
    ior_dry_air : ndarray
        The dry air index of refraction.
    """
    # We need the ideal air case first.
    ior_ideal_air = index_of_refraction_ideal_air(wavelength=wavelength)

    # The Edlén equations use Celsius as the temperature unit, we need to
    # convert from the standard Kelvin.
    temperature = temperature - 273.15
    if temperature < 0:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "The temperature specified for the Edlén equation for the index"
                " of refraction is lower than 0 C. The applicability is of this"
                " temperature is unknown."
            ),
        )

    # Calculating the pressure and temperature term.
    pt_factor = (pressure / 96095.43) * (
        (1 + pressure * (0.601 - 0.009723 * temperature) * 1e-8)
        / (1 + 0.003661 * temperature)
    )

    # Calculating the index of refraction of dry air.
    ior_dry_air = (ior_ideal_air - 1) * pt_factor
    ior_dry_air = ior_dry_air + 1
    return ior_dry_air


def index_of_refraction_moist_air(
    wavelength: hint.ndarray,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.ndarray:
    """Calculate the refraction of air of pressured warm moist air.

    The index of refraction depends on wavelength, pressure, temperature, and
    humidity, we use the updated Edlen equations found in
    DOI: 10.1088/0026-1394/30/3/004. We use the partial pressure of water in
    the atmosphere as opposed to actual humidity.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in microns.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    ior_moist_air : ndarray
        The moist air index of refraction.
    """
    # The wave number is actually used more in these equations.
    wavenumber = 1 / wavelength
    # We need the dry air case first.
    ior_dry_air = index_of_refraction_dry_air(
        wavelength=wavelength,
        pressure=pressure,
        temperature=temperature,
    )

    # Calculating the water vapor factor.
    wv_factor = (
        -1 * water_pressure * (3.7345 - 0.0401 * wavenumber**2) * 1e-10
    )

    # Computing the moist air index of refraction.
    ior_moist_air = ior_dry_air + wv_factor
    return ior_moist_air


def absolute_atmospheric_refraction_function(
    wavelength: hint.ndarray,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Compute the absolute atmospheric refraction function.

    The absolute atmospheric refraction is not as useful as the relative
    atmospheric refraction function. To calculate how the atmosphere refracts
    one's object, use that function instead.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in microns.
    zenith_angle : float
        The zenith angle of the sight line, in radians.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    abs_atm_refr_func : Callable
        The absolute atmospheric refraction function, as an actual callable
        function.
    """
    # We need to determine the index of refraction for moist air.
    ior_moist_air = index_of_refraction_moist_air(
        wavelength=wavelength,
        pressure=pressure,
        temperature=temperature,
        water_pressure=water_pressure,
    )

    # The constant of refraction.
    const_of_refr = (ior_moist_air**2 - 1) / (2 * ior_moist_air**2)
    # Incorporating the zenith angle.
    abs_atm_refr = const_of_refr * np.tan(zenith_angle)

    # Creating the function itself.
    abs_atm_refr_func = library.wrapper.cubic_interpolate_1d_function(
        x=wavelength,
        y=abs_atm_refr,
    )
    return abs_atm_refr_func


def relative_atmospheric_refraction_function(
    wavelength: hint.ndarray,
    reference_wavelength: float,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Compute the relative atmospheric refraction function.

    The relative refraction function is the same as the absolute refraction
    function, however, it is all relative to some specific wavelength.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in microns.
    reference_wavelength : float
        The reference wavelength which the relative refraction is computed
        against, in microns.
    zenith_angle : float
        The zenith angle of the sight line, in radians.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    rel_atm_refr_func : Callable
        The absolute atmospheric refraction function, as an actual callable
        function.
    """
    # We need the absolute refraction function first.
    abs_atm_refr_func = absolute_atmospheric_refraction_function(
        wavelength=wavelength,
        zenith_angle=zenith_angle,
        pressure=pressure,
        temperature=temperature,
        water_pressure=water_pressure,
    )

    # The refraction at the reference wavelength.
    ref_abs_refr = abs_atm_refr_func(reference_wavelength)

    def rel_atm_refr_func(wave: hint.ndarray) -> hint.ndarray:
        """Relative refraction function.

        Parameters
        ----------
        wave : ndarray
            The input wavelength for computation.

        Returns
        -------
        rel_atm_refr : ndarray
            The relative atmospheric refraction.
        """
        rel_atm_refr = abs_atm_refr_func(wave) - ref_abs_refr
        return rel_atm_refr

    # All done.
    return rel_atm_refr_func


def gaussian_psf_kernel(
    shape: tuple,
    x_stddev: float,
    y_stddev: float,
    rotation: float,
) -> hint.ndarray:
    """Return a 2D Gaussian point spread function convolution kernel.

    We normalize the point spread function via the amplitude of the Gaussian
    function as a whole for maximal precision: volume = 1. We require the
    input of the shape of the kernel to allow for `x_stddev` and `y_stddev`
    to be expressed in pixels to keep it general. By definition, the center
    of the Gaussian kernel is in the center of the array.

    Parameters
    ----------
    shape : tuple
        The shape of the 2D kernel, in pixels.
    x_stddev : float
        The standard deviation of the Gaussian in the x direction, in pixels.
    y_stddev : float
        The standard deviation of the Gaussian in the y direction, in pixels.
    rotation : float
        The rotation angle, increasing counterclockwise, in radians.

    Returns
    -------
    gaussian_kernel : ndarray
        The discrete kernel array.
    """
    # The center of the array given by the shape is defined as just the center
    # of it. However, we need to take into account off-by-one errors.
    nrow, ncol = shape
    cen_row = (nrow - 1) / 2
    cen_col = (ncol - 1) / 2

    # The normalization constant is provided as amplitude itself.
    norm_constant = 1 / (2 * np.pi * x_stddev * y_stddev)

    # The mesh grid used to evaluate the Gaussian function to derive the kernel.
    xx, yy = np.meshgrid(np.arange(ncol), np.arange(nrow))

    # Deriving the kernel and computing it.
    gaussian2d = astropy.modeling.models.Gaussian2D(
        amplitude=norm_constant,
        x_mean=cen_col,
        y_mean=cen_row,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=rotation,
    )
    gaussian_kernel = gaussian2d(xx, yy)
    return gaussian_kernel

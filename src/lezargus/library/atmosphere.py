"""Atmospheric functions and other operations.

This file keeps track of all of the functions and computations which deal
with the atmosphere. Note that seeing convolution and spectral convolution
is in the :py:mod:`lezargus.library.convolution` module.
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
from lezargus.library import logging


def airmass(zenith_angle: float | hint.NDArray) -> float | hint.NDArray:
    """Calculate the airmass from the zenith angle.

    This function calculates the airmass provided a zenith angle. We use a
    hybrid plane-parallel model and Young et. al. 1989 model to cover
    higher zenith angles. See [[TODO]] for more information.

    Parameters
    ----------
    zenith_angle : float or ndarray
        The zenith angle, in radians.

    Returns
    -------
    airmass_value : float or ndarray
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
    airmass_value = np.where(
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
    airmass_value[splice_index] = (
        secant_airmass[splice_index] * secant_weights
    ) + (kasten_young_airmass[splice_index] * kasten_young_weights)
    # All done.
    return airmass_value


def index_of_refraction_ideal_air(wavelength: hint.NDArray) -> hint.NDArray:
    """Calculate the ideal refraction of air over wavelength.

    The index of refraction of air depends slightly on wavelength, we use
    the updated Edlen equations; see [[TODO]].

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in meters.

    Returns
    -------
    ior_ideal_air : ndarray
        The ideal air index of refraction.

    """
    # The formal equation accepts only inverse micrometers, so we need to
    # convert. The wave number is actually used more in these equations.
    wavelength_um = lezargus.library.conversion.convert_units(
        value=wavelength,
        value_unit="m",
        result_unit="um",
    )
    wavenumber = 1 / wavelength_um
    # Calculating the index of refraction, left hand then right hand side of
    # the equation.
    ior_ideal_air_num = (
        8342.54
        + 2406147 / (130 - wavenumber**2)
        + 15998 / (38.9 - wavenumber**2)
    )
    ior_ideal_air = ior_ideal_air_num / 1e8 + 1
    return ior_ideal_air


def index_of_refraction_dry_air(
    wavelength: hint.NDArray,
    pressure: float,
    temperature: float,
) -> hint.NDArray:
    """Calculate the refraction of air of pressured warm dry air.

    The index of refraction depends on wavelength, pressure and temperature, we
    use the updated Edlén equations; see [[TODO]].

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in meters.
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
            warning_type=logging.AlgorithmWarning,
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
    wavelength: hint.NDArray,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.NDArray:
    """Calculate the refraction of air of pressured warm moist air.

    The index of refraction depends on wavelength, pressure, temperature, and
    humidity, we use the updated Edlen equations ; see [[TODO]].
    We use the partial pressure of water in the atmosphere as opposed to
    actual humidity.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength that we are calculating the index of refraction over.
        This must in meters.
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
    # We need the dry air case first.
    ior_dry_air = index_of_refraction_dry_air(
        wavelength=wavelength,
        pressure=pressure,
        temperature=temperature,
    )

    # The wave number is actually used more in these equations. However, the
    # wave number must be in inverse micrometers.
    wavelength_um = lezargus.library.conversion.convert_units(
        value=wavelength,
        value_unit="m",
        result_unit="um",
    )
    wavenumber = 1 / wavelength_um

    # Calculating the water vapor factor.
    wv_factor = -1 * water_pressure * (3.7345 - 0.0401 * wavenumber**2) * 1e-10

    # Computing the moist air index of refraction.
    ior_moist_air = ior_dry_air + wv_factor
    return ior_moist_air


def absolute_atmospheric_refraction(
    wavelength: hint.NDArray,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.NDArray:
    """Compute the absolute atmospheric refraction.

    The absolute atmospheric refraction is not as useful as the relative
    atmospheric refraction function. To calculate how the atmosphere refracts
    one's object, use that function: py:func:`relative_atmospheric_refraction`.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in meters.
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
    absolute_refraction : ndarray
        The computed absolute refraction at the input wavelengths, in radians.

    """
    # We need to determine the index of refraction for moist air.
    index_of_refraction = index_of_refraction_moist_air(
        wavelength=wavelength,
        pressure=pressure,
        temperature=temperature,
        water_pressure=water_pressure,
    )

    # The constant of refraction.
    constant_of_refraction = (index_of_refraction**2 - 1) / (
        2 * index_of_refraction**2
    )
    # Incorporating the zenith angle.
    absolute_refraction = constant_of_refraction * np.tan(zenith_angle)
    return absolute_refraction


def absolute_atmospheric_refraction_function(
    wavelength: hint.NDArray,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.Callable[[hint.NDArray], hint.NDArray]:
    """Compute the absolute atmospheric refraction function.

    The absolute atmospheric refraction is not as useful as the relative
    atmospheric refraction function. To calculate how the atmosphere refracts
    one's object, use that function instead.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in meters.
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
    refraction_function : Callable
        The absolute atmospheric refraction function, as an actual callable
        function. The input is wavelength in meters and output is refraction in
        radians.

    """
    # We compute the absolute refraction which we build an interpolating
    # function for.
    absolute_refraction = absolute_atmospheric_refraction(
        wavelength=wavelength,
        zenith_angle=zenith_angle,
        temperature=temperature,
        pressure=pressure,
        water_pressure=water_pressure,
    )

    # Creating the function itself.
    refraction_function = lezargus.library.interpolate.Spline1DInterpolate(
        x=wavelength,
        v=absolute_refraction,
        extrapolate=True,
    )

    return refraction_function


def relative_atmospheric_refraction(
    wavelength: hint.NDArray,
    reference_wavelength: float,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.NDArray:
    """Compute the relative atmospheric refraction.

    The relative atmospheric refraction is computed similarly to the
    absolute refraction, but is measured relative to the absolute refraction
    at the reference wavelength.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in meters.
    reference_wavelength : float
        The reference wavelength which the relative refraction is computed
        against, in meters.
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
    relative_refraction : ndarray
        The computed relative refraction at the input wavelengths, in radians.

    """
    # We need the absolute refraction function first. We use the functional
    # form so that we can determine a more accurate absolute refraction at the
    # provided wavelength.
    absolute_refraction_function = absolute_atmospheric_refraction_function(
        wavelength=wavelength,
        zenith_angle=zenith_angle,
        pressure=pressure,
        temperature=temperature,
        water_pressure=water_pressure,
    )

    # The absolute refraction, and the refraction at the reference wavelength.
    absolute_refraction = absolute_refraction_function(wavelength)
    reference_refraction = absolute_refraction_function(reference_wavelength)

    # Computing the refraction relative to the reference.
    relative_refraction = absolute_refraction - reference_refraction
    return relative_refraction


def relative_atmospheric_refraction_function(
    wavelength: hint.NDArray,
    reference_wavelength: float,
    zenith_angle: float,
    temperature: float,
    pressure: float,
    water_pressure: float,
) -> hint.Callable[[hint.NDArray], hint.NDArray]:
    """Compute the relative atmospheric refraction function.

    The relative refraction function is the same as the absolute refraction
    function, however, it is all relative to some specific wavelength.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength over which the absolute atmospheric refraction is
        being computed over, in meters.
    reference_wavelength : float
        The reference wavelength which the relative refraction is computed
        against, in meters.
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
    refraction_function : Callable
        The relative atmospheric refraction function, as an actual callable
        function. The input is wavelength in meters and output is refraction in
        radians.

    """
    # We compute the relative refraction which we build an interpolating
    # function for.
    relative_refraction = relative_atmospheric_refraction(
        wavelength=wavelength,
        reference_wavelength=reference_wavelength,
        zenith_angle=zenith_angle,
        temperature=temperature,
        pressure=pressure,
        water_pressure=water_pressure,
    )

    # Creating the function itself.
    refraction_function = lezargus.library.interpolate.Spline1DInterpolate(
        x=wavelength,
        v=relative_refraction,
        extrapolate=True,
    )

    return refraction_function


def seeing(
    wavelength: hint.NDArray,
    zenith_angle: float,
    reference_seeing: float,
    reference_wavelength: float,
    reference_zenith_angle: float = 0,
) -> hint.NDArray:
    """Compute seeing as a function of wavelength.

    The seeing, as a function of wavelength, is computed from wavelength and
    airmass ratios from some provided base reference seeing value. See
    [[TODO]] for more information.

    Parameters
    ----------
    wavelength : ndarray
        The wavelengths that we are calculating the seeing at, typically in
        meters.
    zenith_angle : float
        The zenith angle where we are calculating the seeing from, in radians.
    reference_seeing : float
        The provided reference seeing at the `reference_wavelength` and the
        `reference_zenith_angle`, in radians.
    reference_wavelength : float
        The reference wavelength where the reference seeing measurement
        `reference_seeing` is taken at. Must be in the same units as
        the `wavelength` parameter, typically meters.
    reference_zenith_angle : float, default = 0
        The reference zenith angle where the reference seeing measurement
        `reference_seeing` is taken at, in radians.

    Returns
    -------
    seeing_ : ndarray
        The seeing values as a function of wavelength, in the same units as
        the provided `reference_seeing`.

    """
    # The seeing ratios use airmass, not zenith angle. So we compute the
    # airmass from them.
    input_airmass = airmass(zenith_angle=zenith_angle)
    reference_airmass = airmass(zenith_angle=reference_zenith_angle)

    # First the relationship for wavelength.
    wavelength_relationship = (wavelength / reference_wavelength) ** (-1 / 5)

    # Second, the airmass relationship.
    airmass_relationship = (input_airmass / reference_airmass) ** (3 / 5)

    # Applying the ratio relationships.
    seeing_ = reference_seeing * wavelength_relationship * airmass_relationship

    # All done.
    return seeing_

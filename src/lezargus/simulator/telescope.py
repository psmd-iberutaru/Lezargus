"""Simulation code to simulate the telescope properties.

We simulate telescope effects, primarily the emission and reflectivity aspects
of it. We break this module up so that we can potentially simulate different
telescopes other than the IRTF. This is unlikely, but who knows.
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


class IrtfTelescopeSimulator:
    """The NASA IRTF telescope simulation class.

    Here we implement the effects of the primary and secondary mirror of the
    NASA IRTF telescope. Most focus is to the emissive and reflectivity effects
    of the mirrors, but other effects may also be simulated here.

    As the NASA IRTF telescope is a physical object, its specifications are
    determined in the configuration file and data files. We don't allow its
    mutability. Please raise an issue to request the addition of different
    telescopes.
    """

    temperature: float
    """The temperature of the primary and secondary mirrors. By default, this
    is the configured value."""

    _primary_reflectivity_interpolator: hint.Spline1DInterpolate | None = None
    """The interpolation class for the primary mirror reflectivity."""

    _secondary_reflectivity_interpolator: hint.Spline1DInterpolate | None = None
    """The interpolation class for the secondary mirror reflectivity."""

    def __init__(
        self: IrtfTelescopeSimulator,
        temperature: float | None = None,
    ) -> None:
        """Create an instance of the IRTF telescope.

        Parameters
        ----------
        temperature : float, default = None
            The temperature of the IRTF mirrors, used for the blackbody
            emission calculations. If None, we use the configured value instead.

        Returns
        -------
        None

        """
        # Temperature
        self.temperature = (
            lezargus.config.OBSERVATORY_IRTF_PRIMARY_TEMPERATURE
            if temperature is None
            else temperature
        )

        # By default, we use the default reflectivity data. The user
        # is free to change the interpolators as they wish.
        primary_wavelength = lezargus.data.EFFICIENCY_IRTF_PRIMARY.wavelength
        primary_reflectivity = lezargus.data.EFFICIENCY_IRTF_PRIMARY.data
        secondary_wavelength = (
            lezargus.data.EFFICIENCY_IRTF_SECONDARY.wavelength
        )
        secondary_reflectivity = lezargus.data.EFFICIENCY_IRTF_SECONDARY.data
        # Building the interpolators.
        self._primary_reflectivity_interpolator = (
            lezargus.library.interpolate.Spline1DInterpolate(
                x=primary_wavelength,
                v=primary_reflectivity,
                extrapolate=False,
            )
        )
        self._secondary_reflectivity_interpolator = (
            lezargus.library.interpolate.Spline1DInterpolate(
                x=secondary_wavelength,
                v=secondary_reflectivity,
                extrapolate=False,
            )
        )

    @property
    def primary_area(self: hint.Self) -> float:
        """Area of the primary mirror.

        Parameters
        ----------
        None

        Returns
        -------
        area : float
            The area of the primary mirror.

        """
        # We determine it from the configured radius.
        primary_radius = lezargus.config.OBSERVATORY_IRTF_PRIMARY_MIRROR_RADIUS
        area = np.pi * primary_radius**2
        return area

    # Alias for the primary mirror area.
    telescope_area = primary_area

    @property
    def secondary_area(self: hint.Self) -> float:
        """Area of the secondary mirror.

        Parameters
        ----------
        None

        Returns
        -------
        area : float
            The area of the secondary mirror.

        """
        # We determine it from the configured radius.
        secondary_radius = (
            lezargus.config.OBSERVATORY_IRTF_SECONDARY_MIRROR_RADIUS
        )
        area = np.pi * secondary_radius**2
        return area

    def primary_reflectivity(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray | None:
        """Compute the reflectivity of the IRTF primary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            reflectivity, in meters.

        Returns
        -------
        reflectivity : NDArray
            The reflectivity of the primary mirror at the wavelengths provided.

        """
        # If the primary reflectivity interpolator does not exist, we cannot
        # really give any values.
        if self._primary_reflectivity_interpolator is None:
            logging.error(
                error_type=logging.ConfigurationError,
                message=(
                    "The internal primary mirror reflectivity interpolator"
                    " does not exist. One must be provided."
                ),
            )
            return None

        # Otherwise, we can just interpolate as normal.
        raw_reflectivity = self._primary_reflectivity_interpolator(wavelength)
        # Reflectivity cannot be less than zero. The interpolator can do this
        # at times so for those negative values, we assume zero.
        reflectivity = np.where(raw_reflectivity >= 0, raw_reflectivity, 0)

        # All done.
        return reflectivity

    def primary_emission(
        self: hint.Self,
        wavelength: hint.NDArray,
        solid_angle: float,
    ) -> hint.NDArray | None:
        """Compute the spectral flux emission of the IRTF primary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            spectral flux emission, in meters.
        solid_angle : float
            The total solid angle that the primary emission is integrating
            over. This is needed for the blackbody emission integration.

        Returns
        -------
        emission : NDArray
            The spectral flux emission of the primary mirror at the
            wavelengths provided.

        """
        # We assume a blackbody emission function.
        primary_blackbody = lezargus.library.wrapper.blackbody_function(
            temperature=self.temperature,
        )
        primary_blackbody_radiance = primary_blackbody(wavelength)

        # The blackbody is modulated by...
        # ...the primary's own efficiency,
        reflectivity = self.primary_reflectivity(wavelength=wavelength)
        if reflectivity is None:
            logging.critical(
                critical_type=logging.WrongOrderError,
                message="Primary mirror reflectivity function missing.",
            )
            return None
        emission_efficiency = 1 - reflectivity
        # ...the area of the primary mirror,
        primary_area = self.primary_area
        # ...and the integrating solid angle. Though, this is custom provided.
        solid_angle = float(solid_angle)

        # Performing the "integration".
        emission = (emission_efficiency * primary_blackbody_radiance) * (
            primary_area * solid_angle
        )
        return emission

    def secondary_reflectivity(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray | None:
        """Compute the reflectivity of the IRTF secondary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the secondary mirror
            reflectivity, in meters.

        Returns
        -------
        reflectivity : NDArray
            The reflectivity of the secondary mirror at the wavelengths
            provided.

        """
        # If the secondary reflectivity interpolator does not exist, we cannot
        # really give any values.
        if self._secondary_reflectivity_interpolator is None:
            logging.error(
                error_type=logging.ConfigurationError,
                message=(
                    "The internal secondary mirror reflectivity interpolator"
                    " does not exist. One must be provided."
                ),
            )
            return None

        # Otherwise, we can just interpolate as normal.
        raw_reflectivity = self._secondary_reflectivity_interpolator(wavelength)
        # Reflectivity cannot be less than zero. The interpolator can do this
        # at times so for those negative values, we assume zero.
        reflectivity = np.where(raw_reflectivity >= 0, raw_reflectivity, 0)

        # All done.
        return reflectivity

    def secondary_emission(
        self: hint.Self,
        wavelength: hint.NDArray,
        solid_angle: float,
    ) -> hint.NDArray | None:
        """Compute the spectral flux emission of the IRTF secondary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the secondary mirror
            spectral flux emission, in meters.
        solid_angle : float
            The total solid angle that the secondary emission is integrating
            over. This is needed for the blackbody emission integration.

        Returns
        -------
        emission : NDArray
            The spectral flux emission of the secondary mirror at the
            wavelengths provided.

        """
        # We assume a blackbody emission function.
        secondary_blackbody = lezargus.library.wrapper.blackbody_function(
            temperature=self.temperature,
        )
        secondary_blackbody_radiance = secondary_blackbody(wavelength)

        # The blackbody is modulated by...
        # ...the primary's own efficiency,
        reflectivity = self.secondary_reflectivity(wavelength=wavelength)
        if reflectivity is None:
            logging.critical(
                critical_type=logging.WrongOrderError,
                message="Secondary mirror reflectivity function missing.",
            )
            return None
        emission_efficiency = 1 - reflectivity
        # ...the area of the secondary mirror,
        secondary_area = self.secondary_area
        # ...and the integrating solid angle. Though, this is custom provided.
        solid_angle = float(solid_angle)

        # Performing the "integration".
        emission = (emission_efficiency * secondary_blackbody_radiance) * (
            secondary_area * solid_angle
        )
        return emission

    def primary_reflectivity_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.LezargusSpectrum:
        """Compute the primary reflectivity, as a LezargusSpectrum.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            reflectivity, in meters.

        Returns
        -------
        reflectivity_spectrum : LezargusSpectrum
            The reflectivity spectrum of the primary mirror at the
            wavelengths provided.

        """
        # We just package it per usual.
        reflectivity_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=self.primary_reflectivity(wavelength=wavelength),
            uncertainty=None,
            wavelength_unit="m",
            data_unit="",
            spectral_scale=None,
            pixel_scale=None,
            slice_scale=None,
            mask=None,
            flags=None,
            header=None,
        )
        return reflectivity_spectrum

    def primary_emission_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
        solid_angle: float,
    ) -> hint.LezargusSpectrum:
        """Compute the emission of the IRTF primary mirror, as a spectrum.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            spectral flux emission, in meters.
        solid_angle : float
            The total solid angle that the primary emission is integrating
            over. This is needed for the blackbody emission integration.

        Returns
        -------
        emission_spectrum : NDArray
            The spectral flux emission of the primary mirror at the
            wavelengths provided.

        """
        # We just package it per usual.
        emission_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=self.primary_emission(
                wavelength=wavelength,
                solid_angle=solid_angle,
            ),
            uncertainty=None,
            wavelength_unit="m",
            data_unit="W m^-2 m^-1",
            spectral_scale=None,
            pixel_scale=None,
            slice_scale=None,
            mask=None,
            flags=None,
            header=None,
        )
        return emission_spectrum

    def secondary_reflectivity_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.LezargusSpectrum:
        """Compute the secondary reflectivity, as a LezargusSpectrum.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the secondary mirror
            reflectivity, in meters.

        Returns
        -------
        reflectivity_spectrum : LezargusSpectrum
            The reflectivity spectrum of the secondary mirror at the
            wavelengths provided.

        """
        # We just package it per usual.
        reflectivity_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=self.secondary_reflectivity(wavelength=wavelength),
            uncertainty=None,
            wavelength_unit="m",
            data_unit="",
            spectral_scale=None,
            pixel_scale=None,
            slice_scale=None,
            mask=None,
            flags=None,
            header=None,
        )
        return reflectivity_spectrum

    def secondary_emission_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
        solid_angle: float,
    ) -> hint.LezargusSpectrum:
        """Compute the emission of the IRTF secondary mirror, as a spectrum.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the secondary mirror
            spectral flux emission, in meters.
        solid_angle : float
            The total solid angle that the secondary emission is integrating
            over. This is needed for the blackbody emission integration.

        Returns
        -------
        emission_spectrum : NDArray
            The spectral flux emission of the secondary mirror at the
            wavelengths provided.

        """
        # We just package it per usual.
        emission_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=self.secondary_emission(
                wavelength=wavelength,
                solid_angle=solid_angle,
            ),
            uncertainty=None,
            wavelength_unit="m",
            data_unit="W m^-2 m^-1",
            spectral_scale=None,
            pixel_scale=None,
            slice_scale=None,
            mask=None,
            flags=None,
            header=None,
        )
        return emission_spectrum

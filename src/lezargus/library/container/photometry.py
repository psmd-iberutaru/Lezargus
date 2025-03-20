"""Photometry filter wrapper class implementation.

An implementation of photometric filters, and all associated functionality
is described here. This allows for maximum code reuse when creating all of the
filters needed for Lezargus. There are two main types, Vega-based and AB-based.

Moreover, we handle both of the cases for energy-based and photon-counting
based forms of photometric filters. See the documentation
:ref:`technical-photometry`
for more information on energy-based transmission versus photon-counting
based forms for more information.
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


class PhotometricABFilter:
    """AB/ST system based photometric filter."""


class PhotometricVegaFilter:
    """Vega system based photometric filter.

    Most of the attributes listed here are the internal ones often used for
    caching and the like.
    """

    wavelength: hint.NDArray
    """The wavelengths of the energy-based and photon-based
    transmissions for the filter."""

    wavelength_unit: hint.Unit
    """The unit of the wavelength. For Lezargus, this is usually
    meters"""

    transmission_energy: hint.NDArray
    """The energy-based transmission of the filter at the
    wavelengths provided."""

    standard_spectrum: hint.LezargusSpectrum
    """The standard star which is the effective dagger
    standard for this filter. Typically, the standard spectra is a Vega spectra
    but this may change."""

    standard_magnitude: float
    """The magnitude of the standard star in this filter."""

    standard_magnitude_uncertainty: float
    """The uncertainty of the magnitude of the standard star in this filter."""

    zero_point: float
    """The filter zero point value of the filter."""

    zero_point_uncertainty: float
    """The uncertainty in the filter zero point value of the filter."""

    def __init__(
        self: PhotometricVegaFilter,
        wavelength: hint.NDArray,
        transmission: hint.NDArray,
        wavelength_unit: hint.Unit | str = "m",
    ) -> None:
        """Create an instance of a Vega-based photometric filter.

        It may be better to use the class methods
        :py:meth:`from_energy_transmission` and
        :py:meth:`from_photon_transmission` instead.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength specification of the photometric filter.
        transmission : NDArray
            The filter's energy-based photometric filter transmission.
        wavelength_unit : Unit or string, default = "m"
            The unit of the wavelength array.

        Returns
        -------
        None

        """
        # By convention, the transmissions are normalized to the maximum.
        normalized_transmission_energy = transmission / np.nanmax(transmission)

        # We parse the unit.
        wavelength_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=wavelength_unit,
        )

        # We add the wavelength and transmission to the filter.
        self.wavelength = wavelength
        self.wavelength_unit = wavelength_unit
        self.transmission_energy = normalized_transmission_energy

    @classmethod
    def from_energy_transmission(
        cls: type[hint.Self],
        wavelength: hint.NDArray,
        energy_transmission: hint.NDArray,
        wavelength_unit: hint.Unit | str = "m",
    ) -> hint.Self:
        """Create an instance of the filter via energy transmission.

        This function, in essence, is the same as the main initialization
        function. However, we have it here for consistency.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength specification of the photometric filter.
        energy_transmission : NDArray
            The filter's energy-based photometric filter transmission.
        wavelength_unit : Unit or string, default = "m"
            The unit of the wavelength array.

        Returns
        -------
        None

        """
        return cls(
            wavelength=wavelength,
            transmission=energy_transmission,
            wavelength_unit=wavelength_unit,
        )

    @classmethod
    def from_photon_transmission(
        cls: type[hint.Self],
        wavelength: hint.NDArray,
        photon_transmission: hint.NDArray,
        wavelength_unit: hint.Unit | str = "m",
    ) -> hint.Self:
        """Create an instance of the filter via photon transmission.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength specification of the photometric filter.
        photon_transmission : NDArray
            The filter's photon-based photometric filter transmission.
        wavelength_unit : Unit or string, default = "m"
            The unit of the wavelength array.

        Returns
        -------
        None

        """
        # We need to convert the photon transmission to an energy based
        # transmission.
        energy_transmission = photon_transmission / wavelength
        return cls.from_energy_transmission(
            wavelength=wavelength,
            energy_transmission=energy_transmission,
            wavelength_unit=wavelength_unit,
        )

    @property
    def transmission_photon(self: hint.Self) -> hint.NDArray:
        """Compute the photon-based transmission of the filter.

        The photon transmission of the filter is computed from the energy
        transmission.

        Parameters
        ----------
        None

        Returns
        -------
        transmission : NDArray
            The photon-based transmission of the filter.

        """
        transmission_photon = self.transmission_energy * self.wavelength
        # By convention, the transmissions are normalized to the maximum.
        transmission = transmission_photon / np.nanmax(transmission_photon)
        return transmission

    def transmission_energy_function(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """Functional form of the energy-based transmission.

        The functional form is just an interpolation of the current values.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength which we will compute the transmission at.

        Returns
        -------
        transmission : Spline1DInterpolate
            The energy-based transmission computed at the provided wavelength.

        """
        # We just make the interpolator. We assume that outside values of a
        # filter are 0.
        transmission_interpolator = (
            lezargus.library.interpolate.Spline1DInterpolate(
                x=self.wavelength,
                v=self.transmission_energy,
                extrapolate=False,
                extrapolate_fill=0,
            )
        )
        transmission = transmission_interpolator(x=wavelength)
        return transmission

    def transmission_photon_function(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """Functional form of the photon-based transmission.

        The functional form is just an interpolation of the current values.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength which we will compute the transmission at.

        Returns
        -------
        transmission : Spline1DInterpolate
            The photon-based transmission computed at the provided wavelength.

        """
        # We just make the interpolator. We assume that outside values of a
        # filter are 0.
        transmission_interpolator = (
            lezargus.library.interpolate.Spline1DInterpolate(
                x=self.wavelength,
                v=self.transmission_photon,
                extrapolate=False,
                extrapolate_fill=0,
            )
        )
        transmission = transmission_interpolator(x=wavelength)
        return transmission

    def add_standard_star_spectrum(
        self: hint.Self,
        spectrum: hint.LezargusSpectrum,
        magnitude: float,
        magnitude_uncertainty: float,
    ) -> None:
        """Add a standard star spectrum to characterize the filter system.

        A Vega-like standard star, with the filter magnitude (and the
        uncertainty thereof) characterizes the filter system so that it can be
        used to determine synthetic magnitudes and scaling factors.

        Parameters
        ----------
        spectrum : LezargusSpectrum
            The spectrum of the calibrated standard star. As this is a
            Vega-based filter, the standard star should be appropriate to a
            Vega system.
        magnitude : float
            The magnitude of the standard star for this filter.
        magnitude_uncertainty : float
            The uncertainty of the magnitude of the standard star in this
            filter.

        Returns
        -------
        None

        """
        # Just making sure the spectrum is a LezargusSpectrum.
        if not isinstance(
            spectrum,
            lezargus.library.container.LezargusSpectrum,
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input standard spectrum is not a LezargusSpectrum but is"
                    f" instead: {type(spectrum)}."
                ),
            )

        # Adding the new standard.
        self.standard_spectrum = spectrum
        self.standard_magnitude = magnitude
        self.standard_magnitude_uncertainty = magnitude_uncertainty

        # We can calculate the zero point with our new standard.
        zero_point, zero_point_uncertainty = self.__calculate_zero_point(
            standard_spectrum=self.standard_spectrum,
            standard_magnitude=self.standard_magnitude,
            standard_magnitude_uncertainty=self.standard_magnitude_uncertainty,
        )
        # Applying it.
        self.zero_point = zero_point
        self.zero_point_uncertainty = zero_point_uncertainty

        # All done.

    def __calculate_zero_point(
        self: hint.Self,
        standard_spectrum: hint.LezargusSpectrum,
        standard_magnitude: float,
        standard_magnitude_uncertainty: float = 0,
    ) -> tuple[float, float]:
        """Calculate the magnitude zero point of this filter in the Vega system.

        This function computes Vega zero points by integrating over the
        filter band pass based on a provided standard spectrum.

        Parameters
        ----------
        standard_spectrum : LezargusSpectrum
            The standard star, as saved by a spectrum container instance.
        standard_magnitude : float
            The magnitude of the standard star in that filter.
        standard_magnitude_uncertainty : float, default = 0
            The uncertainty in the magnitude of the standard star in that
            filter. Often this is not needed because the magnitude value of
            standard defines the filter system anyways and so, by definition,
            there is no uncertainty; but some systems do want it.

        Returns
        -------
        zero_point : float
            The zero point value.
        zero_point_uncertainty : float
            The uncertainty on the zero point. If the standard star provided
            has some uncertainty, or if the filter does, then we attempt to
            calculate the uncertainty on the value.

        """
        # We assume that the standard star has data covering the entire filter
        # range as otherwise, you cannot really calculate an accurate zero
        # point. However, we should still warn otherwise.
        overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
            base=standard_spectrum.wavelength,
            contain=self.wavelength,
        )
        if overlap < 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "A synthetic magnitude is being computed where the star"
                    " spectra does not cover the entire filter. The overlap"
                    f" fraction is {overlap}"
                ),
            )

        # Computing the total filter-weighted flux, using usual integration
        # methods.
        star_flux, star_uncertainty, __, __ = standard_spectrum.interpolate(
            wavelength=self.wavelength,
            skip_mask=True,
            skip_flags=True,
        )
        # Integrating...
        (
            star_filter_flux,
            star_filter_flux_uncertainty,
        ) = lezargus.library.math.multiply(
            multiplier=star_flux,
            multiplicand=self.transmission_photon,
            multiplier_uncertainty=star_uncertainty,
            multiplicand_uncertainty=0,
        )
        (
            star_filter_integral,
            star_filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=star_filter_flux,
            integrand_uncertainty=star_filter_flux_uncertainty,
        )
        (
            filter_integral,
            filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=self.transmission_photon,
            integrand_uncertainty=0,
        )

        # The zero point itself. Going through the equation with propagation.
        _frac, _frac_uncert = lezargus.library.math.divide(
            numerator=star_filter_integral,
            denominator=filter_integral,
            numerator_uncertainty=star_filter_integral_uncertainty,
            denominator_uncertainty=filter_integral_uncertainty,
        )
        _log, _log_uncert = lezargus.library.math.logarithm(
            antilogarithm=_frac,
            base=10,
            antilogarithm_uncertainty=_frac_uncert,
        )
        _mul, _mul_uncert = lezargus.library.math.multiply(
            multiplier=2.5,
            multiplicand=_log,
            multiplier_uncertainty=0,
            multiplicand_uncertainty=_log_uncert,
        )
        # Finally, the zero point itself.
        zero_point, zero_point_uncertainty = lezargus.library.math.add(
            augend=standard_magnitude,
            addend=_mul,
            augend_uncertainty=standard_magnitude_uncertainty,
            addend_uncertainty=_mul_uncert,
        )
        # Just making sure they are plain numbers.
        zero_point = float(zero_point)
        zero_point_uncertainty = float(zero_point_uncertainty)
        # All done.
        return zero_point, zero_point_uncertainty

    def calculate_magnitude(
        self: hint.Self,
        spectrum: hint.LezargusSpectrum,
    ) -> tuple[float, float]:
        """Calculate the Vega-based synthetic magnitude of a star in a filter.

        We compute the synthetic magnitude of a star using its spectrum,
        provided the filter transmission of this filter.

        Parameters
        ----------
        spectrum : LezargusSpectrum
            The target star that we will compute the synthetic filter
            magnitude of.

        Returns
        -------
        magnitude : float
            The computed synthetic magnitude.
        uncertainty : float
            The uncertainty in the computed synthetic magnitude.

        """
        # If a standard star has not been provided, this filter system is
        # considered un-calibrated so a synthetic magnitude cannot be
        # determined.
        if self.standard_spectrum is None:
            logging.critical(
                critical_type=logging.WrongOrderError,
                message=(
                    "Synthetic magnitudes require a standard spectrum be"
                    " provided; add one via `add_standard_star_spectrum`."
                ),
            )

        # We assume that the target star has data covering the entire filter
        # range as otherwise, you cannot really calculate an accurate magnitude.
        # However, we should still warn otherwise.
        overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
            base=spectrum.wavelength,
            contain=self.wavelength,
        )
        if overlap < 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "A synthetic magnitude is being computed where the star"
                    " spectrum does not cover the entire filter. The overlap"
                    f" fraction is {overlap}"
                ),
            )

        # Computing the total filter-weighted flux, using usual integration
        # methods.
        star_flux, star_uncertainty, __, __ = spectrum.interpolate(
            wavelength=self.wavelength,
            skip_mask=True,
            skip_flags=True,
        )
        # Integrating...
        (
            star_filter_flux,
            star_filter_flux_uncertainty,
        ) = lezargus.library.math.multiply(
            multiplier=star_flux,
            multiplicand=self.transmission_photon,
            multiplier_uncertainty=star_uncertainty,
            multiplicand_uncertainty=0,
        )
        (
            star_filter_integral,
            star_filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=star_filter_flux,
            integrand_uncertainty=star_filter_flux_uncertainty,
        )
        (
            filter_integral,
            filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=self.transmission_photon,
            integrand_uncertainty=0,
        )
        # Going through the equation via propagation.
        _frac, _frac_uncert = lezargus.library.math.divide(
            numerator=star_filter_integral,
            denominator=filter_integral,
            numerator_uncertainty=star_filter_integral_uncertainty,
            denominator_uncertainty=filter_integral_uncertainty,
        )
        _log, _log_uncert = lezargus.library.math.logarithm(
            antilogarithm=_frac,
            base=10,
            antilogarithm_uncertainty=_frac_uncert,
        )
        _mul, _mul_uncert = lezargus.library.math.multiply(
            multiplier=-2.5,
            multiplicand=_log,
            multiplier_uncertainty=0,
            multiplicand_uncertainty=_log_uncert,
        )

        # Finally, the magnitude after the zero point correction.
        magnitude, uncertainty = lezargus.library.math.add(
            augend=_mul,
            addend=self.zero_point,
            augend_uncertainty=_mul_uncert,
            addend_uncertainty=self.zero_point_uncertainty,
        )
        # Just making sure they are plain numbers.
        magnitude = float(magnitude)
        uncertainty = float(uncertainty)
        # All done.
        return magnitude, uncertainty

    def calculate_photometric_correction(
        self: hint.Self,
        spectrum: hint.LezargusSpectrum,
        magnitude: float,
        magnitude_uncertainty: float = 0,
    ) -> tuple[float, float]:
        """Calculate the photometric correction factor for a spectrum.

        We use the definition of the Vega photometric system to try and compute
        the scaling factor to scale the spectrum so that it is
        spectro-photometrically calibrated.

        See [[TODO]] for more information.

        Parameters
        ----------
        spectrum : LezargusSpectrum
            The target star spectrum that we will compute the synthetic
            filter magnitude of.
        magnitude : float
            The magnitude of the star in this given filter. We use this value
            to compute the photometric correction scale factor.
        magnitude_uncertainty : float, default = 0
            The uncertainty on the star's magnitude. If not provided, we
            assume a zero uncertainty.

        Returns
        -------
        factor : float
            The photometric correction factor calculated using synthetic
            photometry.
        factor_uncertainty : float
            The uncertainty on the photometric correction factor.

        """
        # If a standard star has not been provided, this filter system is
        # considered un-calibrated so a photometric correction factor cannot be
        # determined.
        if self.standard_spectrum is None:
            logging.critical(
                critical_type=logging.WrongOrderError,
                message=(
                    "Synthetic magnitudes require a standard spectrum be"
                    " provided; add one via `add_standard_star_spectrum`."
                ),
            )

        # We assume that the target star has data covering the entire filter
        # range as otherwise, you cannot really calculate an accurate magnitude.
        # However, we should still warn otherwise.
        overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
            base=spectrum.wavelength,
            contain=self.wavelength,
        )
        if overlap < 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "A photometric correction factor is being computed where "
                    " the star sptrum does not cover the entire filter. The "
                    f" overlap fraction is {overlap}"
                ),
            )

        # Computing the total filter-weighted flux, using usual integration
        # methods.
        star_flux, star_uncertainty, __, __ = spectrum.interpolate(
            wavelength=self.wavelength,
            skip_mask=True,
            skip_flags=True,
        )
        (
            star_filter_flux,
            star_filter_flux_uncertainty,
        ) = lezargus.library.math.multiply(
            multiplier=star_flux,
            multiplicand=self.transmission_photon,
            multiplier_uncertainty=star_uncertainty,
            multiplicand_uncertainty=0,
        )
        # Integrating.
        (
            star_filter_integral,
            star_filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=star_filter_flux,
            integrand_uncertainty=star_filter_flux_uncertainty,
        )
        (
            filter_integral,
            filter_integral_uncertainty,
        ) = lezargus.library.math.integrate_discrete(
            variable=self.wavelength,
            integrand=self.transmission_photon,
            integrand_uncertainty=0,
        )

        # The inverted fraction section of the calculation.
        fraction, fraction_uncertainty = lezargus.library.math.divide(
            numerator=filter_integral,
            denominator=star_filter_integral,
            numerator_uncertainty=filter_integral_uncertainty,
            denominator_uncertainty=star_filter_integral_uncertainty,
        )

        # The exponential term.
        _sub, _sub_uncert = lezargus.library.math.subtract(
            minuend=self.zero_point,
            subtrahend=magnitude,
            minuend_uncertainty=self.zero_point_uncertainty,
            subtrahend_uncertainty=magnitude_uncertainty,
        )
        _div, _div_uncert = lezargus.library.math.divide(
            numerator=_sub,
            denominator=2.5,
            numerator_uncertainty=_sub_uncert,
            denominator_uncertainty=0,
        )
        exponent, exponent_uncertainty = lezargus.library.math.exponentiate(
            base=10,
            exponent=_div,
            base_uncertainty=0,
            exponent_uncertainty=_div_uncert,
        )

        # And computing the factor itself.
        (
            factor,
            factor_uncertainty,
        ) = lezargus.library.math.multiply(
            multiplier=exponent,
            multiplicand=fraction,
            multiplier_uncertainty=exponent_uncertainty,
            multiplicand_uncertainty=fraction_uncertainty,
        )

        # Making sure they are simple numbers
        factor = float(factor)
        factor_uncertainty = float(factor_uncertainty)
        return factor, factor_uncertainty

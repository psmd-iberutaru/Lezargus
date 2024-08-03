"""Functions to deal with the computation of synthetic and real photometry."""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import lezargus
from lezargus.library import logging


def calculate_filter_zero_point_vega(
    filter_spectra: hint.LezargusSpectrum,
    standard_spectra: hint.LezargusSpectrum,
    standard_filter_magnitude: float,
    standard_filter_uncertainty: float | None = None,
) -> tuple[float, float | None]:
    """Calculate the magnitude zero point of a filter in the Vega system.

    This function computes Vega zero points by integrating it over the
    filter band pass. To better facilitate this as a library function, the
    standard spectra and its magnitude may be input. However, for most cases,
    the Vega spectra in the data collection is good enough.

    Parameters
    ----------
    filter_spectra : LezargusSpectrum
        The filter transmission, saved as a spectra container instance. The
        filter provided must be in the photon counting form. We assume that
        the wavelength term has already been multiplied through. The
        variable name is chosen to prevent a name clash with the Python
        built-in.
    standard_spectra : LezargusSpectrum
        The standard star, as saved by a spectra container instance.
    standard_filter_magnitude : float
        The magnitude of the standard star in that filter.
    standard_filter_uncertainty : float, default = None
        The uncertainty in the magnitude of the standard star in that filter.
        Often this is not needed because the magnitude value of standard
        defines the filter system anyways and so, by definition, there is no
        uncertainty.

    Returns
    -------
    zero_point : float
        The zero point value.
    zero_point_uncertainty : float or None
        The uncertainty on the zero point. If the standard star provided has
        some uncertainty, or if the filter does, then we attempt to calculate
        the uncertainty on the value.

    """
    # We assume that the standard star has data covering the entire filter
    # range as otherwise, you cannot really calculate an accurate zero point.
    # However, we should still warn otherwise.
    overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=standard_spectra.wavelength,
        contain=filter_spectra.wavelength,
    )
    if overlap < 1:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "A filter zero point is being computed where the standard"
                " spectra does not cover the entire filter. The overlap"
                f" fraction is {overlap}"
            ),
        )

    # Computing the total filter-weighted flux, using usual integration methods.
    star_flux, star_uncertainty, __, __ = standard_spectra.interpolate(
        wavelength=filter_spectra.wavelength,
        skip_mask=True,
        skip_flags=True,
    )
    # Integrating...
    (
        star_filter_flux,
        star_filter_flux_uncertainty,
    ) = lezargus.library.math.multiply(
        multiplier=star_flux,
        multiplicand=filter_spectra.data,
        multiplier_uncertainty=star_uncertainty,
        multiplicand_uncertainty=filter_spectra.uncertainty,
    )
    (
        star_filter_integral,
        star_filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=star_filter_flux,
        integrand_uncertainty=star_filter_flux_uncertainty,
    )
    (
        filter_integral,
        filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=filter_spectra.data,
        integrand_uncertainty=filter_spectra.uncertainty,
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
        augend=standard_filter_magnitude,
        addend=_mul,
        augend_uncertainty=standard_filter_uncertainty,
        addend_uncertainty=_mul_uncert,
    )

    # All done.
    return zero_point, zero_point_uncertainty


def calculate_filter_magnitude_vega(
    star_spectra: hint.LezargusSpectrum,
    filter_spectra: hint.LezargusSpectrum,
    filter_zero_point: float,
    filter_zero_point_uncertainty: float | None = None,
) -> tuple[float, float]:
    """Calculate the Vega-based synthetic magnitude of a star in a filter.

    We compute the synthetic magnitude of a star using its spectra, provided
    the filter transmission. This function is the manual method, where the
    filters and its properties must be defined manually. An automatic mode
    is also available, see `auto_calculate_filter_magnitude_vega`.

    Parameters
    ----------
    star_spectra : LezargusSpectrum
        The target star that we will compute the synthetic filter magnitude of.
    filter_spectra : LezargusSpectrum
        The filter transmission, saved as a spectra container instance. The
        filter provided must be in the photon counting form. We assume that
        the wavelength term has already been multiplied through. The
        variable name is chosen to prevent a name clash with the Python
        built-in.
    filter_zero_point : float
        The zero point value for the provided filter.
    filter_zero_point_uncertainty : float, default = None
        The uncertainty on the zero point. If not provided, we assume a zero
        uncertainty.

    Returns
    -------
    magnitude : float
        The computed synthetic magnitude.
    uncertainty : float
        The uncertainty in the computed synthetic magnitude.

    """
    # We assume that the target star has data covering the entire filter
    # range as otherwise, you cannot really calculate an accurate magnitude.
    # However, we should still warn otherwise.
    overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=star_spectra.wavelength,
        contain=filter_spectra.wavelength,
    )
    if overlap < 1:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "A filter magnitude is being computed where the star"
                " spectra does not cover the entire filter. The overlap"
                f" fraction is {overlap}"
            ),
        )

    # Computing the total filter-weighted flux, using usual integration methods.
    star_flux, star_uncertainty, __, __ = star_spectra.interpolate(
        wavelength=filter_spectra.wavelength,
        skip_mask=True,
        skip_flags=True,
    )
    # Integrating...
    (
        star_filter_flux,
        star_filter_flux_uncertainty,
    ) = lezargus.library.math.multiply(
        multiplier=star_flux,
        multiplicand=filter_spectra.data,
        multiplier_uncertainty=star_uncertainty,
        multiplicand_uncertainty=filter_spectra.uncertainty,
    )
    (
        star_filter_integral,
        star_filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=star_filter_flux,
        integrand_uncertainty=star_filter_flux_uncertainty,
    )
    (
        filter_integral,
        filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=filter_spectra.data,
        integrand_uncertainty=filter_spectra.uncertainty,
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
        addend=filter_zero_point,
        augend_uncertainty=_mul_uncert,
        addend_uncertainty=filter_zero_point_uncertainty,
    )

    # All done.
    return magnitude, uncertainty


def calculate_photometric_correction_factor_vega(
    star_spectra: hint.LezargusSpectrum,
    filter_spectra: hint.LezargusSpectrum,
    star_magnitude: float,
    filter_zero_point: float,
    star_magnitude_uncertainty: float | None = None,
    filter_zero_point_uncertainty: float | None = None,
) -> tuple[float, float]:
    """Compute photometric correction factors for Vega-based filters.

    We use the definition of the Vega photometric system to try and compute
    the scaling factor to scale the spectra so that it is
    spectro-photometrically calibrated.

    See [[TODO]] for more information.

    Parameters
    ----------
    star_spectra : LezargusSpectrum
        The target star that we will compute the synthetic filter magnitude of.
    filter_spectra : LezargusSpectrum
        The filter transmission, saved as a spectra container instance. The
        filter provided must be in the photon counting form. We assume that
        the wavelength term has already been multiplied through. The
        variable name is chosen to prevent a name clash with the Python
        built-in.
    star_magnitude : float
        The magnitude of the star in the given filter. We use this value to
        compute the photometric scale factor.
    filter_zero_point : float
        The zero point value for the provided filter.
    star_magnitude_uncertainty : float, default = None
        The uncertainty on the star's magnitude. If not provided, we assume a
        zero uncertainty.
    filter_zero_point_uncertainty : float, default = None
        The uncertainty on the zero point. If not provided, we assume a
        zero uncertainty.

    Returns
    -------
    correction_factor : float
        The correction factor calculated using synthetic photometry.
    correction_factor_uncertainty : float
        The uncertainty on the correction factor.

    """
    # We assume that the target star has data covering the entire filter
    # range as otherwise, you cannot really calculate an accurate magnitude.
    # However, we should still warn otherwise.
    overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=star_spectra.wavelength,
        contain=filter_spectra.wavelength,
    )
    if overlap < 1:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "A filter magnitude is being computed where the star"
                " spectra does not cover the entire filter. The overlap"
                f" fraction is {overlap}"
            ),
        )

    # Computing the total filter-weighted flux, using usual integration
    # methods.
    star_flux, star_uncertainty, __, __ = star_spectra.interpolate(
        wavelength=filter_spectra.wavelength,
        skip_mask=True,
        skip_flags=True,
    )
    (
        star_filter_flux,
        star_filter_flux_uncertainty,
    ) = lezargus.library.math.multiply(
        multiplier=star_flux,
        multiplicand=filter_spectra.data,
        multiplier_uncertainty=star_uncertainty,
        multiplicand_uncertainty=filter_spectra.uncertainty,
    )
    # Integrating.
    (
        star_filter_integral,
        star_filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=star_filter_flux,
        integrand_uncertainty=star_filter_flux_uncertainty,
    )
    (
        filter_integral,
        filter_integral_uncertainty,
    ) = lezargus.library.math.integrate_discrete(
        variable=filter_spectra.wavelength,
        integrand=filter_spectra.data,
        integrand_uncertainty=filter_spectra.uncertainty,
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
        minuend=filter_zero_point,
        subtrahend=star_magnitude,
        minuend_uncertainty=filter_zero_point_uncertainty,
        subtrahend_uncertainty=star_magnitude_uncertainty,
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
        correction_factor,
        correction_factor_uncertainty,
    ) = lezargus.library.math.multiply(
        multiplier=exponent,
        multiplicand=fraction,
        multiplier_uncertainty=exponent_uncertainty,
        multiplicand_uncertainty=fraction_uncertainty,
    )
    return correction_factor, correction_factor_uncertainty

"""Different mathematical operations which we also propagate uncertainty."""

import numpy as np
import scipy.integrate

from lezargus.library import hint
from lezargus.library import logging


def covariance(param_1: hint.ndarray, param_2: hint.ndarray) -> float:
    """Compute the covariance for two parameters.

    If the covariance cannot be computed, we default to 0.

    Parameters
    ----------
    param_1 : ndarray
        The first parameter.
    param_2 : ndarray
        The second parameter.

    Returns
    -------
    covar : float
        The covariance.
    """
    # Preparing the parameters.
    param_1 = np.array(param_1)
    param_2 = np.array(param_2)
    # A quick skip for covariance determinations, if the values are just single
    # floats, then we assume a zero covariance.
    if param_1.size == param_2.size == 1:
        covar = 0
        return covar
    # Calculating the covariance.
    try:
        covar = np.cov(param_1.flatten(), param_2.flatten())[0, 1]
    except ValueError:
        covar = 0

    # We are forcing the covariances to be 0.
    covar = 0

    # All done.
    return covar


def add(
    augend: hint.ndarray,
    addend: hint.ndarray,
    augend_uncertainty: hint.ndarray = None,
    addend_uncertainty: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Add two values and propagate uncertainties.

    Parameters
    ----------
    augend : ndarray
        The "left"-side of the addition.
    addend : ndarray
        The "right"-side of the addition.
    augend_uncertainty : ndarray, default = None
        The uncertainty on the augend term. If None, we assume that the
        uncertainty is 0.
    addend_uncertainty : ndarray, default = None
        The uncertainty on the addend term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the addition operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, then we assume zero.
    augend_uncertainty = 0 if augend_uncertainty is None else augend_uncertainty
    addend_uncertainty = 0 if addend_uncertainty is None else addend_uncertainty

    # Computing the result.
    result = augend + addend
    # Propagating the uncertainty.
    covar = covariance(param_1=augend, param_2=addend)
    uncertainty = np.sqrt(
        augend_uncertainty**2 + addend_uncertainty**2 + 2 * covar,
    )
    return result, uncertainty


def subtract(
    minuend: hint.ndarray,
    subtrahend: hint.ndarray,
    minuend_uncertainty: hint.ndarray = None,
    subtrahend_uncertainty: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Subtract two values and propagate uncertainties.

    Parameters
    ----------
    minuend : ndarray
        The "left"-side of the subtraction.
    subtrahend : ndarray
        The "right"-side of the subtraction.
    minuend_uncertainty : ndarray, default = None
        The uncertainty on the minuend term. If None, we assume that the
        uncertainty is 0.
    subtrahend_uncertainty : ndarray, default = None
        The uncertainty on the subtrahend term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the subtraction operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, we assume they are 0.
    minuend_uncertainty = (
        0 if minuend_uncertainty is None else minuend_uncertainty
    )
    subtrahend_uncertainty = (
        0 if subtrahend_uncertainty is None else subtrahend_uncertainty
    )

    # Computing the result.
    result = minuend - subtrahend
    # Propagating the uncertainty.
    covar = covariance(param_1=minuend, param_2=subtrahend)
    uncertainty = np.sqrt(
        minuend_uncertainty**2 + subtrahend_uncertainty**2 - 2 * covar,
    )
    return result, uncertainty


def multiply(
    multiplier: hint.ndarray,
    multiplicand: hint.ndarray,
    multiplier_uncertainty: hint.ndarray = None,
    multiplicand_uncertainty: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Multiply two values and propagate uncertainties.

    Note, the typical formula for the propagation of uncertainties for
    multiplication can lead to issues because of division by zero. We
    rewrite the equations. This reformulation is based on Astropy's
    reformulation.
    See :ref:`technical-uncertainty-multiplication_and_division` for more
    information.

    Parameters
    ----------
    multiplier : ndarray
        The "left"-side of the multiplication.
    multiplicand : ndarray
        The "right"-side of the multiplication.
    multiplier_uncertainty : ndarray, default = None
        The uncertainty on the multiplier term. If None, we assume that the
        uncertainty is 0.
    multiplicand_uncertainty : ndarray, default = None
        The uncertainty on the multiplicand term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the multiplication operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, then we assume zero.
    multiplier_uncertainty = (
        0 if multiplier_uncertainty is None else multiplier_uncertainty
    )
    multiplicand_uncertainty = (
        0 if multiplicand_uncertainty is None else multiplicand_uncertainty
    )

    # The result.
    result = multiplier * multiplicand

    # Propagating...
    covar = covariance(param_1=multiplier, param_2=multiplicand)
    # Doing the propagation via the new equation to remove some NaN treatment.
    variance = (
        (multiplier_uncertainty * multiplicand) ** 2
        + (multiplicand_uncertainty * multiplier) ** 2
        + (2 * multiplier * multiplicand * covar)
    )
    uncertainty = np.sqrt(variance)

    return result, uncertainty


def divide(
    numerator: hint.ndarray,
    denominator: hint.ndarray,
    numerator_uncertainty: hint.ndarray = None,
    denominator_uncertainty: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Divide two values and propagate uncertainties.

    Note, the typical formula for the propagation of uncertainties for
    division can lead to issues because of division by zero. We
    rewrite the equations. This reformulation is based on Astropy's
    reformulation.
    See :ref:`technical-uncertainty-multiplication_and_division` for more
    information.

    Parameters
    ----------
    numerator : ndarray
        The numerator of the division; the top value.
    denominator : ndarray
        The denominator of the division; the bottom value.
    numerator_uncertainty : ndarray, default = None
        The uncertainty on the numerator term. If None, we assume that the
        uncertainty is 0.
    denominator_uncertainty : ndarray, default = None
        The uncertainty on the denominator term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the division operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, then we assume zero.
    numerator_uncertainty = (
        0 if numerator_uncertainty is None else numerator_uncertainty
    )
    denominator_uncertainty = (
        0 if denominator_uncertainty is None else denominator_uncertainty
    )

    # The result.
    result = numerator / denominator

    # Propagating...
    covar = covariance(param_1=numerator, param_2=denominator)
    # Doing the propagation via the new equation to remove some NaN treatment.
    variance = (
        (numerator_uncertainty * denominator) ** 2
        + (denominator_uncertainty * numerator) ** 2
        + (2 * numerator * denominator * covar)
    )
    uncertainty = np.sqrt(variance)
    return result, uncertainty


def exponentiate(
    base: hint.ndarray,
    exponent: hint.ndarray,
    base_uncertainty: hint.ndarray,
    exponent_uncertainty: hint.ndarray,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Compute the exponent of two values and propagate uncertainties.

    Parameters
    ----------
    base : ndarray
        The base of the exponentiation; the lower value.
    exponent : ndarray
        The exponent of the exponentiation; the upper value.
    base_uncertainty : ndarray, default = None
        The uncertainty on the base term. If None, we assume that the
        uncertainty is 0.
    exponent_uncertainty : ndarray, default = None
        The uncertainty on the exponent term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the exponentiation operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, then we assume zero.
    base_uncertainty = 0 if base_uncertainty is None else base_uncertainty
    exponent_uncertainty = (
        0 if exponent_uncertainty is None else exponent_uncertainty
    )

    # The result.
    result = base**exponent
    # Propagating the uncertainty, term by term.
    covar = covariance(param_1=base, param_2=exponent)
    base_term = ((exponent / base) * base_uncertainty) ** 2
    expo_term = (np.log(base) * exponent_uncertainty) ** 2
    covar_term = 2 * ((exponent * np.log(base)) / base) * covar
    uncertainty = np.abs(result) * np.sqrt(base_term + expo_term + covar_term)
    # All done.
    return result, uncertainty


def logarithm(
    antilogarithm: hint.ndarray,
    base: hint.ndarray,
    antilogarithm_uncertainty: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray]:
    """Compute the logarithm of two values and propagate uncertainties.

    Parameters
    ----------
    antilogarithm : ndarray
        The inside value of the logarithm; what we are taking a logarithm of.
    base : ndarray
        The logarithm base.
    antilogarithm_uncertainty : ndarray, default = None
        The uncertainty in the anti-logarithm.

    Returns
    -------
    result : ndarray
        The result of the exponentiation operation.
    uncertainty : ndarray
        The propagated uncertainty.
    """
    # If the uncertainties are not provided, then we assume zero.
    antilogarithm_uncertainty = (
        0 if antilogarithm_uncertainty is None else antilogarithm_uncertainty
    )

    # Computing the result and propagating.
    result = np.log(antilogarithm) / np.log(base)
    uncertainty = np.abs(
        antilogarithm_uncertainty / (antilogarithm * np.log(base)),
    )
    return result, uncertainty


def integrate_discrete(
    variable: hint.ndarray,
    integrand: hint.ndarray,
    integrand_uncertainty: hint.ndarray = None,
) -> tuple[float, float]:
    """Integrate discrete values and propagate the errors.

    Parameters
    ----------
    variable : ndarray
        The variable being integrated over.
    integrand : ndarray
        The integrand function being integrated.
    integrand_uncertainty : ndarray, default = None
        The uncertainty in the integrand. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : float
        The result the integration.
    uncertainty : float
        The uncertainty on the integration.
    """
    # TODO

    # The result of the integral.
    result = scipy.integrate.trapezoid(
        integrand,
        x=variable,
    )

    logging.error(
        error_type=logging.NotSupportedError,
        message="Uncertainty values on integrations need to be done.",
    )
    uncertainty = 0

    return result, uncertainty


def weighted_mean(
    values: hint.ndarray,
    values_uncertainty: hint.ndarray = None,
    weights: hint.ndarray = None,
) -> tuple[float, float]:
    """Calculate a weighted mean, propagating uncertainties where needed.

    This function calculates the weighted arithmetic mean of a group of samples
    and weights, ignoring any entry that is not a valid number. If the weights
    are not provided, we default to equal weights and thus the ordinary
    arithmetic mean.

    TODO : EXPLAIN ERROR PROPAGATION FOR MEANS.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the weighted mean of.
    values_uncertainty : ndarray, default = None
        The uncertainties in the values. If None, we default to no uncertainty.
    weights : ndarray, default = None
        The weights for the given values for the weighted mean. If None, we
        assume equal weights.

    Returns
    -------
    mean_value : float
        The calculated mean.
    mean_uncertainty : float
        The calculated uncertainty in the mean.
    """
    # We determine the defaults for the uncertainty and the weights.
    values_uncertainty = (
        np.zeros_like(values)
        if values_uncertainty is None
        else values_uncertainty
    )
    weights = np.ones_like(values) if weights is None else weights
    # We also do not include any values which are not actual numbers.
    clean_index = (
        np.isfinite(values)
        & np.isfinite(values_uncertainty)
        & np.isfinite(weights)
    )
    clean_values = values[clean_index]
    clean_uncertainty = values_uncertainty[clean_index]
    clean_weights = weights[clean_index]
    # Finally, calculating the mean.
    mean_value = np.average(clean_values, weights=clean_weights)
    # The error propagation, done as prescribed. We assume next to no
    # covariance and in general we calculate it via variance propagation of
    # the definition of the weighted mean.
    mean_uncertainty = (
        np.sqrt(np.nansum((clean_uncertainty * clean_weights) ** 2))
        / clean_index.sum()
    )
    # All done.
    return mean_value, mean_uncertainty

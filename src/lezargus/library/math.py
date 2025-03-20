"""Different mathematical operations which we also propagate uncertainty.

Many mathematical operations are needed, and the uncertainties of these
operations need to be propagated. We gather these functions so that they
can easily be reused or better default methods used in place.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import decimal

import numpy as np
import scipy.integrate

import lezargus
from lezargus.library import logging


def covariance(
    param_1: hint.NDArray | float,
    param_2: hint.NDArray | float,
) -> float:
    """Compute the covariance for two parameters.

    If the covariance cannot be computed, we default to 0.

    Parameters
    ----------
    param_1 : ndarray or float
        The first parameter.
    param_2 : ndarray or float
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
    augend: hint.NDArray | float,
    addend: hint.NDArray | float,
    augend_uncertainty: hint.NDArray | float | None = 0,
    addend_uncertainty: hint.NDArray | float | None = 0,
) -> tuple[hint.NDArray | float, hint.NDArray | float]:
    """Add two values and propagate uncertainties.

    Parameters
    ----------
    augend : ndarray or float
        The "left"-side of the addition.
    addend : ndarray or float
        The "right"-side of the addition.
    augend_uncertainty : ndarray, default = 0
        The uncertainty on the augend term. If None, we assume that the
        uncertainty is 0.
    addend_uncertainty : ndarray, default = 0
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
    minuend: hint.NDArray | float,
    subtrahend: hint.NDArray | float,
    minuend_uncertainty: hint.NDArray | float | None = None,
    subtrahend_uncertainty: hint.NDArray | float | None = None,
) -> tuple[float | hint.NDArray, float | hint.NDArray]:
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
    multiplier: hint.NDArray | float,
    multiplicand: hint.NDArray | float,
    multiplier_uncertainty: hint.NDArray | float | None = None,
    multiplicand_uncertainty: hint.NDArray | float | None = None,
) -> tuple[float | hint.NDArray, float | hint.NDArray]:
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
    numerator: float | hint.NDArray,
    denominator: float | hint.NDArray,
    numerator_uncertainty: hint.NDArray | float | None = None,
    denominator_uncertainty: hint.NDArray | float | None = None,
) -> tuple[float | hint.NDArray, float | hint.NDArray]:
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


def modulo(
    numerator: hint.NDArray | float,
    denominator: hint.NDArray | float,
    numerator_uncertainty: hint.NDArray | float | None = None,
    denominator_uncertainty: hint.NDArray | float | None = None,
) -> tuple[hint.NDArray, hint.NDArray]:
    """Compute the modulo of two values and propagate uncertainties.

    This function properly handles floating point modulo operations and thus
    is preferred if floats are involved.

    Parameters
    ----------
    numerator : ndarray
        The numerator of the modulo division; the top value.
    denominator : ndarray
        The denominator of the modulo division; the bottom value.
    numerator_uncertainty : ndarray, default = None
        The uncertainty on the numerator term. If None, we assume that the
        uncertainty is 0.
    denominator_uncertainty : ndarray, default = None
        The uncertainty on the denominator term. If None, we assume that the
        uncertainty is 0.

    Returns
    -------
    result : ndarray
        The result of the modulo division operation.
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

    # We utilize the decimal library to do the modulo and so we can only work
    # on one number at a time. However, we leverage Numpy's vectorization
    # convenience function.
    def single_value_modulo(
        num: float,
        den: float,
        num_uncrt: float,
        den_uncert: float,
    ) -> tuple[float, float]:
        """Determine the modulo of a single division."""
        # We leverage the decimal library to help us here.
        num_decimal = decimal.Decimal(str(num))
        den_decimal = decimal.Decimal(str(den))
        quotient = float(num_decimal % den_decimal)
        # Error propagation to be done.
        lezargus.library.wrapper.do_nothing(num_uncrt, den_uncert)
        uncert = 0
        return quotient, uncert

    # Vectorizing the function.
    vector_modulo = np.vectorize(single_value_modulo, otypes=[float, float])
    result = vector_modulo(
        numerator,
        denominator,
        numerator_uncertainty,
        denominator_uncertainty,
    )
    return result


def exponentiate(
    base: float | hint.NDArray,
    exponent: float | hint.NDArray,
    base_uncertainty: hint.NDArray | float | None = None,
    exponent_uncertainty: hint.NDArray | float | None = None,
) -> tuple[hint.NDArray, hint.NDArray]:
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
    antilogarithm: hint.NDArray,
    base: hint.NDArray,
    antilogarithm_uncertainty: hint.NDArray | float | None = None,
) -> tuple[hint.NDArray, hint.NDArray]:
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
    variable: hint.NDArray,
    integrand: hint.NDArray,
    integrand_uncertainty: hint.NDArray | float | None = None,
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
    # The result of the integral.
    result = scipy.integrate.trapezoid(
        integrand,
        x=variable,
    )

    logging.error(
        error_type=logging.ToDoError,
        message="Uncertainty values on integrations need to be done.",
    )
    integrand_uncertainty = 0
    uncertainty = integrand_uncertainty

    return result, uncertainty


def normalize_weights(weights: hint.NDArray) -> hint.NDArray:
    """Normalize weights, handling NaNs so they don't screw things up.

    We do not include NaNs in the normalization of the weights, however we
    still keep them as weights to allow for its proper propagation when needed.

    Parameters
    ----------
    weights : ndarray
        The weights to normalize.

    Returns
    -------
    normalized : ndarray
        The weights normalized.

    """
    # It is easiest to deal with arrays.
    weights = np.asarray(weights, dtype=float)

    # Weights only make sense if they are real and finite. We clean up the
    # weights to get a finite subset to calculate the normalization factor.
    real_weights = lezargus.library.sanitize.clean_finite_arrays(weights)

    # If there is no left over data to calculate the normalization factor,
    # we use uniform weights...
    if len(real_weights) == 0 or np.count_nonzero(real_weights) == 0:
        using_weights = np.ones_like(real_weights)
    # ...otherwise, we just use the real weights.
    else:
        using_weights = real_weights

    # Computing the normalization summation.
    weight_sum = np.nansum(using_weights)

    # Normalizing the weights. We just ignore the parts with NaN and pass them
    # on.
    normalized = weights / weight_sum

    return normalized


def weighted_mean(
    values: hint.NDArray,
    uncertainties: hint.NDArray | None = None,
    weights: hint.NDArray | None = None,
) -> tuple[float, float]:
    """Calculate a weighted mean, propagating uncertainties where needed.

    This function calculates the weighted arithmetic mean of a group of samples
    and weights. If the weights are not provided, we default to equal weights
    and thus the ordinary arithmetic mean. If any value, uncertainty, or weight
    is NaN, the result is a NaN.

    See :ref:`technical-uncertainty-weighted-mean` for more
    information.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the weighted mean of.
    uncertainties : ndarray, default = None
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
    uncertainties = (
        np.zeros_like(values) if uncertainties is None else uncertainties
    )
    weights = np.ones_like(values) if weights is None else weights

    # Normalize the weights.
    norm_weights = normalize_weights(weights=weights)

    # Finally, calculating the mean.
    mean_value = float(np.average(values, weights=norm_weights))
    # The error propagation, done as prescribed. We assume next to no
    # covariance and in general we calculate it via variance propagation of
    # the definition of the weighted mean.
    mean_uncertainty = np.sqrt(np.sum((uncertainties * norm_weights) ** 2))
    # All done.
    return mean_value, mean_uncertainty


def nan_weighted_mean(
    values: hint.NDArray,
    uncertainties: hint.NDArray | None = None,
    weights: hint.NDArray | None = None,
) -> tuple[float, float]:
    """Calculate the no-NaN weighted mean and uncertainty.

    This function is similar to :py:func:`weighted_mean`,
    but we do not include any non-finite values.

    See :ref:`technical-uncertainty-weighted-mean` for more information.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the weighted mean of.
    uncertainties : ndarray, default = None
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
    uncertainties = (
        np.zeros_like(values) if uncertainties is None else uncertainties
    )
    weights = np.ones_like(values) if weights is None else weights

    # We also do not include any values which are not actual numbers.
    (
        clean_values,
        clean_uncertainty,
        clean_weights,
    ) = lezargus.library.sanitize.clean_finite_arrays(
        values,
        uncertainties,
        weights,
    )

    # And we just send it to the original function to compute it.
    mean_value, mean_uncertainty = weighted_mean(
        values=clean_values,
        uncertainties=clean_uncertainty,
        weights=clean_weights,
    )
    return mean_value, mean_uncertainty


def weighted_quantile_mean(
    values: hint.NDArray,
    uncertainties: hint.NDArray | None = None,
    weights: hint.NDArray | None = None,
    quantile: float | tuple[float, float] = (0, 1),
) -> tuple[float, float]:
    """Calculate the no-NaN weighted quantile mean and uncertainty.

    See :ref:`technical-uncertainty-weighted-mean` for more
    information on the general weighted mean; the quantile functionality only
    reduces the input space.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the weighted quantile mean of.
    uncertainties : ndarray, default = None
        The uncertainties in the values. If None, we default to no uncertainty.
    weights : ndarray, default = None
        The weights for the given values for the weighted quantile mean. If
        None, we assume equal weights.
    quantile : float or tuple, default = (0, 1)
        A single quantile value to cut off each end; or the minimum and
        maximum quantile to cut from the entire array before taking the
        average. Must be between 0 and 1.

    Returns
    -------
    mean_value : float
        The calculated mean.
    mean_uncertainty : float
        The calculated uncertainty in the mean.

    """
    # We determine the defaults for the uncertainty and the weights.
    uncertainties = (
        np.zeros_like(values) if uncertainties is None else uncertainties
    )
    weights = np.ones_like(values) if weights is None else weights

    # We also need to convert the quantile if they just provided a single
    # number.
    if isinstance(quantile, int | float | np.number):
        # Sanity check.
        half_quantile = 0.5
        if quantile >= half_quantile:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Single value quantile {quantile} is greater than half."
                    " All data may be cut."
                ),
            )
        dual_quantile = (quantile, 1 - quantile)
    else:
        dual_quantile = tuple(quantile)

    # We need to obtain only the values within the quartile before computing
    # the average.
    lower_limit, upper_limit = np.nanquantile(values, dual_quantile)
    valid_index = (lower_limit <= values) & (values <= upper_limit)
    valid_values = values[valid_index]
    valid_uncertainties = uncertainties[valid_index]
    valid_weights = weights[valid_index]

    # After the cut, the rest is just a normal weighted average.
    mean_value, mean_uncertainty = weighted_mean(
        values=valid_values,
        uncertainties=valid_uncertainties,
        weights=valid_weights,
    )
    # All done.
    return mean_value, mean_uncertainty


def median(
    values: hint.NDArray,
    uncertainties: hint.NDArray | None = None,
    weights: hint.NDArray | None = None,
) -> tuple[float, float]:
    """Calculate the median and uncertainty.

    See :ref:`technical-uncertainty-median` for more information on the
    general median.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the median of.
    uncertainties : ndarray, default = None
        The uncertainties in the values. If None, we default to no uncertainty.
    weights : ndarray, default = None
        The weights for the given values for the median of. If None, we
        assume equal weights. This is only used for uncertainty propagation.

    Returns
    -------
    median_value : float
        The calculated median.
    median_uncertainty : float
        The calculated uncertainty in the median.

    """
    # We determine the defaults for the uncertainty and the weights.
    uncertainties = (
        np.zeros_like(values) if uncertainties is None else uncertainties
    )
    weights = np.ones_like(values) if weights is None else weights

    # Calculating the median.
    median_value = float(np.median(values))

    # The uncertainty propagation on the median.
    logging.error(
        error_type=logging.ToDoError,
        message="Uncertainty propagation on medians.",
    )
    median_uncertainty = 0

    # All done.
    return median_value, median_uncertainty


def nan_median(
    values: hint.NDArray,
    uncertainties: hint.NDArray | None = None,
    weights: hint.NDArray | None = None,
) -> tuple[float, float]:
    """Calculate the no-NaN median and uncertainty.

    This function is similar to :py:func:`median`,
    but we do not include any non-finite values.

    See :ref:`technical-uncertainty-median` for more information.

    Parameters
    ----------
    values : ndarray
        The values which we will compute the median of.
    uncertainties : ndarray, default = None
        The uncertainties in the values. If None, we default to no uncertainty.
    weights : ndarray, default = None
        The weights for the given values for the median of. If None, we
        assume equal weights. This is only used for uncertainty propagation.

    Returns
    -------
    median_value : float
        The calculated median.
    median_uncertainty : float
        The calculated uncertainty in the median.

    """
    # We determine the defaults for the uncertainty and the weights.
    uncertainties = (
        np.zeros_like(values) if uncertainties is None else uncertainties
    )
    weights = np.ones_like(values) if weights is None else weights

    # We also do not include any values which are not actual numbers.
    (
        clean_values,
        clean_uncertainty,
        clean_weights,
    ) = lezargus.library.sanitize.clean_finite_arrays(
        values,
        uncertainties,
        weights,
    )

    # And we just send it to the original function to compute it.
    median_value, median_uncertainty = median(
        values=clean_values,
        uncertainties=clean_uncertainty,
        weights=clean_weights,
    )
    return median_value, median_uncertainty

"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
"""

import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


def stitch_spectra_functional(
    wavelength_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    data_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    uncertainty_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    weight_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    average_routine: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ] = None,
    reference_points: hint.ndarray = None,
) -> tuple[
    hint.Callable[[hint.ndarray], hint.ndarray],
    hint.Callable[[hint.ndarray], hint.ndarray],
    hint.Callable[[hint.ndarray], hint.ndarray],
]:
    R"""Stitch spectra functions together.

    We take functional forms of the wavelength, data, uncertainty, and weight
    (in the form of f(wave) = result), and determine the average spectra.
    We assume that the all of the functional forms properly handle any bounds,
    gaps, and interpolative limits. The input lists of functions should be
    parallel and all of them should be of the same (unit) scale.

    For more information, the formal method is described in [[TODO]].

    Parameters
    ----------
    wavelength_functions : list[Callable]
        The list of the wavelength function. The inputs to these functions
        should be the wavelength.
    data_functions : list[Callable]
        The list of the data function. The inputs to these functions should
        be the wavelength.
    uncertainty_functions : list[Callable]
        The list of the uncertainty function. The inputs to these functions
        should be the wavelength.
    weight_functions : list[Callable]
        The list of the weight function. The weights are passed to the
        averaging routine to properly weight the average.
    average_routine : Callable, default = None
        The averaging function. It must be able to support the propagation of
        uncertainties and weights. As such, it should have the form of
        :math:`f(x, \sigma, w) = \bar{x} \pm \sigma`. If None, we used a
        standard weighted average.
    reference_points : ndarray, default = None
        The reference points which we are going to evaluate the above functions
        at. The values should be of the same (unit) scale as the input of the
        above functions. If None, we default to a uniformly distributed set:

        .. math::

            \left\{ x \in \mathbb{R}, N=10^6 \;|\;
            0.30 \leq x \leq 5.50 \right\}

        Otherwise, we use the points provided. We remove any non-finite points
        and sort.

    Returns
    -------
    average_wavelength_function : Callable
        The functional form of the average wavelength.
    average_data_function : Callable
        The functional form of the average data.
    average_uncertainty_function : Callable
        The functional form of the propagated uncertainties.
    """
    # We first determine the defaults, starting with the averaging routine.
    if average_routine is None:
        average_routine = library.uncertainty.weighted_mean
    # And we also determine the reference points, which is vaguely based on
    # the atmospheric optical and infrared windows.
    if reference_points is None:
        reference_points = np.linspace(0.30, 5.50, 1000000)
    else:
        reference_points = np.sort(
            library.array.clean_finite_arrays(reference_points),
        )

    # Now, we need to have the lists all be parallel, a quick and dirty check
    # is to make sure they are all the same length. We assume the user did not
    # make any mistakes when pairing them up.
    if (
        not len(wavelength_functions)
        == len(data_functions)
        == len(uncertainty_functions)
        == len(weight_functions)
    ):
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "The provided lengths of the wavelength, ={wv}; data, ={da};"
                " uncertainty, ={un}; and weight, ={wg}, function lists are of"
                " different sizes and are not parallel.".format(
                    wv=len(wavelength_functions),
                    da=len(data_functions),
                    un=len(uncertainty_functions),
                    wg=len(weight_functions),
                )
            ),
        )

    # We next compute needed discrete values from the functional forms. We
    # can also properly stack them in an array as they are all aligned with
    # the reference points.
    wavelength_points = np.array(
        [functiondex(reference_points) for functiondex in wavelength_functions],
    )
    data_points = np.array(
        [functiondex(reference_points) for functiondex in data_functions],
    )
    uncertainty_points = np.array(
        [
            functiondex(reference_points)
            for functiondex in uncertainty_functions
        ],
    )
    weight_points = np.array(
        [functiondex(reference_points) for functiondex in weight_functions],
    )

    # We determine the average of all of the points using the provided
    # averaging routine. We do not actually need the reference points at this
    # time.
    average_wavelength, average_data, average_uncertainty = (
        [] for __ in range(3)
    )
    for index, __ in enumerate(reference_points):
        # We determine the average wavelength, for consistency. We do not
        # care for the computed uncertainty in the wavelength; the typical
        # trash variable is being used for the loop so we use something else
        # just in case.
        temp_wave, ___ = average_routine(
            values=wavelength_points[:, index],
            uncertainties=None,
            weights=weight_points[:, index],
        )
        temp_data, temp_uncertainty = average_routine(
            values=data_points[:, index],
            uncertainties=uncertainty_points[:, index],
            weights=weight_points[:, index],
        )
        # Adding the points.
        average_wavelength.append(temp_wave)
        average_data.append(temp_data)
        average_uncertainty.append(temp_uncertainty)
    # Making things into arrays is easier.
    average_wavelength = np.array(average_wavelength)
    average_data = np.array(average_data)
    average_uncertainty = np.array(average_uncertainty)

    # A quick check. The wavelengths calculated should be above the same as
    # the reference points. We don't care about NaNs for this check.
    (
        check_reference_points,
        check_average_wavelength,
    ) = library.array.clean_finite_arrays(reference_points, average_wavelength)
    if not np.all(np.isclose(check_reference_points, check_average_wavelength)):
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "The reference points and the resulting wavelength points seem"
                " to differ; double check the inputs."
            ),
        )

    # We need to compute the new functional form of the wavelength, data,
    # and uncertainty. However, we need to keep in mind of any NaNs which were
    # present before creating the new interpolator. All of the interpolators
    # remove NaNs and so we reintroduce them by assuming a NaN gap where the
    # data spacing is strictly larger than the largest spacing of data points.
    reference_gap = (1 + 1e-3) * np.nanmax(
        reference_points[1:] - reference_points[:-1],
    )

    # Building the interpolators.
    average_wavelength_function = (
        library.interpolate.cubic_1d_interpolate_gap_factory(
            x=average_wavelength,
            y=average_wavelength,
            gap_size=reference_gap,
        )
    )
    average_data_function = (
        library.interpolate.cubic_1d_interpolate_gap_factory(
            x=average_wavelength,
            y=average_data,
            gap_size=reference_gap,
        )
    )
    average_uncertainty_function = (
        library.interpolate.cubic_1d_interpolate_gap_factory(
            x=average_wavelength,
            y=average_uncertainty,
            gap_size=reference_gap,
        )
    )
    # All done.
    return (
        average_wavelength_function,
        average_data_function,
        average_uncertainty_function,
    )

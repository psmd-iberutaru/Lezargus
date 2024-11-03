"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
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


def calculate_spectra_scale_factor(
    base_wavelength: hint.NDArray,
    base_data: hint.NDArray,
    input_wavelength: hint.NDArray,
    input_data: hint.NDArray,
    base_uncertainty: hint.NDArray | None = None,
    input_uncertainty: hint.NDArray | None = None,
    bounds: tuple[float, float] = (-np.inf, +np.inf),
) -> tuple[float, float]:
    """Find the scale factor to scale one overlapping spectrum to another.

    We determine what scale factor would properly match some input spectrum
    data to some base data, provided that they have some wavelength overlap.
    An additional bounds may be set in addition to the wavelength overlap.
    The method provided is described in [[TODO]].

    Parameter
    ---------
    base_wavelength : ndarray
        The wavelength array of the base spectrum data. Must be the same unit
        as the input wavelength.
    base_data : ndarray
        The spectral data of the base, which the scale factor would scale the
        input to.
    input_wavelength : ndarray
        The wavelength array of the input spectrum data. Must be the same unit
        as the base wavelength.
    input_data : ndarray
        The spectral data of the input, what the scale factor is for.
    base_uncertainty : ndarray, default = None
        The uncertainty on the base data. If None, we default to no
        uncertainty.
    input_uncertainty : ndarray, default = None
        The uncertainty on the input data. If None, we default to no
        uncertainty.
    bounds : tuple, default = (-np.inf, +np.inf)
        An additional set of wavelength bounds to specify the limits of the
        overlap which we use to determine the scale factor. Format is
        (minimum, maximum). Must be in the same units as the base and input
        wavelengths.

    Returns
    -------
    scale_factor : float
        The scale factor to scale the input data to match the base data.
    scale_uncertainty : float
        The uncertainty in the scale factor. This is usually not relevant.

    """
    # The uncertainty defaults.
    base_uncertainty = (
        np.zeros_like(base_wavelength)
        if base_uncertainty is None
        else base_uncertainty
    )
    input_uncertainty = (
        np.zeros_like(input_wavelength)
        if input_uncertainty is None
        else input_uncertainty
    )

    # We rebase both spectra to some common wavelength. We also need to account
    # for any gaps in the region.
    common_wavelength = np.append(base_wavelength, input_wavelength)
    small_gap = lezargus.library.interpolate.get_smallest_gap(
        wavelength=common_wavelength,
    )

    interpolator = (
        lezargus.library.interpolate.Spline1DInterpolate.template_class(
            extrapolate=False,
            gap=small_gap,
            gap_fill=np.nan,
        )
    )
    common_base_data = interpolator(x=base_wavelength, v=base_data)(
        common_wavelength,
    )
    common_base_uncertainty = interpolator(
        x=base_wavelength,
        v=base_uncertainty,
    )(common_wavelength)
    common_input_data = interpolator(x=input_wavelength, v=input_data)(
        common_wavelength,
    )
    common_input_uncertainty = interpolator(
        x=input_wavelength,
        v=input_uncertainty,
    )(common_wavelength)

    # And, we only use the data where there is proper overlap between both
    # spectra; and within the bounds where provided.
    overlap_index = (
        (
            (np.nanmin(base_wavelength) <= common_wavelength)
            & (common_wavelength <= np.nanmax(base_wavelength))
        )
        & (
            (np.nanmin(input_wavelength) <= common_wavelength)
            & (common_wavelength <= np.nanmax(input_wavelength))
        )
        & (
            (min(bounds) <= common_wavelength)
            & (common_wavelength <= max(bounds))
        )
    )
    overlap_wavelength = common_wavelength[overlap_index]
    overlap_base_data = common_base_data[overlap_index]
    overlap_base_uncertainty = common_base_uncertainty[overlap_index]
    overlap_input_data = common_input_data[overlap_index]
    overlap_input_uncertainty = common_input_uncertainty[overlap_index]

    # We do not want to include any NaNs or non-numbers.
    (
        __,
        clean_base_data,
        clean_base_uncertainty,
        clean_input_data,
        clean_input_uncertainty,
    ) = lezargus.library.sanitize.clean_finite_arrays(
        overlap_wavelength,
        overlap_base_data,
        overlap_base_uncertainty,
        overlap_input_data,
        overlap_input_uncertainty,
    )

    # If there is no remaining data, the bounds likely do not cover any
    # region with spectra.
    if overlap_base_data.size == 0 or overlap_input_data.size == 0:
        logging.warning(
            warning_type=logging.DataLossWarning,
            message=(
                "No valid data remaining within the provided bounds; NaN scale"
                " factor returned."
            ),
        )
        return np.nan, np.nan

    # We determine the scale factor; though we initially apply a guess so that
    # we can better leverage the precision of the weighted average.
    guess_scale_factor = np.nanmedian(clean_base_data / clean_input_data)
    guess_input_data, guess_input_uncertainty = lezargus.library.math.multiply(
        multiplier=clean_input_data,
        multiplicand=guess_scale_factor,
        multiplier_uncertainty=clean_input_uncertainty,
        multiplicand_uncertainty=0,
    )

    # We then figure out the fine tuned scale factor using the weighted
    # average, allowing us to use the uncertainty properly.
    base_input_ratio, base_input_ratio_uncertainty = (
        lezargus.library.math.divide(
            numerator=clean_base_data,
            denominator=guess_input_data,
            numerator_uncertainty=clean_base_uncertainty,
            denominator_uncertainty=guess_input_uncertainty,
        )
    )
    # Using inverse squared weights. We NaN out any zero uncertainty values
    # as they are likely not real.
    using_uncertainty = np.where(
        base_input_ratio_uncertainty == 0,
        np.nan,
        base_input_ratio_uncertainty,
    )
    weights = 1 / using_uncertainty**2
    weights = lezargus.library.math.normalize_weights(weights=weights)

    # Finding the average ratio.
    quantile_cut_ratio = 0.05
    adjust_scale_factor, adjust_scale_uncertainty = (
        lezargus.library.math.weighted_quantile_mean(
            values=base_input_ratio,
            uncertainties=base_input_ratio_uncertainty,
            weights=weights,
            quantile=quantile_cut_ratio,
        )
    )

    # The uncertainty is not fully propagated from the median as it is a firm
    # guess. The actual mean should contain all of the uncertainty needed.
    scale_factor = guess_scale_factor * adjust_scale_factor
    scale_uncertainty = guess_scale_factor * adjust_scale_uncertainty

    return scale_factor, scale_uncertainty


def stitch_wavelengths_discrete(
    *wavelengths: hint.NDArray,
    sample_mode: str = "hierarchy",
) -> hint.NDArray:
    """Stitch only wavelength arrays together.

    This function simply takes input wavelength arrays and outputs a single
    wavelength array which serves as the combination of all of them, depending
    on the sampling mode. For more information, see [[TODO]].

    Parameters
    ----------
    *wavelengths : ndarray
        Positional arguments for the wavelength arrays we are combining. We
        remove any NaNs.
    sample_mode : string, default = "hierarchy"
        The sampling mode of stitching that we will be doing. It must be one of
        the following modes:

        - `merge`: We just combine them as one array, ignoring the sampling
          of the input wavelength arrays.
        - `hierarchy`: We combine each wavelength with those first input taking
          precedence within their wavelength limits.

    Returns
    -------
    stitched_wavelength_points : ndarray
        The combined wavelength.

    """
    # We need to determine the sampling mode for combining the wavelengths.
    sample_mode = sample_mode.casefold()
    stitched_wavelength_points = []
    if sample_mode == "merge":
        # We are sampling based on total interlacing, without care. We just
        # merge the arrays.
        # Cleaning the arrays first.
        wavelengths = lezargus.library.sanitize.clean_finite_arrays(
            *wavelengths,
        )
        # And just combining them.
        for wavedex in wavelengths:
            stitched_wavelength_points = (
                stitched_wavelength_points + wavedex.tolist()
            )
    elif sample_mode == "hierarchy":
        # We combine the spectra hierarchically, taking into account the
        # minimum and maximum bounds of the higher level spectrum. We first
        # start with a case that is always true.
        min_hist = [+np.inf]
        max_hist = [-np.inf]
        for wavedex in wavelengths:
            # We only add points in wave bands that have not already been
            # covered, we use this by checking the history.
            valid_points = np.full_like(wavedex, True, dtype=bool)
            for mindex, maxdex in zip(min_hist, max_hist, strict=True):
                # If any of the points were within the historical bounds,
                # they are invalid.
                valid_points = valid_points & ~(
                    (mindex <= wavedex) & (wavedex <= maxdex)
                )
            # We add only the valid points to the combined wavelength.
            stitched_wavelength_points = (
                stitched_wavelength_points + wavedex[valid_points].tolist()
            )
            # And we also update the minimum and maximum history to establish
            # this spectrum for the provided region.
            min_hist.append(np.nanmin(wavedex))
            max_hist.append(np.nanmax(wavedex))
    else:
        # The provided mode is not one of the supported ones.
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The input sample mode {sample_mode} is not a supported"
                " option."
            ),
        )

    # Lastly, we sort as none of the algorithms above ensure a sorted
    # wavelength array.
    stitched_wavelength_points = np.sort(stitched_wavelength_points)
    return stitched_wavelength_points


def stitch_spectra_functional(
    wavelength_functions: list[hint.Callable[[hint.NDArray], hint.NDArray]],
    data_functions: list[hint.Callable[[hint.NDArray], hint.NDArray]],
    uncertainty_functions: (
        list[hint.Callable[[hint.NDArray], hint.NDArray]] | None
    ) = None,
    weight_functions: (
        list[hint.Callable[[hint.NDArray], hint.NDArray]] | None
    ) = None,
    average_routine: hint.Callable[
        [hint.NDArray, hint.NDArray, hint.NDArray],
        tuple[float, float],
    ] = None,
    interpolate_routine: type[hint.Generic1DInterpolate] | None = None,
    reference_wavelength: hint.NDArray = None,
) -> tuple[
    hint.Callable[[hint.NDArray], hint.NDArray],
    hint.Callable[[hint.NDArray], hint.NDArray],
    hint.Callable[[hint.NDArray], hint.NDArray],
]:
    R"""Stitch spectra functions together.

    We take functional forms of the wavelength, data, uncertainty, and weight
    (in the form of f(wave) = result), and determine the average spectrum.
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
    uncertainty_functions : list[Callable], default = None
        The list of the uncertainty function. The inputs to these functions
        should be the wavelength.
    weight_functions : list[Callable], default = None
        The list of the weight function. The weights are passed to the
        averaging routine to properly weight the average. If None, we assume
        equal weights.
    average_routine : Callable, default = None
        The averaging function. It must be able to support the propagation of
        uncertainties and weights. As such, it should have the input form of
        :math:`\text{avg}(x, \sigma, w) \rightarrow \bar{x} \pm \sigma`.
        If None, we use a standard weighted average, ignoring NaNs.
    interpolate_routine : Generic1DInterpolate, default = None
        The 1D interpolation class used to handle interpolation.
    reference_wavelength : ndarray, default = None
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
    stitched_wavelength_function : Callable
        The functional form of the average wavelength.
    stitched_data_function : Callable
        The functional form of the average data.
    stitched_uncertainty_function : Callable
        The functional form of the propagated uncertainties.

    """
    # We first determine the defaults.
    if uncertainty_functions is None:
        uncertainty_functions = [
            np.zeros_like for __ in range(len(wavelength_functions))
        ]
    if weight_functions is None:
        weight_functions = [
            np.ones_like for __ in range(len(wavelength_functions))
        ]
    if average_routine is None:
        average_routine = lezargus.library.math.nan_weighted_mean

    # If a custom routine is provided, we need to make sure it is the right
    # type. Otherwise, we just use a default spline interpolator.
    interpolate_routine = (
        lezargus.library.interpolate.Linear1DInterpolate
        if interpolate_routine is None
        else interpolate_routine
    )
    if not issubclass(
        interpolate_routine,
        lezargus.library.interpolate.Generic1DInterpolate,
    ):
        logging.error(
            error_type=logging.InputError,
            message=(
                "Interpolation routine class not of the expected type"
                f" Generic1DInterpolate, instead is {interpolate_routine}"
            ),
        )

    # And we also determine the reference points, which is vaguely based on
    # the atmospheric optical and infrared windows.
    if reference_wavelength is None:
        logging.error(
            error_type=logging.ToDoError,
            message=(
                "Stitch functional default bounds should be configuration"
                " filed."
            ),
        )
        reference_wavelength = np.linspace(0.30, 5.50, 1000000)
    else:
        reference_wavelength = np.sort(
            *lezargus.library.sanitize.clean_finite_arrays(
                reference_wavelength,
            ),
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
                "The provided lengths of the wavelength,"
                f" ={len(wavelength_functions)}; data, ={len(data_functions)};"
                f" uncertainty, ={len(uncertainty_functions)}; and weight,"
                f" ={len(weight_functions)}, function lists are of different"
                " sizes and are not parallel."
            ),
        )

    # We next compute needed discrete values from the functional forms. We
    # can also properly stack them in an array as they are all aligned with
    # the reference points.
    wavelength_points = np.array(
        [
            functiondex(reference_wavelength)
            for functiondex in wavelength_functions
        ],
    )
    data_points = np.array(
        [functiondex(reference_wavelength) for functiondex in data_functions],
    )
    uncertainty_points = np.array(
        [
            functiondex(reference_wavelength)
            for functiondex in uncertainty_functions
        ],
    )
    weight_points = np.array(
        [functiondex(reference_wavelength) for functiondex in weight_functions],
    )

    # We use the user's provided average function, but we adapt for the case
    # where there is no valid data within the range, we just return NaN.
    def average_handle_no_data(
        _values: hint.NDArray,
        _uncertainty: hint.NDArray,
        _weights: hint.NDArray,
    ) -> tuple[float, float]:
        """Extend the average fraction to handle no usable data.

        If there is no usable data, we return NaN for both outputs.

        Parameters
        ----------
        _values : ndarray
            The data values.
        _uncertainty : ndarray
            The uncertainties of the data.
        _weights : ndarray
            The average weights.

        Returns
        -------
        average_value : float
            The average.
        uncertainty_value : float
            The uncertainty on the average as propagated.

        """
        # We clean out the data, this is the primary way to determine if there
        # is usable data or not.
        clean_values, clean_uncertainties, clean_weights = (
            lezargus.library.sanitize.clean_finite_arrays(
                _values,
                _uncertainty,
                _weights,
            )
        )
        # If any of the arrays are blank, there are no clean values to use.
        if (
            clean_values.size == 0
            or clean_uncertainties.size == 0
            or clean_weights.size == 0
        ):
            average_value = np.nan
            uncertainty_value = np.nan
        else:
            # The data has at least one value so an average can be determined.
            # We pass the NaNs to the average function as it is assumed that
            # they can handle it.
            average_value, uncertainty_value = average_routine(
                _values,
                _uncertainty,
                _weights,
            )
        # All done.
        return average_value, uncertainty_value

    # We determine the average of all of the points using the provided
    # averaging routine. We do not actually need the reference points at this
    # time.
    average_wavelength = []
    average_data = []
    average_uncertainty = []

    for index, __ in enumerate(reference_wavelength):
        # We determine the average wavelength, for consistency. We do not
        # care for the computed uncertainty in the wavelength; the typical
        # trash variable is being used for the loop so we use something else
        # just in case.
        temp_wave, ___ = average_handle_no_data(
            _values=wavelength_points[:, index],
            _uncertainty=0,
            _weights=weight_points[:, index],
        )
        temp_data, temp_uncertainty = average_handle_no_data(
            _values=data_points[:, index],
            _uncertainty=uncertainty_points[:, index],
            _weights=weight_points[:, index],
        )
        # Adding the points.
        average_wavelength.append(temp_wave)
        average_data.append(temp_data)
        average_uncertainty.append(temp_uncertainty)
    # Making things into arrays is easier.
    average_wavelength = np.array(average_wavelength)
    average_data = np.array(average_data)
    average_uncertainty = np.array(average_uncertainty)

    # We need to compute the new functional form of the wavelength, data,
    # and uncertainty. However, we need to keep in mind of any NaNs which were
    # present before creating the new interpolator. All of the interpolators
    # remove NaNs and so we reintroduce them by assuming a NaN gap where the
    # data spacing is strictly larger than the largest spacing of data points.
    reference_gap = lezargus.library.interpolate.get_smallest_gap(
        wavelength=average_wavelength,
    )

    # Building the interpolators.
    stitched_wavelength_function = interpolate_routine.template_class(
        extrapolate=False,
        extrapolate_fill=np.nan,
        gap=reference_gap,
    )(x=average_wavelength, v=average_wavelength)

    stitched_data_function = interpolate_routine.template_class(
        extrapolate=False,
        extrapolate_fill=np.nan,
        gap=reference_gap,
    )(x=average_wavelength, v=average_data)

    stitched_uncertainty_function = interpolate_routine.template_class(
        extrapolate=False,
        extrapolate_fill=np.nan,
        gap=reference_gap,
    )(x=average_wavelength, v=average_uncertainty)

    # All done.
    return (
        stitched_wavelength_function,
        stitched_data_function,
        stitched_uncertainty_function,
    )


def stitch_spectra_discrete(
    wavelength_arrays: list[hint.NDArray],
    data_arrays: list[hint.NDArray],
    uncertainty_arrays: list[hint.NDArray] | None = None,
    weight_arrays: list[hint.NDArray] | None = None,
    average_routine: (
        hint.Callable[
            [hint.NDArray, hint.NDArray, hint.NDArray],
            tuple[float, float],
        ]
        | None
    ) = None,
    interpolate_routine: type[hint.Generic1DInterpolate] | None = None,
    reference_wavelength: hint.NDArray | None = None,
) -> tuple[hint.NDArray, hint.NDArray, hint.NDArray]:
    R"""Stitch spectra data arrays together.

    We take the discrete point data of spectra (wavelength, data, and
    uncertainty), along with weights, to stitch together and determine the
    average spectrum. The scale of the data and uncertainty should be of the
    same scale, as should the wavelength and reference points.

    This function serves as the intended way to stitch spectra, though
    :py:func:`stitch.stitch_spectra_functional` is the
    work-horse function and more information can be found there. We build
    interpolators for said function using the input data and attempt to
    guess for any gaps.

    Parameters
    ----------
    wavelength_arrays : list[ndarray]
        The list of the wavelength arrays representing each spectrum.
    data_arrays : list[ndarray]
        The list of the data arrays representing each spectrum.
    uncertainty_arrays : list[ndarray], default = None
        The list of the uncertainty arrays representing the data of each
        spectrum. The scale of the data arrays and uncertainty arrays should be
        the same. If None, we default to no uncertainty.
    weight_arrays : list[ndarray], default = None
        The list of the weight arrays to weight each spectrum for the average
        routine. If None, we assume uniform weights.
    average_routine : Callable, default = None
        The averaging function. It must be able to support the propagation of
        uncertainties and weights. As such, it should have the form of
        :math:`\text{avg}(x, \sigma, w) \rightarrow \bar{x} \pm \sigma`.
        If None, we use a standard weighted average, ignoring NaNs.
    interpolate_routine : Generic1DInterpolate, default = None
        The 1D interpolation routine class used to handle interpolation.
    reference_wavelength : ndarray, default = None
        The reference wavelength is where the stitched spectrum wavelength
        values should be. If None, we attempt to construct it based on the
        overlap and ordering of the input wavelength arrays. We do not accept
        NaNs in either cases and remove them.

    Returns
    -------
    stitched_wavelength_points : ndarray
        The discrete data points of the average wavelength.
    stitched_data_points : ndarray
        The discrete data points of the average data.
    stitched_uncertainty_points : ndarray
        The discrete data points of the propagated uncertainties.

    """
    # We first determine the defaults.
    if uncertainty_arrays is None:
        uncertainty_arrays = [
            np.zeros_like(wavedex) for wavedex in wavelength_arrays
        ]
    if weight_arrays is None:
        weight_arrays = [np.ones_like(wavedex) for wavedex in wavelength_arrays]
    if average_routine is None:
        average_routine = lezargus.library.math.nan_weighted_mean

    # The weights should be normalized.
    weight_arrays = [
        lezargus.library.math.normalize_weights(weights=weightdex)
        for weightdex in weight_arrays
    ]

    # If a custom routine is provided, we need to make sure it is the right
    # type. Otherwise, we just use a default spline interpolator.
    interpolate_routine = (
        lezargus.library.interpolate.Linear1DInterpolate
        if interpolate_routine is None
        else interpolate_routine
    )
    if not issubclass(
        interpolate_routine,
        lezargus.library.interpolate.Generic1DInterpolate,
    ):
        logging.error(
            error_type=logging.InputError,
            message=(
                "Interpolation class not of the expected type"
                f" Generic1DInterpolate, instead is {interpolate_routine}"
            ),
        )

    # We need to determine the reference wavelength, either from the
    # information provided or a custom inputted value.
    if reference_wavelength is None:
        # We try and parse the reference wavelength; we assume the defaults of
        # this function is good enough.
        reference_wavelength = stitch_wavelengths_discrete(*wavelength_arrays)
    # Still sorting it and making sure it is clean.
    reference_wavelength = np.sort(
        *lezargus.library.sanitize.clean_finite_arrays(reference_wavelength),
    )

    # We next need to check the shape and the broadcasting of values for all
    # data. This is mostly a check to make sure that the shapes are compatible
    # and to also format them to better broadcasted versions (in the event) of
    # single value entires.
    wavelength_broadcasts = []
    data_broadcasts = []
    uncertainty_broadcasts = []
    weight_broadcasts = []
    for index, (wavedex, datadex, uncertdex, weightdex) in enumerate(
        zip(
            wavelength_arrays,
            data_arrays,
            uncertainty_arrays,
            weight_arrays,
            strict=True,
        ),
    ):
        # We assume that the wavelength array is the canonical data shape
        # for each and every data.
        temp_wave = wavedex
        # We now check for all of the other arrays, checking notating any
        # irregularities. We of course log if there is an issue.
        verify_data, temp_data = (
            lezargus.library.sanitize.verify_broadcastability(
                reference_array=temp_wave,
                test_array=datadex,
            )
        )
        verify_uncert, temp_uncert = (
            lezargus.library.sanitize.verify_broadcastability(
                reference_array=temp_wave,
                test_array=uncertdex,
            )
        )
        verify_weight, temp_weight = (
            lezargus.library.sanitize.verify_broadcastability(
                reference_array=temp_wave,
                test_array=weightdex,
            )
        )
        if not (verify_data and verify_uncert and verify_weight):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The {index}-th array input have incompatible shapes with"
                    f" the wavelength, {wavedex.shape}; data, {datadex.shape};"
                    f" uncertainty, {uncertdex.shape}; and weight"
                    f" {weightdex.shape} arrays all having the listed"
                    " incompatible and unbroadcastable shapes."
                ),
            )

        # We use the broadcasted arrays as the main ones we will use.
        wavelength_broadcasts.append(temp_wave)
        data_broadcasts.append(temp_data)
        uncertainty_broadcasts.append(temp_uncert)
        weight_broadcasts.append(temp_weight)

    # We need to build the interpolators for each section of the spectrum, as
    # it is what we will input.
    # We attempt to find the gaps in the data, assuming that the wavelength
    # arrays are complete.
    gap_guess = [
        lezargus.library.interpolate.get_smallest_gap(wavelength=wavedex)
        for wavedex in wavelength_broadcasts
    ]
    # Building the interpolators. if there is any array which does not have
    # any usable data, where the interpolator cannot be built, we ignore it.
    wavelength_interpolators = []
    data_interpolators = []
    uncertainty_interpolators = []
    weight_interpolators = []
    for wavedex, datadex, uncertdex, weightdex, gapdex in zip(
        wavelength_broadcasts,
        data_broadcasts,
        uncertainty_broadcasts,
        weight_broadcasts,
        gap_guess,
        strict=True,
    ):
        # We clean up all of the data, the gap is not included.
        clean_wave, clean_data, clean_uncert, clean_weight = (
            lezargus.library.sanitize.clean_finite_arrays(
                wavedex,
                datadex,
                uncertdex,
                weightdex,
            )
        )
        # If any of the arrays are lacking enough data points for interpolation
        # (2), then  we cannot build an interpolator for it.
        n_points_req = 2
        if (
            clean_wave.size < n_points_req
            or clean_data.size < n_points_req
            or clean_uncert.size < n_points_req
        ):
            continue
        # Otherwise, we build the interpolators.
        wavelength_interpolators.append(
            interpolate_routine.template_class(
                extrapolate=False,
                extrapolate_fill=np.nan,
                gap=gapdex,
            )(x=wavedex, v=wavedex),
        )
        data_interpolators.append(
            interpolate_routine.template_class(
                extrapolate=False,
                extrapolate_fill=np.nan,
                gap=gapdex,
            )(x=wavedex, v=datadex),
        )
        uncertainty_interpolators.append(
            interpolate_routine.template_class(
                extrapolate=False,
                extrapolate_fill=np.nan,
                gap=gapdex,
            )(x=wavedex, v=uncertdex),
        )

        # The weight interpolator is a little different as we just want the
        # nearest weight as we assume the weight is a section as opposed to a
        # function.
        weight_interpolators.append(
            lezargus.library.interpolate.Nearest1DInterpolate(
                x=clean_wave,
                v=clean_weight,
                extrapolate=True,
            ),
        )

    # Now we determine the stitched interpolator.
    (
        stitched_wavelength_function,
        stitched_data_function,
        stitched_uncertainty_function,
    ) = stitch_spectra_functional(
        wavelength_functions=wavelength_interpolators,
        data_functions=data_interpolators,
        uncertainty_functions=uncertainty_interpolators,
        weight_functions=weight_interpolators,
        average_routine=average_routine,
        interpolate_routine=interpolate_routine,
        reference_wavelength=reference_wavelength,
    )

    # And, using the reference wavelength, we compute the data values.
    stitched_wavelength_points = stitched_wavelength_function(
        reference_wavelength,
    )
    stitched_data_points = stitched_data_function(reference_wavelength)
    stitched_uncertainty_points = stitched_uncertainty_function(
        reference_wavelength,
    )
    return (
        stitched_wavelength_points,
        stitched_data_points,
        stitched_uncertainty_points,
    )

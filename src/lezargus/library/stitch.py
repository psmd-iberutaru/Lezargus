"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
"""

import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


def stitch_wavelengths_discrete(
    *wavelengths: hint.ndarray,
    sample_mode: str = "hierarchy",
) -> hint.ndarray:
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
        wavelengths = library.array.clean_finite_arrays(*wavelengths)
        # And just combining them.
        for wavedex in wavelengths:
            stitched_wavelength_points = (
                stitched_wavelength_points + wavedex.tolist()
            )
    elif sample_mode == "hierarchy":
        # We combine the spectra hierarchically, taking into account the
        # minimum and maximum bounds of the higher level spectra. We first
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
            # this spectra for the provided region.
            min_hist.append(np.nanmin(wavedex))
            max_hist.append(np.nanmax(wavedex))
    else:
        # The provided mode does not exist.
        logging.critical(
            critical_type=logging.InputError,
            message=f"The input sample mode {sample_mode} does not exist.",
        )

    # Lastly, we sort as none of the algorithms above ensure a sorted
    # wavelength array.
    stitched_wavelength_points = np.sort(stitched_wavelength_points)
    return stitched_wavelength_points


def stitch_spectra_functional(
    wavelength_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    data_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    uncertainty_functions: (
        list[hint.Callable[[hint.ndarray], hint.ndarray]] | None
    ) = None,
    weight_functions: (
        list[hint.Callable[[hint.ndarray], hint.ndarray]] | None
    ) = None,
    average_routine: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ] = None,
    interpolation_routine: hint.Callable[
        [hint.ndarray, hint.ndarray],
        hint.Callable[[hint.ndarray], hint.ndarray],
    ] = None,
    reference_wavelength: hint.ndarray = None,
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
    interpolation_routine : Callable, default = None
        The interpolation routine factory which we use to interpolate each of
        the spectra to each other wavelength frame. It should have the input
        form of :math:`\text{ipr}(x,y) \rightarrow f:x \mapsto y`. If None, we
        default to a cubic spline, handling gaps.
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
        average_routine = library.uncertainty.nan_weighted_mean
    if interpolation_routine is None:
        interpolation_routine_wrapped = (
            library.interpolate.cubic_1d_interpolate_bounds_gap_factory
        )
    else:
        # If a custom routine is provided, then we need to make sure it
        # can handle gaps in the data, and the limits of the interpolation.
        def custom_interpolate_bounds(
            x: hint.ndarray,
            y: hint.ndarray,
        ) -> hint.Callable[[hint.ndarray], hint.ndarray]:
            return library.interpolate.custom_1d_interpolate_bounds_factory(
                interpolation=interpolation_routine,
                x=x,
                y=y,
            )

        def interpolation_routine_wrapped(
            x: hint.ndarray,
            y: hint.ndarray,
            gap_size: float,
        ) -> hint.Callable[[hint.ndarray], hint.ndarray]:
            return library.interpolate.custom_1d_interpolate_gap_factory(
                interpolation=custom_interpolate_bounds,
                x=x,
                y=y,
                gap_size=gap_size,
            )

    # And we also determine the reference points, which is vaguely based on
    # the atmospheric optical and infrared windows.
    if reference_wavelength is None:
        reference_wavelength = np.linspace(0.30, 5.50, 1000000)
    else:
        reference_wavelength = np.sort(
            *library.array.clean_finite_arrays(reference_wavelength),
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
        _values: hint.ndarray,
        _uncertainty: hint.ndarray,
        _weights: hint.ndarray,
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
            library.array.clean_finite_arrays(_values, _uncertainty, _weights)
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
    reference_gap = (1 + 1e-3) * np.nanmax(
        reference_wavelength[1:] - reference_wavelength[:-1],
    )

    # Building the interpolators.
    stitched_wavelength_function = interpolation_routine_wrapped(
        x=average_wavelength,
        y=average_wavelength,
        gap_size=reference_gap,
    )
    stitched_data_function = interpolation_routine_wrapped(
        x=average_wavelength,
        y=average_data,
        gap_size=reference_gap,
    )
    stitched_uncertainty_function = interpolation_routine_wrapped(
        x=average_wavelength,
        y=average_uncertainty,
        gap_size=reference_gap,
    )
    # All done.
    return (
        stitched_wavelength_function,
        stitched_data_function,
        stitched_uncertainty_function,
    )


def stitch_spectra_discrete(
    wavelength_arrays: list[hint.ndarray],
    data_arrays: list[hint.ndarray],
    uncertainty_arrays: list[hint.ndarray] | None = None,
    weight_arrays: list[hint.ndarray] | None = None,
    average_routine: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ] = None,
    interpolation_routine: hint.Callable[
        [hint.ndarray, hint.ndarray],
        hint.Callable[[hint.ndarray], hint.ndarray],
    ] = None,
    reference_wavelength: hint.ndarray = None,
) -> tuple[hint.ndarray, hint.ndarray, hint.ndarray]:
    R"""Stitch spectra data arrays together.

    We take the discrete point data of spectra (wavelength, data, and
    uncertainty), along with weights, to stitch together and determine the
    average spectra. The scale of the data and uncertainty should be of the
    same scale, as should the wavelength and reference points.

    This function serves as the intended way to stitch spectra, though
    :py:func:`lezargus.library.stitch.stitch_spectra_functional` is the
    work-horse function and more information can be found there. We build
    interpolators for said function using the input data and attempt to
    guess for any gaps.

    Parameters
    ----------
    wavelength_arrays : list[ndarray]
        The list of the wavelength arrays representing each spectra.
    data_arrays : list[ndarray]
        The list of the data arrays representing each spectra.
    uncertainty_arrays : list[ndarray], default = None
        The list of the uncertainty arrays representing the data of each
        spectra. The scale of the data arrays and uncertainty arrays should be
        the same. If None, we default to no uncertainty.
    weight_arrays : list[ndarray], default = None
        The list of the weight arrays to weight each spectra for the average
        routine. If None, we assume uniform weights.
    average_routine : Callable, default = None
        The averaging function. It must be able to support the propagation of
        uncertainties and weights. As such, it should have the form of
        :math:`\text{avg}(x, \sigma, w) \rightarrow \bar{x} \pm \sigma`.
        If None, we use a standard weighted average, ignoring NaNs.
    interpolation_routine : Callable, default = None
        The interpolation routine factory which we use to interpolate each of
        the spectra to each other wavelength frame. It should have the input
        form of :math:`\text{ipr}(x,y) \rightarrow f:x \mapsto y`. If None, we
        default to a cubic spline, handling gaps.
    reference_wavelength : ndarray, default = None
        The reference wavelength is where the stitched spectra wavelength
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
        average_routine = library.uncertainty.nan_weighted_mean
    if interpolation_routine is None:
        interpolation_routine_wrapped = (
            library.interpolate.cubic_1d_interpolate_bounds_gap_factory
        )
    else:
        # If a custom routine is provided, then we need to make sure it
        # can handle gaps in the data, and the limits of the interpolation.
        def custom_interpolate_bounds(
            x: hint.ndarray,
            y: hint.ndarray,
        ) -> hint.Callable[[hint.ndarray], hint.ndarray]:
            return library.interpolate.custom_1d_interpolate_bounds_factory(
                interpolation=interpolation_routine,
                x=x,
                y=y,
            )

        def interpolation_routine_wrapped(
            x: hint.ndarray,
            y: hint.ndarray,
            gap_size: float,
        ) -> hint.Callable[[hint.ndarray], hint.ndarray]:
            return library.interpolate.custom_1d_interpolate_gap_factory(
                interpolation=custom_interpolate_bounds,
                x=x,
                y=y,
                gap_size=gap_size,
            )

    # And we also determine the reference points, which is vaguely based on
    # the atmospheric optical and infrared windows.
    if reference_wavelength is None:
        # We try and parse the reference wavelength; we assume the defaults of
        # this function is good enough.
        reference_wavelength = stitch_wavelengths_discrete(*wavelength_arrays)
    # Still sorting it and making sure it is clean.
    reference_wavelength = np.sort(
        *library.array.clean_finite_arrays(reference_wavelength),
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
        verify_data, temp_data = library.array.verify_shape_compatibility(
            reference_array=temp_wave,
            test_array=datadex,
            return_broadcast=True,
        )
        verify_uncert, temp_uncert = library.array.verify_shape_compatibility(
            reference_array=temp_wave,
            test_array=uncertdex,
            return_broadcast=True,
        )
        verify_weight, temp_weight = library.array.verify_shape_compatibility(
            reference_array=temp_wave,
            test_array=weightdex,
            return_broadcast=True,
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

    # We need to build the interpolators for each section of the spectra, as
    # it is what we will input.
    # We attempt to find the gaps in the data, assuming that the wavelength
    # arrays are complete.
    gap_guess = [
        np.nanmax(np.abs(wavedex[1:] - wavedex[:-1]))
        for wavedex in wavelength_broadcasts
    ]
    # Building the interpolators.
    wavelength_interpolators = [
        interpolation_routine_wrapped(x=wavedex, y=wavedex, gap_size=gapdex)
        for wavedex, gapdex in zip(
            wavelength_broadcasts,
            gap_guess,
            strict=True,
        )
    ]
    data_interpolators = [
        interpolation_routine_wrapped(x=wavedex, y=datadex, gap_size=gapdex)
        for wavedex, datadex, gapdex in zip(
            wavelength_broadcasts,
            data_broadcasts,
            gap_guess,
            strict=True,
        )
    ]
    uncertainty_interpolators = [
        interpolation_routine_wrapped(x=wavedex, y=uncertdex, gap_size=gapdex)
        for wavedex, uncertdex, gapdex in zip(
            wavelength_broadcasts,
            uncertainty_broadcasts,
            gap_guess,
            strict=True,
        )
    ]
    # The weight interpolator is a little different as we just want the
    # nearest weight as we assume the weight is a section as opposed to a
    # function.
    weight_interpolators = [
        library.interpolate.nearest_neighbor_1d_interpolate_factory(
            x=wavedex,
            y=weightdex,
        )
        for wavedex, weightdex in zip(
            wavelength_broadcasts,
            weight_broadcasts,
            strict=True,
        )
    ]

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
        interpolation_routine=interpolation_routine,
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

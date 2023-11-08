"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
"""


import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging




def stitch_spectra_arrays(
    wavelength: list[hint.ndarray],
    data: list[hint.ndarray],
    uncertainty: list[hint.ndarray] | None = None,
    weight: list[hint.ndarray] | None = None,
    average_function: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ] = None,
) -> tuple[hint.ndarray, hint.ndarray, hint.ndarray]:
    """Stitch and average different spectra together using a custom function.

    This function takes any number of spectra, represented as parallel
    arrays of wavelength, spectral flux or data, and (optionally) weights,
    and combines them. We stitch them together; the formal method is described
    in [[TODO]]. In summary, we interpolate over overlapping regions and
    combine the data according to the average function.

    Note, the average function provided is re-wrapped to properly handle NaNs.
    NaNs in the data are ignored and not used for any stitching. If NaNs are
    desired to be preserved and propagated, you will need to re-NaN values
    after the stitched spectra; see [[TODO; MAKE NAN PROPAGATE VERSION]].

    Parameters
    ----------
    wavelength : list[ndarray]
        A list of wavelengths for stitching multiple spectra arrays together.
        Each entry in the list must have a corresponding entry in the data list.
    data : list[ndarray]
        A list of spectra/data for stitching multiple spectra arrays together.
        Each entry in the list must have a corresponding entry in the
        wavelength list.
    uncertainty : list[ndarray], default = None
        A list of the uncertainties in the data for stitching. Each entry in
        the list must have a corresponding entry in the wavelength and
        data list, or None. If the entry (or entire input) is None, we assume
        zero uncertainty for the corresponding input spectra arrays.
    weight : list[ndarray], default = None
        A list of the weights in the data for stitching. Each entry in
        the list must have a corresponding entry in the wavelength and
        data list, or None. If the entry (or entire input) is None, we assume
        uniform weights for the corresponding input spectra arrays.
    average_function : Callable, str, default = None
        The function used to average all of the spectra together. 
        It must also be able to accept weights and propagate uncertainties.
        If None, we default to the weighted mean. Namely, it must be of the
        form f(val, uncert, weight) = avg, uncert.

    .. note::
        This function uses negative infinity internally for calculations as a
        flag value. Unexpected behavior might happen if the data contains
        negative infinity values for any of the input or as a hard return of
        the average function. (We handle inputs of negative infinity.) We
        suggest using NaN instead.

    Returns
    -------
    stitch_wavelength : ndarray
        The wavelength of the snitched spectra made from the input spectra.
    stitch_spectra : ndarray
        The spectral data of the stitched spectra made from the input spectra.
    stitch_uncertainty : ndarray
        The spectral uncertainty of the stitched spectra made from the input
        spectra.
    """
    # We first need to handle all of the default cases. Simple default cases
    # are first. We just replace the Nones with what is documented. We assume
    # that the wavelength array is primary.
    uncertainty = (
        [np.zeros_like(wavedex) for wavedex in wavelength]
        if uncertainty is None
        else uncertainty
    )
    weight = (
        [np.ones_like(wavedex) for wavedex in wavelength]
        if weight is None
        else weight
    )
    average_function = (
        library.uncertainty.weighted_mean
        if average_function is None
        else average_function
    )
    # We start with reformatting the wavelength, data, uncertainty, and weights
    # into structures with all the same shape. We assume that the wavelength
    # list is the primary array to base it all off of.
    # We first need to check the parallelism of the arrays.
    if not (len(wavelength) == len(data) == len(uncertainty) == len(weight)):
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "The provided lengths of the wavelength, ={wv}; data, ={da};"
                " uncertainty, ={un}; and weight, ={wg}, lists are of different"
                " sizes and are not parallel.".format(
                    wv=len(wavelength),
                    da=len(data),
                    un=len(uncertainty),
                    wg=len(weight),
                )
            ),
        )
    # We now might be able to reformat the arrays. We store them as lists. The
    # empty list instantiation fanciness done here is to satisfy PLR0915 from
    # the linter.
    full_wavelengths, full_data, full_uncertainty, full_weights = (
        [] for __ in range(4)
    )
    for index, wavedex, datadex, uncertdex, weightdex in zip(
        range(len(wavelength)),
        wavelength,
        data,
        uncertainty,
        weight,
        strict=True,
    ):
        # If the wavelengths and data are not the same size, then the data
        # means nothing to us as it is no longer wave-flux pairs.
        data_verify, using_data = library.array.verify_shape_compatibility(
            reference_array=wavedex,
            test_array=datadex,
            return_broadcast=True,
        )
        if not data_verify:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "The {n}-th array input of wavelength and data"
                    " lists have differing shapes: {wv} != {da}".format(
                        n=index,
                        wv=wavedex.shape,
                        da=datadex.shape,
                    )
                ),
            )

        # We broadcast single value uncertainties, else we check the shapes
        # again.
        uncert_verify, using_uncert = library.array.verify_shape_compatibility(
            reference_array=wavedex,
            test_array=uncertdex,
            return_broadcast=True,
        )
        if not uncert_verify:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "The {n}-th array input of wavelength and uncertainty"
                    " lists have differing shapes: {wv} != {un}".format(
                        n=index,
                        wv=wavedex.shape,
                        un=uncertdex.shape,
                    )
                ),
            )
        # Similarly, we broadcast the weights.
        weight_verify, using_weight = library.array.verify_shape_compatibility(
            reference_array=wavedex,
            test_array=weightdex,
            return_broadcast=True,
        )
        if not weight_verify:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "The {n}-th array input of wavelength and weight"
                    " lists have differing shapes: {wv} != {wg}".format(
                        n=index,
                        wv=wavedex.shape,
                        wg=weightdex.shape,
                    )
                ),
            )
        # If it passed above, we it has been formatted and cleaned up.
        # The wavelength is always considered to be correct.
        full_wavelengths.append(wavedex)
        full_data.append(using_data)
        full_uncertainty.append(using_uncert)
        full_weights.append(using_weight)

    # We only include spectra/interpolators that are valid. If an
    # interpolator fails to build, we do not include any part of its data.
    # As such, we compile the interpolators first to determine the valid
    # wavelength range, then we compute everything afterwards.
    # The empty list instantiation fanciness done here is to satisfy PLR0915
    # from the linter.
    interp_wave, interp_data, interp_uncertainty, interp_weights = (
        [] for __ in range(4)
    )
    for index, wavedex, datadex, uncertdex, weightdex in zip(
        range(len(full_wavelengths)),
        full_wavelengths,
        full_data,
        full_uncertainty,
        full_weights,
        strict=True,
    ):
        try:
            # Attempting to build the interpolators.
            temp_data = __spectra_rewrapped_interpolation_factory(
                base_wave=wavedex,
                base_data=datadex,
                interp_factory=library.wrapper.cubic_interpolate_1d_function,
            )
            temp_uncertainty = __spectra_rewrapped_interpolation_factory(
                base_wave=wavedex,
                base_data=uncertdex,
                interp_factory=library.wrapper.nearest_neighbor_interpolate_1d_function,
            )
            temp_weights = __spectra_rewrapped_interpolation_factory(
                base_wave=wavedex,
                base_data=weightdex,
                interp_factory=library.wrapper.nearest_neighbor_interpolate_1d_function,
            )
        except ValueError:
            # The interpolation cannot be derived, this is likely because there
            # is not enough data in the spectra.
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Interpolation routines cannot be derived the {i} index"
                    " spectra; likely not enough valid data. Skipping.".format(
                        i=index,
                    )
                ),
            )
            continue
        else:
            # Appending them to compile the list.
            interp_wave.append(wavedex)
            interp_data.append(temp_data)
            interp_uncertainty.append(temp_uncertainty)
            interp_weights.append(temp_weights)
    # Finally building the data.
    total_wavelength = np.sort(
        library.wrapper.flatten_list_recursively(interp_wave),
    )
    total_data = np.array(
        [interpdex(total_wavelength) for interpdex in interp_data],
    )
    total_uncertainty = np.array(
        [interpdex(total_wavelength) for interpdex in interp_uncertainty],
    )
    total_weights = np.array(
        [interpdex(total_wavelength) for interpdex in interp_weights],
    )

    # We now average the spectra together, which is a form of stitching. We
    # use the provided averaging function. We redefine the symbols as it is
    # a "new paradigm". We also apply the weights here.
    # The empty list instantiation fanciness done here is to satisfy PLR0915
    # from the linter.
    average_wavelength, average_data, average_uncertainty = (
        [] for __ in range(3)
    )
    for index in range(len(total_wavelength)):
        # We use the averaging function to determine the data. We also
        # use the function's expected uncertainty propagation.
        temp_wave = total_wavelength[index]
        temp_data, temp_uncertainty = (
            __spectra_rewrapped_interpolation_average_function(
                data=total_data[:, index],
                uncertainty=total_uncertainty[:, index],
                weights=total_weights[:, index],
                average_function=average_function,
            )
        )
        average_wavelength.append(temp_wave)
        average_data.append(temp_data)
        average_uncertainty.append(temp_uncertainty)

    # Because this is an average of multiple spectra, there may be duplicate
    # entires for a single wavelength value. Interpolation cannot handle this
    # so we just take the first-most unique values. We avoid trusting the
    # extra sorting of the unique function and just rely on the indexes.
    __, unique_index = np.unique(average_wavelength, return_index=True)
    unique_wavelength = np.array(average_wavelength)[unique_index]
    unique_data = np.array(average_data)[unique_index]
    unique_uncertainty = np.array(average_uncertainty)[unique_index]

    # Building the final wavelength array.
    stitch_wavelength = library.wrapper.combine_overlap_wavelength_array(
        *wavelength,
    )
    # Interpolating out the proper data and uncertainty.
    stitch_data = library.wrapper.cubic_interpolate_1d_function(
        x=unique_wavelength,
        y=unique_data,
    )(stitch_wavelength)
    stitch_uncertainty = (
        library.wrapper.nearest_neighbor_interpolate_1d_function(
            x=unique_wavelength,
            y=unique_uncertainty,
        )(stitch_wavelength)
    )

    # All done.
    return stitch_wavelength, stitch_data, stitch_uncertainty


def __spectra_rewrapped_interpolation_factory(
    base_wave: hint.ndarray,
    base_data: hint.ndarray,
    interp_factory: hint.Callable[
        [hint.ndarray, hint.ndarray],
        hint.Callable[[hint.ndarray], hint.ndarray],
    ],
) -> hint.Callable[[hint.ndarray], hint.ndarray]:
    """Generate data interpolators for stitching for 1D spectra.

    This is an internal function and should not be used outside of it.
    We have small wrappers around the interpolation functions to suit our
    goals. Namely, cubic interpolation with -inf out-of-bounds; maximum
    uncertainty, and minimum weight.

    Parameters
    ----------
    base_wave : ndarray
        The base or original wavelength of the data we are interpolating.
    base_data : ndarray
        The base or original data we are interpolating.
    interp_factory : Callable
        The interpolation function. Its function call must be of the
        form: interp(x, y) = z

    Returns
    -------
    rewrapped_interpolation : Callable
        The wrapped interpolation function, built to handle -inf.
    """
    # We create the interpolation function.
    interpolation = interp_factory(base_wave, base_data)

    # We wrap the interpolation function around special handing for -inf.
    def rewrapped_interpolation(wave: hint.ndarray) -> hint.ndarray:
        """Wrap the interpolation function."""
        # We only interpolate over values which are within the original
        # base wavelength range.
        wave_min = np.nanmin(base_wave)
        wave_max = np.nanmax(base_wave)
        interpolating_index = (wave_min <= wave) & (wave <= wave_max)
        # All other values are flagged as -inf.
        new_data = np.full_like(wave, -np.inf)
        # Calculating.
        new_data[interpolating_index] = interpolation(
            wave[interpolating_index],
        )
        # Return the data.
        return new_data

    # All done.
    return rewrapped_interpolation


def __spectra_rewrapped_interpolation_average_function(
    data: hint.ndarray,
    uncertainty: hint.ndarray,
    weights: hint.ndarray,
    average_function: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ],
) -> float:
    """Average values, per the provided function, handing -inf.

    Parameters
    ----------
    data : ndarray
        The data values which we are going to take an average of.
    uncertainty : ndarray
        The uncertainty of the data.
    weights : ndarray
        The weights for the average.
    average_function : Callable
        The average function we are wrapping around. It must be able to
        handle uncertainties and weights. Namely, it must be of the
        form f(val, uncert, weight) = avg, uncert.

    Returns
    -------
    average_value : float
        The calculated average value.
    average_uncertainty : float
        The uncertainty in the average value.
    """
    # We just do not include any values which are -inf. Normalizing
    # the weights.
    valid = (
        (~np.isneginf(data)) & np.isfinite(uncertainty) & np.isfinite(weights)
    )
    valid_data = data[valid]
    valid_uncertainty = uncertainty[valid]
    valid_weights = weights[valid] / np.nansum(weights[valid])
    # Computing the average. We do not use keywords here because we do not
    # know the input parameters.
    average_value, average_uncertainty = average_function(
        valid_data,
        valid_uncertainty,
        valid_weights,
    )
    return average_value, average_uncertainty

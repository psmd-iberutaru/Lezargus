"""Spectrum data container, holding spectral data.

This module and class primarily deals with spectral data.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import copy

import numpy as np

import lezargus
from lezargus.library import logging
from lezargus.library.container import LezargusContainerArithmetic


class LezargusSpectrum(LezargusContainerArithmetic):
    """Container to hold spectral data and perform operations on it.

    Attributes
    ----------
    For all available attributes, see :py:class:`LezargusContainerArithmetic`.

    """

    def __init__(
        self: LezargusSpectrum,
        wavelength: hint.NDArray,
        data: hint.NDArray,
        uncertainty: hint.NDArray | float | None = None,
        wavelength_unit: str | hint.Unit | None = None,
        data_unit: str | hint.Unit | None = None,
        spectral_scale: float | None = None,
        pixel_scale: float | None = None,
        slice_scale: float | None = None,
        mask: hint.NDArray | None = None,
        flags: hint.NDArray | None = None,
        header: hint.Header | None = None,
    ) -> None:
        """Instantiate the spectrum class.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength axis of the spectral component of the data, if any.
            The unit of wavelength is typically in meters; but, check the
            :py:attr:`wavelength_unit` value.
        data : ndarray
            The data stored in this container. The unit of the flux is typically
            in W m^-2 m^-1; but, check the :py:attr:`data_unit` value.
        uncertainty : ndarray
            The uncertainty in the data of the spectrum. The unit of the
            uncertainty is the same as the data value; per
            :py:attr:`uncertainty_unit`.
        wavelength_unit : Astropy Unit
            The unit of the wavelength array. If None, we assume unit-less.
        data_unit : Astropy Unit
            The unit of the data array. If None, we assume unit-less.
        spectral_scale : float, default = None
            The spectral scale, or spectral resolution, of the spectral
            component, if any. Must be in meters per pixel. Scale is None if
            none is provided.
        pixel_scale : float, default = None
            The E-W, "x" dimension, pixel plate scale of the spatial component,
            if any. Must be in radians per pixel. Scale is None if none
            is provided.
        slice_scale : float, default = None
            The N-S, "y" dimension, pixel slice scale of the spatial component,
            if any. Must be in radians per slice/pixel. Scale is None if none
            is provided.
        mask : ndarray, default = None
            A mask of the data, used to remove problematic areas. Where True,
            the values of the data is considered masked. If None, we assume
            the mask is all clear.
        flags : ndarray, default = None
            Flags of the data. These flags store metadata about the data. If
            None, we assume that there are no harmful flags.
        header : Header, default = None
            A set of header data describing the data. Note that when saving,
            this header is written to disk with minimal processing. We highly
            suggest writing of the metadata to conform to the FITS Header
            specification as much as possible. If None, we just use an
            empty header.

        Returns
        -------
        None

        """
        # The data must be one dimensional.
        container_dimensions = 1
        if len(data.shape) != container_dimensions:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The input data for a LezargusSpectrum instantiation has a"
                    f" shape {data.shape}, which is not the expected one"
                    " dimension."
                ),
            )
        # The wavelength and the data must be parallel, and thus the same
        # shape.
        wavelength = np.array(wavelength, dtype=float)
        data = np.array(data, dtype=float)
        if wavelength.shape != data.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"Wavelength array shape: {wavelength.shape}; data array"
                    f" shape: {data.shape}. The arrays need to be the same"
                    " shape or cast-able to such."
                ),
            )

        # Constructing the original class. We do not deal with WCS here because
        # the base class does not support it. We do not involve units here as
        # well for speed concerns. Both are handled during reading and writing.
        super().__init__(
            wavelength=wavelength,
            data=data,
            uncertainty=uncertainty,
            wavelength_unit=wavelength_unit,
            data_unit=data_unit,
            spectral_scale=spectral_scale,
            pixel_scale=pixel_scale,
            slice_scale=slice_scale,
            mask=mask,
            flags=flags,
            header=header,
        )

    @classmethod
    def read_fits_file(
        cls: type[hint.Self],
        filename: str,
    ) -> hint.Self:
        """Read a Lezargus spectrum FITS file.

        We load a Lezargus FITS file from disk. Note that this should only
        be used for a 1-D spectrum file.

        Parameters
        ----------
        filename : str
            The filename to load.

        Returns
        -------
        spectrum : Self-like
            The LezargusSpectrum class instance.

        """
        # Any pre-processing is done here.
        # Loading the file.
        spectrum = cls._read_fits_file(filename=filename)
        # Any post-processing is done here.
        # All done.
        return spectrum

    @classmethod
    def stitch(
        cls: type[hint.Self],
        *spectra: hint.LezargusSpectrum,
        weights: list[hint.NDArray] | str = "uniform",
        average_routine: hint.Callable[
            [hint.NDArray, hint.NDArray, hint.NDArray],
            tuple[float, float],
        ] = None,
    ) -> hint.Self:
        """Stitch spectra together to make a single spectrum.

        We stitch all of the input spectra. If the spectrum are not already
        to the same scale however, this will result in wildly incorrect
        results. The header information is preserved, though we take what we
        can from the other objects.

        Parameters
        ----------
        *spectra : LezargusSpectrum
            The set of Lezargus spectra which we will stitch together.
        weights : list[ndarray] or str, default = None
            A list of the weights in the data for stitching. Each entry in
            the list must have a corresponding entry in the wavelength and
            data list, or None. For convenience, we provide short-cut inputs
            for the following:

                - `uniform` : Uniform weights.
                - `invar` : Inverse variance weights.
        average_routine : Callable, str, default = None
            The function used to average all of the spectra together.
            It must also be able to accept weights and propagate uncertainties.
            If None, we default to the weighted mean. Namely, it must be of the
            form f(val, uncert, weight) = avg, uncert.

        Returns
        -------
        stitch_spectrum : LezargusSpectrum
            The spectrum after stitching.

        """
        # If there are no spectra to stitch, then we do nothing.
        if len(spectra) == 0:
            # We still warn just in case.
            logging.warning(
                warning_type=logging.InputWarning,
                message="No spectra supplied to stitch.",
            )
            return spectra

        # We need to make sure these are all Lezargus spectrum.
        lz_spectra = []
        for spectrumdex in spectra:
            if not isinstance(spectrumdex, LezargusSpectrum):
                logging.critical(
                    critical_type=logging.InputError,
                    message=(
                        f"Input type {type(spectrumdex)} is not a"
                        " LezargusSpectrum, we cannot use it to stitch."
                    ),
                )
            lz_spectra.append(spectrumdex)

        # We need to translate the weight input.
        if isinstance(weights, str):
            weights = weights.casefold()
            if weights == "uniform":
                # We compute uniform weights.
                using_weights = [
                    np.ones_like(spectrumdex.wavelength)
                    for spectrumdex in lz_spectra
                ]
            elif weights == "invar":
                # We compute weights which are the inverse of the variance
                # in the data.
                using_weights = [
                    1 / spectrumdex.uncertainty**2 for spectrumdex in lz_spectra
                ]
            else:
                using_weights = None
                # A valid shortcut string has not been provided.
                accepted_options = ["uniform", "invar"]
                logging.critical(
                    critical_type=logging.InputError,
                    message=(
                        f"The weight shortcut option {weights} is not valid; it"
                        f" must be one of: {accepted_options}"
                    ),
                )
        else:
            using_weights = weights

        # Next, we stitch together the data for the spectrum.
        (
            stitch_wavelength,
            stitch_data,
            stitch_uncertainty,
        ) = lezargus.library.stitch.stitch_spectra_discrete(
            wavelength_arrays=[
                spectrumdex.wavelength for spectrumdex in lz_spectra
            ],
            data_arrays=[spectrumdex.data for spectrumdex in lz_spectra],
            uncertainty_arrays=[
                spectrumdex.uncertainty for spectrumdex in lz_spectra
            ],
            weight_arrays=using_weights,
            average_routine=average_routine,
            interpolate_routine=None,
        )
        # We also stitch together the flags and the mask. They are handled
        # with a different function.
        logging.error(
            error_type=logging.ToDoError,
            message="Flag and mask stitching not yet supported.",
        )
        stitch_mask = None
        stitch_flags = None

        # We merge the header.
        logging.error(
            error_type=logging.ToDoError,
            message="Header stitching not yet supported.",
        )
        stitch_header = None

        # We compile the new spectrum. We do not expect a subclass but we
        # try and allow it.
        stitch_spectrum = cls(
            wavelength=stitch_wavelength,
            data=stitch_data,
            uncertainty=stitch_uncertainty,
            wavelength_unit=lz_spectra[0].wavelength_unit,
            data_unit=lz_spectra[0].data_unit,
            mask=stitch_mask,
            flags=stitch_flags,
            header=stitch_header,
        )
        # All done.
        return stitch_spectrum

    def write_fits_file(
        self: hint.Self,
        filename: str,
        overwrite: bool = False,
    ) -> hint.Self:
        """Write a Lezargus spectrum FITS file.

        We write a Lezargus FITS file to disk.

        Parameters
        ----------
        filename : str
            The filename to write to.
        overwrite : bool, default = False
            If True, overwrite file conflicts.

        Returns
        -------
        None

        """
        # Any pre-processing is done here.
        # Saving the file.
        self._write_fits_file(filename=filename, overwrite=overwrite)
        # Any post-processing is done here.
        # All done.

    def convolve(
        self: hint.Self,
        kernel: hint.NDArray | None = None,
        kernel_stack: hint.NDArray | None = None,
        kernel_function: hint.Callable | None = None,
    ) -> hint.Self | hint.LezargusSpectrum:
        """Convolve the spectrum with a spectral kernel.

        See py:func:`convolve_spectrum_by_spectral_kernel` for full
        documentation.

        Parameters
        ----------
        kernel : ndarray, default = None
            The static 2D kernel.
        kernel_stack : ndarray, default = None
            The variable 2D kernel stack.
        kernel_function : Callable, default = None
            The dynamic 2D kernel function.

        Returns
        -------
        convolved_spectrum : ndarray
            A near copy of the data cube after convolution.

        """
        # We split up the function just for line length.
        _functionality = lezargus.library.container.functionality
        convolved_spectrum = (
            _functionality.convolve_spectrum_by_spectral_kernel(
                spectrum=self,
                kernel=kernel,
                kernel_stack=kernel_stack,
                kernel_function=kernel_function,
            )
        )
        return convolved_spectrum

    def interpolate(
        self: hint.Self,
        wavelength: hint.NDArray,
        extrapolate: bool = False,
        skip_mask: bool = True,
        skip_flags: bool = True,
    ) -> tuple[
        hint.NDArray,
        hint.NDArray,
        hint.NDArray | None,
        hint.NDArray | None,
    ]:
        """Interpolation calling function for spectrum.

        Each entry is considered a single point to interpolate over.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength values which we are going to interpolate to. The
            units of the data of this array should be the same as the
            wavelength unit stored.
        extrapolate : bool, default = False
            If True, we extrapolate. Otherwise, the edges are NaNs.
        skip_mask : bool, default = True
            If provided, the propagation of data mask through the
            interpolation is skipped. It is computationally a little expensive
            otherwise.
        skip_flags : bool, default = True
            If provided, the propagation of data flags through the
            interpolation is skipped. It is computationally a little expensive
            otherwise.

        Returns
        -------
        interp_data : ndarray
            The interpolated data.
        interp_uncertainty : ndarray
            The interpolated uncertainty.
        interp_mask : ndarray or None
            A best guess attempt at finding the appropriate mask for the
            interpolated data. If skip_mask=True, then we skip the computation
            and return None instead.
        interp_flags : ndarray or None
            A best guess attempt at finding the appropriate flags for the
            interpolated data. If skip_flags=True, then we skip the computation
            and return None instead.

        """
        # Interpolation cannot deal with NaNs, so we exclude any set of data
        # which includes them.
        (
            clean_wavelength,
            clean_data,
            clean_uncertainty,
        ) = lezargus.library.sanitize.clean_finite_arrays(
            self.wavelength,
            self.data,
            self.uncertainty,
        )

        # If the wavelengths we are using to interpolate to are not all
        # numbers, it is a good idea to warn. It is not a good idea to change
        # the input the user provided.
        if not np.all(np.isfinite(wavelength)):
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "The input wavelength for interpolation are not all finite."
                ),
            )

        # As a sanity check, we check if we are trying to interpolate outside
        # of our data range.
        overlap = lezargus.library.wrapper.wavelength_overlap_fraction(
            base=clean_wavelength,
            contain=wavelength,
        )
        if overlap < 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Interpolation is attempted at a wavelength beyond the"
                    " domain of wavelengths of this spectrum. The overlap"
                    f" fraction is {overlap}."
                ),
            )

        # The interpolated data for both the data itself and uncertainty.
        # We use gaps to remove any unwanted data, assuming the cleaned
        # wavelength is perfect.
        gap_size = lezargus.library.interpolate.get_smallest_gap(
            wavelength=clean_wavelength,
        )
        interp_data = lezargus.library.interpolate.Spline1DInterpolate(
            x=clean_wavelength,
            v=clean_data,
            extrapolate=extrapolate,
            extrapolate_fill=np.nan,
            gap=gap_size,
        )(wavelength)
        interp_uncertainty = lezargus.library.interpolate.Spline1DInterpolate(
            x=clean_wavelength,
            v=clean_uncertainty,
            extrapolate=extrapolate,
            extrapolate_fill=np.nan,
            gap=gap_size,
        )(wavelength)

        # Checking if we need to compute the interpolation of a mask.
        if skip_mask:
            interp_mask = None
        else:
            interp_mask = np.full_like(interp_data, False)
            logging.error(
                error_type=logging.ToDoError,
                message=(
                    "The interpolation of a mask for a spectrum is not yet"
                    " implemented."
                ),
            )
        # Checking if we need to compute the interpolation of flag array.
        if skip_flags:
            interp_flags = None
        else:
            interp_flags = np.full_like(interp_data, 1)
            logging.error(
                error_type=logging.ToDoError,
                message=(
                    "The interpolation of flags for a spectrum is not yet"
                    " implemented."
                ),
            )

        # All done.
        return interp_data, interp_uncertainty, interp_mask, interp_flags

    def interpolate_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
        extrapolate: bool = False,
        skip_mask: bool = True,
        skip_flags: bool = True,
    ) -> hint.LezargusSpectrum:
        """Interpolation calling function for spectrum.

        Each entry is considered a single point to interpolate over.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength values which we are going to interpolate to. The
            units of the data of this array should be the same as the
            wavelength unit stored.
        extrapolate : bool, default = False
            If True, we extrapolate. Otherwise, the edges are NaNs.
        skip_mask : bool, default = True
            If provided, the propagation of data mask through the
            interpolation is skipped. It is computationally a little expensive
            otherwise.
        skip_flags : bool, default = True
            If provided, the propagation of data flags through the
            interpolation is skipped. It is computationally a little expensive
            otherwise.

        Returns
        -------
        interp_spectrum : LezargusSpectrum
            The interpolated spectrum, packaged as a spectrum class.

        """
        # We rely on the primary interpolation routine.
        interp_data, interp_uncertainty, interp_mask, interp_flags = (
            self.interpolate(
                wavelength=wavelength,
                extrapolate=extrapolate,
                skip_mask=skip_mask,
                skip_flags=skip_flags,
            )
        )
        # Repackaging it.
        interp_spectrum = type(self)(
            wavelength=wavelength,
            data=interp_data,
            uncertainty=interp_uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            spectral_scale=self.spectral_scale,
            pixel_scale=self.pixel_scale,
            slice_scale=self.slice_scale,
            mask=interp_mask,
            flags=interp_flags,
            header=self.header,
        )
        return interp_spectrum

    def stitch_on(
        self: hint.Self,
        *spectra: hint.LezargusSpectrum,
        **kwargs: object,
    ) -> hint.Self:
        """Stitch this spectrum with other input spectra.

        We stitch spectra onto this spectra. This function is basically
        a compatibility wrapper around the class method :py:meth:`stitch`.

        Parameters
        ----------
        *spectra : LezargusSpectrum
            A set of Lezargus spectra which we will stitch to this one.
        **kwargs : object
            Arguments passed to :py:meth:`stitch`.

        Returns
        -------
        stitch_spectrum : LezargusSpectrum
            The spectrum after stitching.

        """
        # We just add ourselves to the rest of the spectra to stitch together.
        self_copy = copy.deepcopy(self)
        all_spectra = [self_copy, *spectra]
        return self.stitch(*all_spectra, **kwargs)

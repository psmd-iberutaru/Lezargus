"""Spectra data container.

This module and class primarily deals with spectral data.
"""

import copy
import numpy as np
import scipy.interpolate

from lezargus import library
from lezargus.container import LezargusContainerArithmetic
from lezargus.library import hint
from lezargus.library import logging


class LezargusSpectra(LezargusContainerArithmetic):
    """Container to hold spectral data and perform operations on it.

    Attributes
    ----------
    wavelength : ndarray
        The wavelength of the spectra. The unit of wavelength is typically
        in microns; but, check the `wavelength_unit` value.
    data : ndarray
        The flux of the spectra. The unit of the flux is typically
        in flam; but, check the `flux_unit` value.
    uncertainty : ndarray
        The uncertainty in the flux of the spectra. The unit of the uncertainty
        is the same as the flux value; per `uncertainty_unit`.

    wavelength_unit : Astropy Unit
        The unit of the wavelength array.
    flux_unit : Astropy Unit
        The unit of the flux array.
    uncertainty_unit : Astropy Unit
        The unit of the uncertainty array. This unit is the same as the flux
        unit.

    mask : ndarray
        A mask of the flux data, used to remove problematic areas. Where True,
        the values of the flux is considered mask.
    flags : ndarray
        Flags of the flux data. These flags store metadata about the flux.

    header : Header
        The header information, or metadata in general, about the data.
    """

    def __init__(
        self: "LezargusSpectra",
        wavelength: hint.ndarray,
        data: hint.ndarray,
        uncertainty: hint.ndarray | None = None,
        wavelength_unit: str | hint.Unit | None = None,
        data_unit: str | hint.Unit | None = None,
        mask: hint.ndarray | None = None,
        flags: hint.ndarray | None = None,
        header: hint.Header | None = None,
    ) -> None:
        """Instantiate the spectra class.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength of the spectra.
        data : ndarray
            The flux of the spectra.
        uncertainty : ndarray, default = None
            The uncertainty of the spectra. By default, it is None and the
            uncertainty value is 0.
        wavelength_unit : Astropy-Unit like, default = None
            The wavelength unit of the spectra. It must be interpretable by
            the Astropy Units package. If None, the the unit is dimensionless.
        data_unit : Astropy-Unit like, default = None
            The data unit of the spectra. It must be interpretable by
            the Astropy Units package. If None, the the unit is dimensionless.
        mask : ndarray, default = None
            A mask which should be applied to the spectra, if needed.
        flags : ndarray, default = None
            A set of flags which describe specific points of data in the
            spectra.
        header : Header, default = None
            A set of header data describing the data. Note that when saving,
            this header is written to disk with minimal processing. We highly
            suggest writing of the metadata to conform to the FITS Header
            specification as much as possible.
        """
        # The data must be one dimensional.
        container_dimensions = 1
        if len(data.shape) != container_dimensions:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The input data for a LezargusSpectra instantiation has a"
                    " shape {sh}, which is not the expected one dimension."
                    .format(sh=data.shape)
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
                    "Wavelength array shape: {wv_s}; data array shape: {dt_s}."
                    " The arrays need to be the same shape or cast-able to"
                    " such.".format(wv_s=wavelength.shape, dt_s=data.shape)
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
            mask=mask,
            flags=flags,
            header=header,
        )

    @classmethod
    def read_fits_file(
        cls: hint.Type["LezargusSpectra"],
        filename: str,
    ) -> hint.Self:
        """Read a Lezargus spectra FITS file.

        We load a Lezargus FITS file from disk. Note that this should only
        be used for 1-D spectra files.

        Parameters
        ----------
        filename : str
            The filename to load.

        Returns
        -------
        spectra : Self-like
            The LezargusSpectra class instance.
        """
        # Any pre-processing is done here.
        # Loading the file.
        spectra = cls._read_fits_file(filename=filename)
        # Any post-processing is done here.
        # All done.
        return spectra

    def write_fits_file(
        self: hint.Self,
        filename: str,
        overwrite: bool = False,
    ) -> hint.Self:
        """Write a Lezargus spectra FITS file.

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

    def interpolate(
        self: hint.Self,
        wavelength: hint.ndarray,
        skip_mask: bool = True,
        skip_flags: bool = True,
    ) -> tuple[
        hint.ndarray,
        hint.ndarray,
        hint.ndarray | None,
        hint.ndarray | None,
    ]:
        """Interpolation calling function for spectra.

        Each entry is considered a single point to interpolate over.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength values which we are going to interpolate to. The
            units of the data of this array should be the same as the
            wavelength unit stored.
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
        clean_index = (
            np.isfinite(self.wavelength)
            & np.isfinite(self.data)
            & np.isfinite(self.uncertainty)
        )
        clean_wavelength = self.wavelength[clean_index]
        clean_data = self.data[clean_index]
        clean_uncertainty = self.uncertainty[clean_index]

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
        overlap = library.wrapper.wavelength_overlap_fraction(
            base=clean_wavelength,
            contain=wavelength,
        )
        if overlap < 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Interpolation is attempted at a wavelength beyond the"
                    " domain of wavelengths of this spectrum. The overlap"
                    " fraction is {of}.".format(of=overlap)
                ),
            )

        # The interpolated data for both the data itself and uncertainty.
        interp_data = scipy.interpolate.CubicSpline(
            clean_wavelength,
            clean_data,
            extrapolate=True,
        )(wavelength)
        interp_uncertainty = scipy.interpolate.CubicSpline(
            clean_wavelength,
            clean_uncertainty,
            extrapolate=True,
        )(wavelength)

        # Checking if we need to compute the interpolation of a mask.
        if skip_mask:
            interp_mask = None
        else:
            interp_mask = None
            logging.error(
                error_type=logging.NotSupportedError,
                message=(
                    "The interpolation of a mask for spectra is not yet"
                    " implemented."
                ),
            )
        # Checking if we need to compute the interpolation of flag array.
        if skip_flags:
            interp_flags = None
        else:
            interp_flags = None
            logging.error(
                error_type=logging.NotSupportedError,
                message=(
                    "The interpolation of flags for spectra is not yet"
                    " implemented."
                ),
            )

        # All done.
        return interp_data, interp_uncertainty, interp_mask, interp_flags

    def stitch(
        self:"LezargusSpectra",
        *spectra: "LezargusSpectra",
        weight: list[hint.ndarray] | str = "uniform",
        average_function: hint.Callable[
            [hint.ndarray, hint.ndarray, hint.ndarray],
            tuple[float, float],
        ] = None,
    ) -> hint.Self:
        """Stitch together different spectra; we do not scaling.

        We stitch this spectra with input spectra. If the spectra are not
        already to the same scale however, this will result in wildly incorrect
        results. The header information is preserved, though we take what we
        can from the other objects.

        Parameters
        ----------
        *spectra : LezargusSpectra
            A set of Lezargus spectra which we will stitch to this one.
        weight : list[ndarray] or str, default = None
            A list of the weights in the data for stitching. Each entry in
            the list must have a corresponding entry in the wavelength and
            data list, or None. For convenience, we provide short-cut inputs
            for the following:

                - `uniform` : Uniform weights.
                - `invar` : Inverse variance weights.
        average_function : Callable, str, default = None
            The function used to average all of the spectra together.
            It must also be able to accept weights and propagate uncertainties.
            If None, we default to the weighted mean. Namely, it must be of the
            form f(val, uncert, weight) = avg, uncert.

        Returns
        -------
        stitch_spectra : LezargusSpectra
            The spectra after stitching.
        """
        # If there are no spectra to stitch, then we do nothing.
        if len(spectra) == 0:
            # We still warn just in case.
            logging.warning(warning_type=logging.InputWarning, message="No additional spectra objects were submitted to stitch, no stitching applied.")
            return self

        # We need to make sure these are all Lezargus spectra.
        lz_spectra = []
        for spectradex in spectra:
            if not isinstance(spectradex, LezargusSpectra):
                logging.critical(
                    critical_type=logging.InputError,
                    message=(
                        "Input type {spx} is not a LezargusSpectra, we cannot"
                        " use it to stitch.".format(spx=type(spectradex))
                    ),
                )
            lz_spectra.append(spectradex)
        # We finally append ourselves. Working on a copy is probably for the
        # best.
        lz_spectra.append(copy.deepcopy(self))

        # We need to translate the weight input.
        if isinstance(weight, str):
            weight = weight.casefold()
            if weight == "uniform":
                # We compute uniform weights.
                using_weights = [
                    np.ones_like(spectradex.wavelength)
                    for spectradex in spectra
                ]
            elif weight == "invar":
                # We compute weights which are the inverse of the variance
                # in the data.
                using_weights = [
                    1 / spectradex.uncertainty**2 for spectradex in spectra
                ]
            else:
                # A valid shortcut string has not been provided.
                accepted_options = ["uniform", "invar"]
                logging.critical(
                    critical_type=logging.InputError,
                    message=(
                        "The weight shortcut option {opt} is not valid; it must"
                        " be one of: {acc}".format(
                            opt=weight, acc=accepted_options
                        )
                    ),
                )
        else:
            using_weights = weight

        # Next, we stitch together the data for the spectra.
        stitch_wavelength, stitch_data, stitch_uncertainty = (
            library.stitch.stitch_spectra_arrays(
                wavelength=[spectradex.wavelength for spectradex in spectra],
                data=[spectradex.data for spectradex in spectra],
                uncertainty=[spectradex.uncertainty for spectradex in spectra],
                weight=using_weights,
                average_function=average_function,
            )
        )
        # We also stitch together the flags and the mask. They are handled
        # with a different function. TODO
        logging.error(
            error_type=logging.NotSupportedError, message=
                "Flag and mask stitching not yet supported."
            
        )
        stitch_mask = None
        stitch_flags = None

        # We merge the header. TODO
        logging.error(
            error_type=logging.NotSupportedError, message=
                "Header stitching not yet supported."
            
        )
        stitch_header = None

        # We compile the new Spectra. We do not expect a subclass but we
        # try and allow it.
        stitch_spectra = self.__class__(
            wavelength=stitch_wavelength,
            data=stitch_data,
            uncertainty=stitch_uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.wavelength_unit,
            mask=stitch_mask,
            flags=stitch_flags,
            header=stitch_header,
        )
        # All done.
        return stitch_spectra

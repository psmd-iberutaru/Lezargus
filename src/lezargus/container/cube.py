"""Spectral data cube container.

This module and class primarily deals with spectral data cubes containing both
spatial and spectral information.
"""

import numpy as np

import lezargus
from lezargus.container import LezargusContainerArithmetic
from lezargus.library import hint
from lezargus.library import logging


class LezargusCube(LezargusContainerArithmetic):
    """Container to hold spectral cube data and perform operations on it.

    Attributes
    ----------
    For all available attributes, see :py:class:`LezargusContainerArithmetic`.

    """

    def __init__(
        self: hint.Self,
        wavelength: hint.ndarray,
        data: hint.ndarray,
        uncertainty: hint.ndarray = None,
        wavelength_unit: str | hint.Unit | None = None,
        data_unit: str | hint.Unit | None = None,
        pixel_scale: float | None = None,
        slice_scale: float | None = None,
        mask: hint.ndarray | None = None,
        flags: hint.ndarray | None = None,
        header: hint.Header | None = None,
    ) -> None:
        """Instantiate the spectral cube class.

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
            The uncertainty in the data of the spectra. The unit of the
            uncertainty is the same as the data value; per
            :py:attr:`uncertainty_unit`.
        wavelength_unit : Astropy Unit
            The unit of the wavelength array. If None, we assume unit-less.
        data_unit : Astropy Unit
            The unit of the data array. If None, we assume unit-less.
        pixel_scale : float, default = None
            The E-W, "x" dimension, pixel plate scale of the spatial component,
            if any. Must be in radians per pixel. Scale is None if none
            is provided.
        slice_scale : float, default = None
            The N-S, "y" dimension, pixel slice scale of the spatial component,
            if any. Must be in radians per slice-pixel. Scale is None if none
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

        """
        # The data must be three dimensional.
        container_dimensions = 3
        if len(data.shape) != container_dimensions:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The input data for a LezargusCube instantiation has a"
                    f" shape {data.shape}, which is not the expected three"
                    " dimensions."
                ),
            )
        # The wavelength and the flux data must be parallel, and thus the same
        # shape.
        wavelength = np.array(wavelength, dtype=float)
        data = np.array(data, dtype=float)
        if wavelength.shape[0] != data.shape[2]:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"Wavelength array shape: {wavelength.shape}; flux cube"
                    f" array shape: {data.shape}. The cube wavelength dimension"
                    f" length {data.shape[2]} is not compatible with the"
                    f" wavelength length {wavelength.shape[2]}."
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
            pixel_scale=pixel_scale,
            slice_scale=slice_scale,
            mask=mask,
            flags=flags,
            header=header,
        )

    @classmethod
    def read_fits_file(
        cls: hint.Type[hint.Self],
        filename: str,
    ) -> hint.Self:
        """Read a Lezargus cube FITS file.

        We load a Lezargus FITS file from disk. Note that this should only
        be used for 3-D cube files.

        Parameters
        ----------
        filename : str
            The filename to load.

        Returns
        -------
        cube : Self-like
            The LezargusCube class instance.

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
        """Write a Lezargus cube FITS file.

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

    def convolve_spectra(self: hint.Self, kernel: hint.ndarray) -> hint.Self:
        """Convolve the cube by a spectral kernel convolving spectra slices.

        Convolving a spectral cube can either be done one of two ways;
        convolving by image slices or convolving by spectral slices. We here
        convolve by spectral slices.

        Parameters
        ----------
        kernel : ndarray
            The spectral kernel which we are using to convolve.

        Returns
        -------
        convolved_cube : ndarray
            A near copy of the data cube after convolution.

        """
        # Using this kernel, we convolve the cube. We assume that the
        # uncertainties add in quadrature.
        convolved_data = (
            lezargus.library.convolution.convolve_3d_array_by_1d_kernel(
                array=self.data,
                kernel=kernel,
            )
        )
        convolved_uncertainty = np.sqrt(
            lezargus.library.convolution.convolve_3d_array_by_1d_kernel(
                self.uncertainty**2,
                kernel=kernel,
            ),
        )

        # We also propagate the convolution of the mask and the flags where
        # needed.
        logging.error(
            error_type=logging.ToDoError,
            message=(
                "Propagation of mask and flags via convolution is not done."
            ),
        )
        convolved_mask = self.mask
        convolved_flags = self.flags

        # From the above information, we construct the new spectra.
        cube_class = type(self)
        convolved_cube = cube_class(
            wavelength=self.wavelength,
            data=convolved_data,
            uncertainty=convolved_uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            mask=convolved_mask,
            flags=convolved_flags,
            header=self.header,
        )

        # All done.
        return convolved_cube

    def convolve_image(self: hint.Self, kernel: hint.ndarray) -> hint.Self:
        """Convolve the cube by an image kernel convolving image slices.

        Convolving a spectral cube can either be done one of two ways;
        convolving by image slices or convolving by spectral slices. We here
        convolve by image slices.

        Parameters
        ----------
        kernel : ndarray
            The image kernel which we are using to convolve.

        Returns
        -------
        convolved_cube : ndarray
            A near copy of the data cube after convolution.

        """
        # Using this kernel, we convolve the cube. We assume that the
        # uncertainties add in quadrature.
        convolved_data = (
            lezargus.library.convolution.convolve_3d_array_by_2d_kernel(
                array=self.data,
                kernel=kernel,
            )
        )
        convolved_uncertainty = np.sqrt(
            lezargus.library.convolution.convolve_3d_array_by_2d_kernel(
                self.uncertainty**2,
                kernel=kernel,
            ),
        )

        # We also propagate the convolution of the mask and the flags where
        # needed.
        logging.error(
            error_type=logging.ToDoError,
            message=(
                "Propagation of mask and flags via convolution is not done."
            ),
        )
        convolved_mask = self.mask
        convolved_flags = self.flags

        # From the above information, we construct the new spectra.
        cube_class = type(self)
        convolved_cube = cube_class(
            wavelength=self.wavelength,
            data=convolved_data,
            uncertainty=convolved_uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            mask=convolved_mask,
            flags=convolved_flags,
            header=self.header,
        )

        # All done.
        return convolved_cube

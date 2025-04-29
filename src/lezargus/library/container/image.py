"""Image data container.

This module and class primarily deals with images containing spatial
information.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import numpy as np

from lezargus.library import logging
from lezargus.library.container import LezargusContainerArithmetic


class LezargusImage(LezargusContainerArithmetic):
    """Container to hold image and perform operations on it.

    Attributes
    ----------
    For all available attributes, see :py:class:`LezargusContainerArithmetic`.

    """

    def __init__(
        self: LezargusImage,
        data: hint.NDArray,
        uncertainty: hint.NDArray | None = None,
        wavelength: float | None = None,
        wavelength_unit: str | hint.Unit | None = None,
        data_unit: str | hint.Unit | None = None,
        spectral_scale: float | None = None,
        pixel_scale: float | None = None,
        slice_scale: float | None = None,
        mask: hint.NDArray | None = None,
        flags: hint.NDArray | None = None,
        header: hint.Header | None = None,
    ) -> None:
        """Instantiate the spectra class.

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
        # The data must be two dimensional.
        container_dimensions = 2
        if len(data.shape) != container_dimensions:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The input data for a LezargusImage instantiation has a"
                    f" shape {data.shape}, which is not the expected two"
                    " dimension."
                ),
            )

        # The wavelength parameter is more metadata describing the image. It is
        # completely optional. If provided, we add it. Otherwise we use a
        # None-like value which is can be more properly handled by the other
        # parts of Lezargus.
        if wavelength is not None:
            set_wavelength = np.array([float(wavelength)])
        else:
            set_wavelength = np.array([np.nan])

        # Constructing the original class. We do not deal with WCS here because
        # the base class does not support it. We do not involve units here as
        # well for speed concerns. Both are handled during reading and writing.
        super().__init__(
            wavelength=set_wavelength,
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
        """Read a Lezargus image FITS file.

        We load a Lezargus FITS file from disk. Note that this should only
        be used for 2-D image files.

        Parameters
        ----------
        filename : str
            The filename to load.

        Returns
        -------
        cube : Self-like
            The LezargusImage class instance.

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
    ) -> None:
        """Write a Lezargus image FITS file.

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

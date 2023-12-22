"""Image data container.

This module and class primarily deals with images containing spatial
information.
"""

import numpy as np

from lezargus.container import LezargusContainerArithmetic
from lezargus.library import hint
from lezargus.library import logging


class LezargusImage(LezargusContainerArithmetic):
    """Container to hold image and perform operations on it.

    Attributes
    ----------
    wavelength : float
        The wavelength of the image. The unit of wavelength is typically
        in microns; but, check the `wavelength_unit` value. If none has
        been provided, this value is an array of None.
    data : ndarray
        The flux of the spectra cube. The unit of the flux is typically
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
        self: "LezargusImage",
        data: hint.ndarray,
        uncertainty: hint.ndarray | None = None,
        wavelength: float | None = None,
        wavelength_unit: str | hint.Unit = None,
        data_unit: str | hint.Unit | None = None,
        mask: hint.ndarray | None = None,
        flags: hint.ndarray | None = None,
        header: hint.Header | None = None,
    ) -> None:
        """Instantiate the spectra class.

        Parameters
        ----------
        data : ndarray
            The flux of the spectra.
        uncertainty : ndarray, default = None
            The uncertainty of the spectra. By default, it is None and the
            uncertainty value is 0.
        wavelength : ndarray, default = None
            The wavelength of the image. If this is not provided, it defaults
            to 0, otherwise, it is an array of a single value.
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
        # The data must be two dimensional.
        container_dimensions = 2
        if len(data.shape) != container_dimensions:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The input data for a LezargusImage instantiation has a"
                    " shape {data.shape}, which is not the expected two"
                    " dimension."
                ),
            )

        # The wavelength parameter is more metadata describing the image. It is
        # completely optional. If provided, we add it.
        if wavelength is not None:
            self.wavelength = np.array(float(wavelength))
        else:
            self.wavelength = np.array(None)

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
        cls: hint.Type["LezargusImage"],
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
    ) -> hint.Self:
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

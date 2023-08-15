"""Spectra data container.

This module and class primarily deals with spectral data.
"""


import numpy as np

from lezargus.container import LezargusContainerArithmetic
from lezargus.library import hint
from lezargus.library import logging


class LezargusSpectra(LezargusContainerArithmetic):
    """Container to hold spectral data and perform operations on it.

    Attributes
    ----------
    wavelength : Array
        The wavelength of the spectra. The unit of wavelength is typically
        in microns; but, check the `wavelength_unit` value.
    data : Array
        The flux of the spectra. The unit of the flux is typically
        in flam; but, check the `flux_unit` value.
    uncertainty : Array
        The uncertainty in the flux of the spectra. The unit of the uncertainty
        is the same as the flux value; per `uncertainty_unit`.

    wavelength_unit : Astropy Unit
        The unit of the wavelength array.
    flux_unit : Astropy Unit
        The unit of the flux array.
    uncertainty_unit : Astropy Unit
        The unit of the uncertainty array. This unit is the same as the flux
        unit.

    mask : Array
        A mask of the flux data, used to remove problematic areas. Where True,
        the values of the flux is considered mask.
    flags : Array
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
        wavelength : Array
            The wavelength of the spectra.
        data : Array
            The flux of the spectra.
        uncertainty : Array, default = None
            The uncertainty of the spectra. By default, it is None and the
            uncertainty value is 0.
        wavelength_unit : Astropy-Unit like, default = None
            The wavelength unit of the spectra. It must be interpretable by
            the Astropy Units package. If None, the the unit is dimensionless.
        data_unit : Astropy-Unit like, default = None
            The data unit of the spectra. It must be interpretable by
            the Astropy Units package. If None, the the unit is dimensionless.
        mask : Array, default = None
            A mask which should be applied to the spectra, if needed.
        flags : Array, default = None
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

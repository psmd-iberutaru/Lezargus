"""Parent class for the containers to implement arithmetic and other functions.

The Astropy NDArrayData arithmetic class is not wavelength aware. This class
overwrites and wraps around the NDArithmeticMixin class and allows it to be
wavelength aware. We also avoid the need to do a lot of the recreating of the
data object.
"""
import copy

import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


class LezargusContainerArithmetic:
    """Lezargus wavelength-aware arithmetic.

    This is the class which allows for the arithmetic behind the scenes to
    work with wavelength knowledge. All we do is overwrite the NDDataArray
    arithmetic functions to perform wavelength checks and pass it through
    without wavelength issues.

    Attributes
    ----------
    wavelength : Array
        The wavelength of the spectra. The unit of wavelength is typically
        in microns; but, check the `wavelength_unit` value.
    data : Array
        The data or flux of the spectra cube. The unit of the flux is typically
        in flam; but, check the `data_unit` value.
    uncertainty : Array
        The uncertainty in the data. The unit of the uncertainty
        is the same as the flux value; per `uncertainty_unit`.
    wavelength_unit : Astropy Unit
        The unit of the wavelength array.
    data_unit : Astropy Unit
        The unit of the data array.
    uncertainty_unit : Astropy Unit
        The unit of the uncertainty array. This unit is the same as the data
        unit.
    mask : Array
        A mask of the data, used to remove problematic areas. Where True,
        the values of the data is considered masked.
    flags : Array
        Flags of the data. These flags store metadata about the data.
    header : Header
        The header information, or metadata in general, about the data.
    """

    def __init__(
        self: "LezargusContainerArithmetic",
        wavelength: hint.ndarray,
        data: hint.ndarray,
        uncertainty: hint.ndarray,
        wavelength_unit: hint.Unit,
        data_unit: hint.Unit,
        mask: hint.ndarray,
        flags: hint.ndarray,
        header: dict,
    ) -> None:
        """Construct a wavelength-aware NDDataArray for arithmetic.

        Parameters
        ----------
        wavelength : Array
            The wavelength of the spectra. The unit of wavelength is typically
            in microns; but, check the `wavelength_unit` value.
        data : Array
            The data of the spectra cube. The unit of the flux is typically
            in flam; but, check the `data_unit` value.
        uncertainty : Array
            The uncertainty in the data of the spectra. The unit of the
            uncertainty is the same as the data value; per `uncertainty_unit`.
        wavelength_unit : Astropy Unit
            The unit of the wavelength array.
        data_unit : Astropy Unit
            The unit of the data array.
        mask : Array
            A mask of the data, used to remove problematic areas. Where True,
            the values of the data is considered masked.
        flags : Array
            Flags of the data. These flags store metadata about the data.
        header : Header, default = None
            A set of header data describing the data. Note that when saving,
            this header is written to disk with minimal processing. We highly
            suggest writing of the metadata to conform to the FITS Header
            specification as much as possible.

        Returns
        -------
        None
        """
        # The data is taken by reference, we don't want any side effects
        # so we just copy it.
        data = np.array(data, copy=True)

        # If the uncertainty is broadcast-able, we do so and properly format it
        # so it can be used later.
        uncertainty = 0 if uncertainty is None else uncertainty
        uncertainty = np.array(uncertainty, copy=True)
        if uncertainty.size == 1:
            # The uncertainty seems to be single value, we fill it to fit the
            # entire array.
            uncertainty = np.full_like(data, uncertainty)

        # If there is no mask, we just provide a blank one for convenience.
        # Otherwise we need to format the mask so it can be used properly by
        # the subclass.
        if mask is None:
            mask = np.full_like(data, False, dtype=bool)
        else:
            mask = np.array(mask, dtype=bool)
        if mask.size == 1:
            mask = np.full_like(data, bool(mask))
        # Similarly for the flags.
        if flags is None:
            flags = np.full_like(data, 1, dtype=np.uint)
        else:
            flags = np.array(flags, dtype=np.uint)
        if flags.size == 1:
            flags = np.full_like(flags, np.uint)

        # The uncertainty must be the same size and shape of the data, else it
        # does not make any sense. The mask as well.
        if data.shape != uncertainty.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "Data array shape: {dt_s}; uncertainty array shape: {un_s}."
                    " The arrays need to be the same shape or broadcast-able to"
                    " such.".format(dt_s=data.shape, un_s=uncertainty.shape)
                ),
            )
        # Moreover, the mask and flags must be the same shape as well.
        if data.shape != mask.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "Data array shape: {dt_s}; mask array shape: {mk_s}. The"
                    " arrays need to be the same shape or broadcast-able to"
                    " such.".format(dt_s=data.shape, mk_s=mask.shape)
                ),
            )
        if data.shape != flags.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "Data array shape: {dt_s}; flag array shape: {fg_s}. The"
                    " arrays need to be the same shape or broadcast-able to"
                    " such.".format(dt_s=data.shape, fg_s=flags.shape)
                ),
            )

        # Constructing the original class. We do not deal with WCS here because
        # the base class does not support it. We do not involve units here as
        # well for speed concerns. Both are handled during reading and writing.

        # Add the mainstays of the data.
        self.wavelength = np.asarray(wavelength)
        self.data = np.asarray(data)
        self.uncertainty = np.asarray(uncertainty)
        # Parsing the units.
        self.wavelength_unit = library.conversion.parse_unit_to_astropy_unit(
            unit_string=wavelength_unit,
        )
        self.data_unit = library.conversion.parse_unit_to_astropy_unit(
            unit_string=data_unit,
        )
        self.uncertainty_unit = self.data_unit
        # Metadata.
        self.mask = np.asarray(mask)
        self.flags = np.asarray(flags)
        self.header = header
        # All done.

    def __add__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform an addition operation.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.
        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__justify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # If the operand is a single value, then we need to take that into
        # account.
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)

        # Now we perform the addition.
        result.data = self.data + operand_data
        # Propagating the uncertainty.
        covariance = np.cov(self.data.flatten(), operand_data.flatten())[0, 1]
        result.uncertainty = np.sqrt(
            self.uncertainty**2 + operand_uncertainty**2 + 2 * covariance,
        )
        # All done.
        return result

    def __sub__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform a subtraction operation.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.
        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__justify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # If the operand is a single value, then we need to take that into
        # account.
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)

        # Now we perform the addition.
        result.data = self.data - operand_data
        # Propagating the uncertainty.
        covariance = np.cov(self.data.flatten(), operand_data.flatten())[0, 1]
        result.uncertainty = np.sqrt(
            self.uncertainty**2 + operand_uncertainty**2 - 2 * covariance,
        )
        # All done.
        return result

    def __mul__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform a multiplication operation.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.
        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__justify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # If the operand is a single value, then we need to take that into
        # account.
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)

        # Now we perform the addition.
        result.data = self.data * operand_data
        # Propagating the uncertainty.
        covariance = np.cov(self.data.flatten(), operand_data.flatten())[0, 1]
        result.uncertainty = np.abs(result.data) * np.sqrt(
            (self.uncertainty / self.data) ** 2
            + (operand_uncertainty / operand_data) ** 2
            + ((2 * covariance) / (result.data)),
        )
        # All done.
        return result

    def __truediv__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform a true division operation.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.
        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__justify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # If the operand is a single value, then we need to take that into
        # account.
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)

        # Now we perform the addition.
        result.data = self.data / operand_data
        # Propagating the uncertainty.
        covariance = np.cov(self.data.flatten(), operand_data.flatten())[0, 1]
        result.uncertainty = np.abs(result.data) * np.sqrt(
            (self.uncertainty / self.data) ** 2
            + (operand_uncertainty / operand_data) ** 2
            - ((2 * covariance) / (result.data)),
        )
        # All done.
        return result

    def __pow__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform a true division operation.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.
        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__justify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # If the operand is a single value, then we need to take that into
        # account.
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)

        # Now we perform the addition.
        result.data = self.data**operand_data
        # Propagating the uncertainty.
        covariance = np.cov(self.data.flatten(), operand_data.flatten())[0, 1]
        result.uncertainty = np.abs(result.data) * np.sqrt(
            (self.uncertainty * operand_data / self.data) ** 2
            + (np.log(self.data) * operand_uncertainty) ** 2
            + (
                (2 * operand_data * np.log(self.data) * covariance)
                / (self.data)
            ),
        )
        # All done.
        return result

    def __justify_arithmetic_operation(
        self: hint.Self,
        operand: hint.Self | float | int,
    ) -> bool:
        """Justify operations between two objects is valid.

        Operations done between different instances of the Lezargus data
        structure need to keep in mind the wavelength dependance of the data.
        We implement simple checks here to formalize if an operation between
        this object, and some other operand, can be performed.

        Parameters
        ----------
        operand : Self-like or number
            The container object that we have an operation to apply with.

        Returns
        -------
        justification : bool
            The state of the justification test. If it is True, then the
            operation can continue, otherwise, False.

        .. note::
            This function will also raise exceptions upon discovery of
            incompatible objects. Therefore, the False return case is not
            really that impactful.
        """
        # We assume that the two objects are incompatible, until proven
        # otherwise.
        justification = False

        # We first check for appropriate types. Only singular values, and
        # equivalent Lezargus containers can be accessed.
        if isinstance(operand, int | float):
            # The operand is likely a singular value, so it can be properly
            # broadcast together.
            operand_data = np.array(operand)
        # If the Lezargus data types are the same.
        elif self.__class__ == operand.__class__:
            # All good.
            operand_data = operand.data
        else:
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    "Arithmetics with Lezargus type {ltp} and operand type"
                    " {otp} is not compatible.".format(
                        ltp=type(self),
                        otp=type(operand),
                    )
                ),
            )

        # Next we check if the data types are broadcast-able in the first place.
        try:
            broadcast_shape = np.broadcast_shapes(
                self.data.shape,
                operand_data.shape,
            )
        except ValueError:
            # The data is unable to be broadcast together.
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    "The Lezargus container data shape {lds} is not"
                    " broadcast-able to the operand data shape {ods}.".format(
                        lds=self.data.shape,
                        ods=operand_data.shape,
                    )
                ),
            )
        else:
            # The shapes are broadcast-able, but the container data shape must
            # be preserved and it itself cannot be broadcast.
            if self.data.shape != broadcast_shape:
                logging.critical(
                    critical_type=logging.ArithmeticalError,
                    message=(
                        "The Lezargus container shape {lds} cannot be changed."
                        " A broadcast with the operand data shape {ods} would"
                        " force the container shape to {nlds}.".format(
                            lds=self.data.shape,
                            ods=operand_data.shape,
                            nlds=broadcast_shape,
                        )
                    ),
                )
            else:
                # All good.
                pass

        # Now we need to check if the wavelengths are compatible. Attempting to
        # do math of two Lezargus containers without aligned wavelength values
        # is just not proper.
        if self.wavelength.shape != operand.wavelength.shape:
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    "The wavelength array shape of the Lezargus container {lw}"
                    " and the operand container {ow} is not the same."
                    " Arithmetic cannot be performed.".format(
                        lw=self.wavelength.shape,
                        ow=operand.wavelength.shape,
                    )
                ),
            )
        if not np.allclose(self.wavelength, operand.wavelength):
            # This is a serious error which can lead to bad results. However,
            # it only affects accuracy and not the overall computation of the
            # program.
            logging.error(
                error_type=logging.AccuracyError,
                message=(
                    "The wavelength arrays between two Lezargus containers are"
                    " not matching; operation had interpolation performed to"
                    " account for this."
                ),
            )

        # If the wavelength or data units are all off, it will lead to
        # incorrect results.
        if self.wavelength_unit != operand.wavelength_unit:
            logging.error(
                error_type=logging.AccuracyError,
                message=(
                    "The Lezargus container wavelength unit {lwu} is not the"
                    " same as the operand unit {owu}.".format(
                        lwu=self.wavelength_unit,
                        owu=operand.wavelength_unit,
                    )
                ),
            )
        if self.data_unit != operand.data_unit:
            logging.error(
                error_type=logging.AccuracyError,
                message=(
                    "The Lezargus container data/flux unit {lwu} is not the"
                    " same as the operand unit {owu}.".format(
                        lwu=self.wavelength_unit,
                        owu=operand.wavelength_unit,
                    )
                ),
            )

        # If it survived all of the tests above, then it should be fine.
        justification = True
        return justification

    @classmethod
    def _read_fits_file(
        cls: hint.Type["LezargusContainerArithmetic"],
        filename: str,
    ) -> hint.Self:
        """Read in a FITS file into an object.

        This is a wrapper around the main FITS class for uniform handling.
        The respective containers should wrap around this for
        container-specific handling and should not overwrite this function.

        Parameters
        ----------
        filename : str
            The file to read in.

        Returns
        -------
        container : Self-like
            The Lezargus container which was read into the file.
        """
        # Read in the FITS file.
        (
            header,
            wavelength,
            data,
            uncertainty,
            wavelength_unit,
            data_unit,
            mask,
            flags,
        ) = library.fits.read_lezargus_fits_file(filename=filename)
        # Check if the FITS file format is correct for the container.
        if header.get("LZ_FITSF", None) != cls.__name__:
            logging.error(
                error_type=logging.FileError,
                message=(
                    "The following FITS file {fit} is coded to be a {code} type"
                    " of FITS file, but it is being loaded with the {ncls}."
                    .format(
                        fit=filename,
                        code=header.get("LZ_FITSF", None),
                        ncls=cls.__name__,
                    )
                ),
            )

        # Loading the file up.
        container = cls(
            header=header,
            wavelength=wavelength,
            data=data,
            uncertainty=uncertainty,
            wavelength_unit=wavelength_unit,
            data_unit=data_unit,
            mask=mask,
            flags=flags,
        )
        # All done.
        return container

    def _write_fits_file(
        self: hint.Self,
        filename: str,
        overwrite: bool = False,
    ) -> None:
        """Write a FITS object to disk..

        This is a wrapper around the main FITS class for uniform handling.
        The respective containers should wrap around this for
        container-specific handling and should not overwrite this function.

        Parameters
        ----------
        filename : str
            The file to be written out.
        overwrite : bool, default = False
            If True, overwrite any file conflicts.

        Returns
        -------
        None
        """
        # We send the file to the library function write.
        library.fits.write_lezargus_fits_file(
            filename=filename,
            header=self.header,
            wavelength=self.wavelength,
            data=self.data,
            uncertainty=self.uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            uncertainty_unit=self.uncertainty_unit,
            mask=self.mask,
            flags=self.flags,
            overwrite=overwrite,
        )
        # All done.
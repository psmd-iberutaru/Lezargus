"""Parent class for the containers to implement arithmetic and other functions.

The Astropy NDArrayData arithmetic class is not wavelength aware, and so we
implement our own class with similar functionality to it to allow for
wavelength aware operations. Unlike specutils, the interpolation for doing
arithmetic operations of spectral classes of different wavelength solutions
must be explicit. This is to prevent errors from mis-matched spectra being
operated on.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import copy

import astropy.io.fits
import numpy as np

import lezargus
from lezargus.library import logging


class LezargusContainerArithmetic:
    """Lezargus wavelength-aware arithmetic for the major containers.

    This is the class which allows for the arithmetic behind the scenes to
    work with wavelength knowledge.

    Attributes
    ----------
    wavelength : ndarray

    data : ndarray
        The data or flux of the spectra cube. The unit of the flux is typically
        in W m^-2 m^-1; but, check the :py:attr:`data_unit` value.
    uncertainty : ndarray
        The uncertainty in the data. The unit of the uncertainty
        is the same as the flux value; per :py:attr:`uncertainty_unit`.
    wavelength_unit : Astropy Unit
        The unit of the wavelength array.
    data_unit : Astropy Unit
        The unit of the data array.
    spectral_scale : float
        The spectral scale of the image, as a resolution, in wavelength
        separation per pixel. The point spacing on the wavelength axis is not
        always the spectral resolution.
    pixel_scale : float
        The pixel plate scale of the image, in radians per pixel. Typically,
        this is the scale in the E-W or "x" direction. See
        :py:attr:`slice_scale` for the N-S or "y" direction.
    slice_scale : float
        The pixel slice scale of the image, in radians per slice. Typically,
        this is the scale in the N-S or "y" direction. See
        :py:attr:`pixel_scale` for the E-W or "x" direction.
    mask : ndarray
        A mask of the data, used to remove problematic areas. Where True,
        the values of the data is considered masked.
    flags : ndarray
        Flags of the data. These flags store metadata about the data.
    header : Header
        The header information, or metadata in general, about the data.

    """

    _wavelength: hint.NDArray
    """Internal variable for storing the wavelength array. Use
    :py:attr:`wavelength` instead."""

    _data: hint.NDArray
    """Internal variable for storing the data array. Use
    :py:attr:`data` instead."""

    _uncertainty: hint.NDArray
    """Internal variable for storing the uncertainty array. Use
    :py:attr:`uncertainty` instead."""

    def __init__(
        self: LezargusContainerArithmetic,
        wavelength: hint.NDArray | float,
        data: hint.NDArray | float,
        uncertainty: hint.NDArray | float | None,
        wavelength_unit: hint.Unit | str | None = None,
        data_unit: hint.Unit | str | None = None,
        spectral_scale: float | None = None,
        pixel_scale: float | None = None,
        slice_scale: float | None = None,
        mask: hint.NDArray | None = None,
        flags: hint.NDArray | None = None,
        header: hint.Header | dict | None = None,
    ) -> None:
        """Construct a wavelength-aware Lezargus data container.

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
            if any. Must be in radians per pixel. Scale is None if none
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
        # The data is taken by reference, we don't want any side effects
        # so we just copy it.
        data = np.array(data, copy=True, dtype=None)

        # If the uncertainty is broadcast-able, we do so and properly format it
        # so it can be used later.
        uncertainty = 0 if uncertainty is None else uncertainty
        uncertainty = np.array(uncertainty, copy=True)
        if uncertainty.size == 1:
            # The uncertainty seems to be single value, we fill it to fit the
            # entire array.
            uncertainty = np.full_like(data, float(uncertainty))

        # If there is no mask, we just provide a blank one for convenience.
        # Otherwise we need to format the mask so it can be used properly by
        # the subclass.
        if mask is None:
            mask = np.zeros_like(data, dtype=bool)
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
            flags = np.full_like(int(flags), np.uint)

        # The uncertainty must be the same size and shape of the data, else it
        # does not make any sense. The mask as well.
        if data.shape != uncertainty.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"Data array shape: {data.shape}; uncertainty array shape:"
                    f" {uncertainty.shape}. The arrays need to be the same"
                    " shape or broadcast-able to such."
                ),
            )
        # Moreover, the mask and flags must be the same shape as well.
        if data.shape != mask.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"Data array shape: {data.shape}; mask array shape:"
                    f" {mask.shape}. The arrays need to be the same shape or"
                    " broadcast-able to such."
                ),
            )
        if data.shape != flags.shape:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"Data array shape: {data.shape}; flag array shape:"
                    f" {flags.shape}. The arrays need to be the same shape or"
                    " broadcast-able to such."
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
        self.wavelength_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=wavelength_unit,
        )
        self.data_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=data_unit,
        )

        # The spectral, pixel, and slice scale.
        self.spectral_scale = spectral_scale
        self.pixel_scale = pixel_scale
        self.slice_scale = slice_scale

        # Metadata.
        self.mask = (
            np.full_like(self.data, False) if mask is None else np.asarray(mask)
        )
        self.flags = (
            np.full_like(self.data, 1, dtype=int)
            if flags is None
            else np.asarray(flags)
        )
        # We just use a blank header if none has been provided.
        header = {} if header is None else header
        self.header = astropy.io.fits.Header(header)
        # All done.

    @property
    def wavelength(self: hint.Self) -> hint.NDArray:
        """Get the wavelength array.

        The wavelength of the spectra. The unit of wavelength is typically
        in meters; but, check the :py:attr:`wavelength_unit` value.

        Parameters
        ----------
        None

        Returns
        -------
        wavelength : NDArray
            The wavelength array of the container.

        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self: hint.Self, wave: hint.NDArray) -> None:
        """Set the wavelength array.

        Parameters
        ----------
        wave : NDArray
            The input wavelength data.

        Returns
        -------
        None

        """
        # Data type conversion to match configuration file.
        dtype = lezargus.library.conversion.parse_numpy_dtype(
            dtype_string=lezargus.config.META_CONTAINER_FLOAT_DATA_TYPE,
        )
        # Converting.
        self._wavelength = np.asarray(wave, dtype=dtype)

    @wavelength.deleter
    def wavelength(self: hint.Self) -> None:
        """Delete the wavelength array.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Deleting the data.
        self._wavelength = np.full(1, np.nan)
        del self.wavelength

    @property
    def data(self: hint.Self) -> hint.NDArray:
        """Get the data array.

        The data of the spectra. The unit of data is typically
        in meters; but, check the :py:attr:`data_unit` value.

        Parameters
        ----------
        None

        Returns
        -------
        data : NDArray
            The data array of the container.

        """
        return self._data

    @data.setter
    def data(self: hint.Self, data_: hint.NDArray) -> None:
        """Set the data array.

        Parameters
        ----------
        data_ : NDArray
            The input data.

        Returns
        -------
        None

        """
        # Data type conversion to match configuration file.
        dtype = lezargus.library.conversion.parse_numpy_dtype(
            dtype_string=lezargus.config.META_CONTAINER_FLOAT_DATA_TYPE,
        )
        self._data = np.asarray(data_, dtype=dtype)

    @data.deleter
    def data(self: hint.Self) -> None:
        """Delete the data array.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Deleting the data.
        self._data = np.full(1, np.nan)
        del self.data

    @property
    def uncertainty(self: hint.Self) -> hint.NDArray:
        """Get the uncertainty array.

        The uncertainty of the spectra. The unit of uncertainty is typically
        in meters; but, check the :py:attr:`uncertainty_unit` value.

        Parameters
        ----------
        None

        Returns
        -------
        uncertainty : NDArray
            The uncertainty array of the container.

        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self: hint.Self, uncert: hint.NDArray | float) -> None:
        """Set the uncertainty array.

        Parameters
        ----------
        uncert : NDArray
            The input uncertainty data.

        Returns
        -------
        None

        """
        # If the uncertainty is a single value, then we need to broadcast it.
        if isinstance(uncert, int | float | np.number):
            uncert = np.full_like(self.data, uncert)

        # Data type conversion to match configuration file.
        dtype = lezargus.library.conversion.parse_numpy_dtype(
            dtype_string=lezargus.config.META_CONTAINER_FLOAT_DATA_TYPE,
        )
        self._uncertainty = np.asarray(uncert, dtype=dtype)

    @uncertainty.deleter
    def uncertainty(self: hint.Self) -> None:
        """Delete the uncertainty array.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Deleting the data.
        self._uncertainty = np.full(1, np.nan)
        del self.uncertainty

    @property
    def uncertainty_unit(self: hint.Self) -> hint.Unit:
        """Return the uncertainty unit, i.e. the data unit.

        Parameters
        ----------
        None

        Returns
        -------
        uncertainty_unit : Unit
            The uncertainty unit.

        """
        return self.data_unit

    def __verify_arithmetic_operation(
        self: hint.Self,
        operand: hint.Self | float,
    ) -> bool:
        """Verify operations between two objects is valid.

        Operations done between different instances of the Lezargus data
        structure need to keep in mind the wavelength dependence of the data.
        We implement simple checks here to formalize if an operation between
        this object, and some other operand, can be performed.

        Parameters
        ----------
        operand : Self-like or float
            The container object that we have an operation to apply with.

        Returns
        -------
        verified : bool
            The state of the verification test. If it is True, then the
            operation can continue, otherwise, False.

        .. note::
            This function will also raise exceptions upon discovery of
            incompatible objects. Therefore, the False return case is not
            really that impactful.

        """
        # We assume that the two objects are incompatible, until proven
        # otherwise.
        verified = False

        # We first check for appropriate types. Only singular values, and
        # equivalent Lezargus containers can be accessed.
        if isinstance(operand, int | float | np.number):
            # The operand is likely a singular value, so it can be properly
            # broadcast together. It is a singe value, all other checks
            # are unneeded.
            verified = True
            return verified

        # If the Lezargus data types are the same.
        if isinstance(operand, type(self)):
            # All good.
            operand_data = operand.data
        else:
            operand_data = None
            verified = False
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    f"Arithmetics with Lezargus type {type(self)} and operand"
                    f" type {type(operand)} are not compatible."
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
            verified = False
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    f"The Lezargus container data shape {self.data.shape} is"
                    " not broadcast-able to the operand data shape"
                    f" {operand_data.shape}."
                ),
            )
        else:
            # The shapes are broadcast-able, but the container data shape must
            # be preserved and it itself cannot be broadcast.
            if self.data.shape != broadcast_shape:
                verified = False
                logging.critical(
                    critical_type=logging.ArithmeticalError,
                    message=(
                        f"The Lezargus container shape {self.data.shape} cannot"
                        " be changed. A broadcast with the operand data shape"
                        f" {operand_data.shape} would force the container shape"
                        f" to {broadcast_shape}."
                    ),
                )
            else:
                # All good.
                pass

        # Now we need to check if the wavelengths are compatible. Attempting to
        # do math of two Lezargus containers without aligned wavelength values
        # is just not proper.
        if self.wavelength.shape != operand.wavelength.shape:
            verified = False
            logging.critical(
                critical_type=logging.ArithmeticalError,
                message=(
                    "The wavelength array shape of the Lezargus container"
                    f" {self.wavelength.shape} and the operand container"
                    f" {operand.wavelength.shape} is not the same. Arithmetic"
                    " cannot be performed."
                ),
            )
        if not np.allclose(self.wavelength, operand.wavelength):
            # This is a serious problem which can lead to bad results. However,
            # it only affects accuracy and not the overall computation of the
            # program.
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "The wavelength arrays between two Lezargus containers are"
                    " not matching; operation had interpolation performed to"
                    " account for this."
                ),
            )

        # If the wavelength or data units are all off, it will lead to
        # incorrect results.
        if self.wavelength_unit != operand.wavelength_unit:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "The Lezargus container wavelength unit"
                    f" {self.wavelength_unit} is not the same as the operand"
                    f" unit {operand.wavelength_unit}."
                ),
            )

        # If it survived all of the tests above, then it should be fine.
        verified = True
        return verified

    def __add__(self: hint.Self, operand: float | hint.Self) -> hint.Self:
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
        if not self.__verify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # There are two cases, if the operand object is a container, or
        # a singular value.
        if isinstance(operand, LezargusContainerArithmetic):
            # It is a container, we perform operations as normal.
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
            operand_unit = operand.data_unit
        else:
            # The operand is just a single value, so, we handle it as so.
            # We assume a single value does not have any uncertainty that
            # we really care about.
            # We trust the addition of non-unit values respect the unit
            # of the container.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)
            operand_unit = self.data_unit

        # Addition and subtraction are unique in that we need to also check
        # the data units.
        if self.data_unit != operand_unit:
            logging.error(
                error_type=logging.ArithmeticalError,
                message=(
                    "The Lezargus container data/flux unit"
                    f" {self.data_unit} is not the same as the operand"
                    f" unit {operand_unit}."
                ),
            )

        # Now we do the addition.
        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)
        result.data, result.uncertainty = lezargus.library.math.add(
            augend=self.data,
            addend=operand_data,
            augend_uncertainty=self.uncertainty,
            addend_uncertainty=operand_uncertainty,
        )
        # All done.
        return result

    def __radd__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform an addition operation, commutative with __add__.

        Parameters
        ----------
        operand : Self-like
            The container object to add to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        return self.__add__(operand=operand)

    def __sub__(self: hint.Self, operand: float | hint.Self) -> hint.Self:
        """Perform a subtraction operation.

        Parameters
        ----------
        operand : Self-like
            The container object to subtract to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__verify_arithmetic_operation(operand=operand):
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "The arithmetic justification check returned False, but it"
                    " really should have raised an error and should not have"
                    " returned here."
                ),
            )

        # There are two cases, if the operand object is a container, or
        # a singular value.
        if isinstance(operand, LezargusContainerArithmetic):
            # It is a container, we perform operations as normal.
            operand_data = operand.data
            operand_uncertainty = operand.uncertainty
            operand_unit = operand.data_unit
        else:
            # The operand is just a single value, so, we handle it as so.
            # We assume a single value does not have any uncertainty that
            # we really care about.
            # We trust the addition of non-unit values respect the unit
            # of the container.
            operand_data = operand
            operand_uncertainty = np.zeros_like(self.uncertainty)
            operand_unit = self.data_unit

        # Addition and subtraction are unique in that we need to also check
        # the data units.
        if self.data_unit != operand_unit:
            logging.error(
                error_type=logging.ArithmeticalError,
                message=(
                    "The Lezargus container data/flux unit"
                    f" {self.data_unit} is not the same as the operand"
                    f" unit {operand_unit}."
                ),
            )

        # Now we do the subtraction.
        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)
        result.data, result.uncertainty = lezargus.library.math.subtract(
            minuend=self.data,
            subtrahend=operand_data,
            minuend_uncertainty=self.uncertainty,
            subtrahend_uncertainty=operand_uncertainty,
        )
        # All done.
        return result

    def __mul__(self: hint.Self, operand: float | hint.Self) -> hint.Self:
        """Perform a multiplication operation.

        Parameters
        ----------
        operand : Self-like
            The container object to multiply to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__verify_arithmetic_operation(operand=operand):
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
            operand_unit = operand.data_unit
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_unit = lezargus.library.conversion.parse_astropy_unit(
                unit_string="",
            )
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # Now we perform the multiplication.
        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)
        result.data, result.uncertainty = lezargus.library.math.multiply(
            multiplier=self.data,
            multiplicand=operand_data,
            multiplier_uncertainty=self.uncertainty,
            multiplicand_uncertainty=operand_uncertainty,
        )
        # We also need to propagate the unit.

        result.data_unit = self.data_unit * operand_unit
        # All done.
        return result

    def __rmul__(self: hint.Self, operand: hint.Self) -> hint.Self:
        """Perform a multiplication operation, commutative with __mul__.

        Parameters
        ----------
        operand : Self-like
            The container object to multiply to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        return self.__mul__(operand=operand)

    def __truediv__(
        self: hint.Self,
        operand: float | hint.Self,
    ) -> hint.Self:
        """Perform a true division operation.

        Parameters
        ----------
        operand : Self-like
            The container object to true divide to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__verify_arithmetic_operation(operand=operand):
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
            operand_unit = operand.data_unit
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_unit = lezargus.library.conversion.parse_astropy_unit(
                unit_string="",
            )
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # Now we perform the division.
        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)
        result.data, result.uncertainty = lezargus.library.math.divide(
            numerator=self.data,
            denominator=operand_data,
            numerator_uncertainty=self.uncertainty,
            denominator_uncertainty=operand_uncertainty,
        )
        # We also need to propagate the unit.
        result.data_unit = self.data_unit / operand_unit
        # All done.
        return result

    def __pow__(self: hint.Self, operand: float | hint.Self) -> hint.Self:
        """Perform a true division operation.

        Parameters
        ----------
        operand : Self-like
            The container object to exponentiate to this.

        Returns
        -------
        result : Self-like
            A copy of this object with the resultant calculations done.

        """
        # We need to check the applicability of the operand and the operation
        # being attempted. The actual return is likely not needed, but we
        # still test for it.
        if not self.__verify_arithmetic_operation(operand=operand):
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
        no_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string="",
        )
        if isinstance(operand, LezargusContainerArithmetic):
            operand_data = operand.data
            operand_unit = operand.data_unit
            operand_uncertainty = operand.uncertainty
        else:
            # We assume a single value does not have any uncertainty that
            # we really care about.
            operand_data = operand
            operand_unit = no_unit
            operand_uncertainty = np.zeros_like(self.uncertainty)

        # Now we perform the exponentiation.
        # We do not want to modify our own objects as that goes against the
        # the main idea of operator operations.
        result = copy.deepcopy(self)
        result.data, result.uncertainty = lezargus.library.math.exponentiate(
            base=self.data,
            exponent=operand_data,
            base_uncertainty=self.uncertainty,
            exponent_uncertainty=operand_uncertainty,
        )

        # We propagate the units; however, by general practice, the exponent
        # should not have a unit.
        if operand_unit != no_unit:
            logging.error(
                error_type=logging.ArithmeticalError,
                message=(
                    "The exponent should be unitless, it has units:"
                    f" {operand_unit}"
                ),
            )
        result.data_unit = self.data_unit**operand

        # All done.
        return result

    @classmethod
    def _read_fits_file(
        cls: type[hint.Self],
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
            spectral_scale,
            pixel_scale,
            slice_scale,
            mask,
            flags,
        ) = lezargus.library.fits.read_lezargus_fits_file(filename=filename)
        # Check if the FITS file format is correct for the container.
        lz_fits_encode = header.get("LZ_FITSF", None)
        if lz_fits_encode != cls.__name__:
            logging.error(
                error_type=logging.FileError,
                message=(
                    f"The following FITS file {filename} is coded to be a"
                    f" {lz_fits_encode} type of FITS file, but it"
                    f" is being loaded with the {cls.__name__}."
                ),
            )

        # Loading the file up.
        container = cls(
            header=header,
            wavelength=wavelength,
            data=data,
            uncertainty=uncertainty,
            wavelength_unit=wavelength_unit,
            spectral_scale=spectral_scale,
            data_unit=data_unit,
            pixel_scale=pixel_scale,
            slice_scale=slice_scale,
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
        # The Lezargus container is the FITS cube format. However, we do not
        # want to modify the actual header itself.
        header_copy = self.header.copy()
        header_copy["LZ_FITSF"] = type(self).__name__
        # We send the file to the library function write.
        lezargus.library.fits.write_lezargus_fits_file(
            filename=filename,
            header=header_copy,
            wavelength=self.wavelength,
            data=self.data,
            uncertainty=self.uncertainty,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            spectral_scale=self.spectral_scale,
            pixel_scale=self.pixel_scale,
            slice_scale=self.slice_scale,
            mask=self.mask,
            flags=self.flags,
            overwrite=overwrite,
        )
        # All done.

    def to_unit(
        self: hint.Self,
        data_unit: str | hint.Unit | None = None,
        wavelength_unit: str | hint.Unit | None = None,
    ) -> hint.Self:
        """Convert the units of the current data to the new set of units.

        This function only does simple unit conversion. Any equivalency
        conversion is not handled.

        Parameters
        ----------
        data_unit : str | Unit
            The unit we will be converting to. If the unit is not compatible,
            we raise an exception. If None, we default to no data unit
            conversion.
        wavelength_unit : str | Unit | None, default = None
            A new wavelength unit to convert to. If the unit is not compatible,
            we raise an exception. If None, we default to no wavelength unit
            conversion.

        Returns
        -------
        converted : Self-like
            The converted container, after the unit conversion.

        """
        # We need to handle the default cases where no unit conversion is
        # wanted.
        do_wavelength = wavelength_unit is not None
        do_data = data_unit is not None

        # Parsing the units, in the event that they are strings.
        wavelength_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=wavelength_unit,
        )
        data_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=data_unit,
        )

        # It is easiest to work on a copy.
        converted = copy.deepcopy(self)

        # Wavelength conversion, if specified.
        if do_wavelength:
            converted.wavelength = lezargus.library.conversion.convert_units(
                value=self.wavelength,
                value_unit=self.wavelength_unit,
                result_unit=wavelength_unit,
            )
            converted.wavelength_unit = wavelength_unit

        # Data and uncertainty conversion, if specified.
        if do_data:
            converted.data = lezargus.library.conversion.convert_units(
                value=self.data,
                value_unit=self.data_unit,
                result_unit=data_unit,
            )
            converted.uncertainty = lezargus.library.conversion.convert_units(
                value=self.uncertainty,
                value_unit=self.uncertainty_unit,
                result_unit=data_unit,
            )
            converted.data_unit = data_unit

        # All done.
        return converted

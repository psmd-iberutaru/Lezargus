"""Arithmetical operations between Lezargus container classes.

We extend the Lezargus container classes to properly have arithmetical
operations (such as addition and subtraction) to simplify the code in the
main container classes themselves.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import numpy as np

import lezargus
from lezargus.library import logging


def _verify_wavelength(container_1: hint.Any, container_2: hint.Any) -> bool:
    """Verify matching wavelengths between operating containers.

    Attempting to do operations between two containers with differing
    wavelengths is not proper and can lead to incorrect results. We check that
    the wavelengths are in the correct units, correct shape, and have similar
    values.

    Parameters
    ----------
    container_1 : Any
        The first of the containers we are comparing against.
    container_2 : Any
        The second of the containers we are comparing against.

    Returns
    -------
    verification : bool
        If True, then the verification of the wavelengths were complete and
        the containers are considered compatible.

    """
    # We need to get the wavelengths from the objects.
    wavelength_1 = getattr(container_1, "wavelength", None)
    wavelength_unit_1 = getattr(container_1, "wavelength_unit", None)
    wavelength_2 = getattr(container_2, "wavelength", None)
    wavelength_unit_2 = getattr(container_2, "wavelength_unit", None)

    # If both wavelengths just do not exist, they are considered compatible.
    # This is a concession on usability; well, as long as the units are also
    # compatible.
    if (
        (wavelength_1 is None)
        and (wavelength_2 is None)
        and (wavelength_unit_1 == wavelength_unit_2)
    ):
        verification = True
        return verification

    # As both wavelengths exist, we format them to Numpy, and the unit to
    # Astropy.
    wavelength_1 = np.asarray(wavelength_1)
    wavelength_2 = np.asarray(wavelength_2)
    wavelength_unit_1 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=wavelength_unit_1,
    )
    wavelength_unit_2 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=wavelength_unit_2,
    )

    # The units must have the same units.
    if wavelength_unit_1 != wavelength_unit_2:
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                f"Container arithmetic wavelength units {wavelength_unit_1} and"
                f" {wavelength_unit_2} are not the same."
            ),
        )
        verification = False
        return verification

    # The wavelength shapes must be the same (this is a precursor to
    # determining if the wavelength grid is close enough).
    if wavelength_1.shape != wavelength_2.shape:
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container arithmetic wavelength shape"
                f" {wavelength_1.shape} and {wavelength_2.shape} are not the"
                " same."
            ),
        )
        verification = False
        return verification

    # The wavelengths themselves must generally be close together, otherwise
    # the adding of mismatched wavelengths would happen.
    if not np.allclose(wavelength_1, wavelength_2):
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container arithmetic wavelength grid values are not similar."
            ),
        )
        verification = False
        return verification

    # If the function did not exit early, it must have passed the above
    # verifications.
    verification = True
    return verification


def _verify_units(container_1: hint.Any, container_2: hint.Any) -> bool:
    """Verify matching data units between operating containers.

    Attempting to do operations between two containers with differing
    units is not proper and can lead to incorrect results. We check that
    the units are the same. We do not do implicit conversions.

    Parameters
    ----------
    container_1 : Any
        The first of the containers we are comparing against.
    container_2 : Any
        The second of the containers we are comparing against.

    Returns
    -------
    verification : bool
        If True, then the verification of the data units were complete and
        the containers are considered compatible.

    """
    # We need to get the units from the objects.
    wavelength_unit_1 = getattr(container_1, "wavelength_unit", None)
    wavelength_unit_2 = getattr(container_2, "wavelength_unit", None)
    data_unit_1 = getattr(container_1, "data_unit", None)
    data_unit_2 = getattr(container_2, "data_unit", None)
    uncertainty_unit_1 = getattr(container_1, "data_unit", None)
    uncertainty_unit_2 = getattr(container_2, "data_unit", None)

    # We parse the units to a more mature unit library, just in case.
    wavelength_unit_1 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=wavelength_unit_1,
    )
    wavelength_unit_2 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=wavelength_unit_2,
    )
    data_unit_1 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=data_unit_1,
    )
    data_unit_2 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=data_unit_2,
    )
    uncertainty_unit_1 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=uncertainty_unit_1,
    )
    uncertainty_unit_2 = lezargus.library.conversion.parse_astropy_unit(
        unit_input=uncertainty_unit_2,
    )

    # And we just need to make sure all of the units are matching.
    if wavelength_unit_1 != wavelength_unit_2:
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                f"Container arithmetic wavelength units {wavelength_unit_1} and"
                f" {wavelength_unit_2} are not the same."
            ),
        )
        verification = False
        return verification

    if data_unit_1 != data_unit_2:
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                f"Container arithmetic data units {data_unit_1} and"
                f" {data_unit_2} are not the same."
            ),
        )
        verification = False
        return verification

    if uncertainty_unit_1 != uncertainty_unit_2:
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container arithmetic uncertainty units"
                f" {uncertainty_unit_1} and {uncertainty_unit_2} are not the"
                " same."
            ),
        )
        verification = False
        return verification

    # If the function did not exit early, it must have passed the above
    # verifications.
    verification = True
    return verification


def _verify_shape(container_1: hint.Any, container_2: hint.Any) -> bool:
    """Verify matching data shape between operating containers.

    Attempting to do operations between two containers with differing
    data shapes is impossible and will raise an error. We check that
    the shapes of the data are compatible. We do not do implicit conversions.

    Parameters
    ----------
    container_1 : Any
        The first of the containers we are comparing against.
    container_2 : Any
        The second of the containers we are comparing against.

    Returns
    -------
    verification : bool
        If True, then the verification of the data shapes were complete and
        the containers are considered compatible.

    """
    # We need to get the data (and the array shapes for the data) from the
    # objects. Propagation of masks and flags should be done so the check is
    # applied.
    data_1 = getattr(container_1, "data", None)
    data_2 = getattr(container_2, "data", None)
    uncertainty_1 = getattr(container_1, "uncertainty", None)
    uncertainty_2 = getattr(container_2, "uncertainty", None)
    mask_1 = getattr(container_1, "mask", None)
    mask_2 = getattr(container_2, "mask", None)
    flags_1 = getattr(container_1, "flags", None)
    flags_2 = getattr(container_2, "flags", None)

    # Checking the shapes are equal to each other. These objects should be
    # Numpy-based arrays. A wrapper function to handle the None defaults if
    # the object does not exist.
    def _get_numpy_shape(input_: hint.Any | None) -> tuple:
        """Get the Numpy shape of the array, handling strange objects."""
        if input_ is None:
            # There is no shape.
            shape = ()
        else:
            try:
                shape = input_.shape
            except AttributeError:
                logging.error(
                    error_type=logging.DevelopmentError,
                    message=(
                        f"Object of type {type(input_)} does not have a shape"
                        " attribute, cannot verify shape for container"
                        " arithmetic."
                    ),
                )
        # All done.
        return shape

    # Checking all of the shapes.
    if _get_numpy_shape(input_=data_1) != _get_numpy_shape(input_=data_2):
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container data shapes do not have the same shape:"
                f" {_get_numpy_shape(input_=data_1)} versus"
                f" {_get_numpy_shape(input_=data_2)}."
            ),
        )
        verification = False
        return verification

    if _get_numpy_shape(input_=uncertainty_1) != _get_numpy_shape(
        input_=uncertainty_2,
    ):
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container uncertainty shapes do not have the same shape:"
                f" {_get_numpy_shape(input_=uncertainty_1)} versus"
                f" {_get_numpy_shape(input_=uncertainty_2)}."
            ),
        )
        verification = False
        return verification

    if _get_numpy_shape(input_=mask_1) != _get_numpy_shape(input_=mask_2):
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container mask shapes do not have the same shape:"
                f" {_get_numpy_shape(input_=mask_1)} versus"
                f" {_get_numpy_shape(input_=mask_2)}."
            ),
        )
        verification = False
        return verification

    if _get_numpy_shape(input_=flags_1) != _get_numpy_shape(input_=flags_2):
        logging.error(
            error_type=logging.ArithmeticalError,
            message=(
                "Container flags shapes do not have the same shape:"
                f" {_get_numpy_shape(input_=flags_1)} versus"
                f" {_get_numpy_shape(input_=flags_2)}."
            ),
        )
        verification = False
        return verification

    # If the function did not exit early, it must have passed the above
    # verifications.
    verification = True
    return verification


def lezargus_spectrum_spectrum_arithmetic(
    spectrum_1: hint.LezargusSpectrum,
    spectrum_2: hint.LezargusSpectrum,
    operation: str,
) -> hint.LezargusSpectrum:
    """Perform a provided operation between two spectra.

    This function adds two Lezargus spectra together, after appropriate checks
    are done to ensure that the objects are fully compatible with each other.

    Parameters
    ----------
    spectrum_1 : LezargusSpectrum
        The first spectrum which we will apply the operation with.
    spectrum_2 : LezargusSpectrum
        The second spectrum which we will apply the operation with.
    operation : str
        The supported operation, typically the elementary arithmetic
        operations.

    Returns
    -------
    result_spectrum : LezargusSpectrum
        The resulting spectrum after the addition.

    """
    # For later.
    lezargus.library.wrapper.do_nothing(spectrum_1, spectrum_2, operation)

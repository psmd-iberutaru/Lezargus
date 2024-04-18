"""Container classes to hold both atmospheric transmission and radiance.

We define small wrappers to hold atmospheric transmission and radiance data
so that it can be used more easily. The data itself usually has been derived
from PSG. These container classes are just intuitive wrappers around
interpolation.
"""

import lezargus
from lezargus.library import hint
from lezargus.library import logging


class AtmosphereSpectrumGenerator:
    """Atmospheric spectrum generator/interpolator.

    This class generates, via interpolation of a pre-computed grid,
    atmospheric transmission and radiance spectrum.

    Attributes
    ----------
    wavelength : ndarray
        The wavelength axis of the grid data we are interpolating over.
    zenith_angle : ndarray
        The zenith angle axis of the grid data we are interpolating over.
        The actual interpolation uses airmass instead of zenith angle.
    airmass : ndarray
        The calculated airmass axis of the grid data we are interpolating over.
    pwv : ndarray
        The precipitable water vapor axis of the grid data we are interpolating
        over.
    transmission : ndarray
        The transmission data grid, axes defined by other attributes.
    radiance : ndarray
        The radiance data grid, axes defined by other attributes.
    _transmission_interpolator : RegularNDInterpolator
        The interpolator class for the transmission data which we use as the
        backbone of this generator.
    _radiance_interpolator : RegularNDInterpolator
        The interpolator class for the transmission data which we use as the
        backbone of this generator.

    """

    def __init__(
        self: "AtmosphereSpectrumGenerator",
        wavelength: hint.ndarray,
        zenith_angle: hint.ndarray,
        pwv: hint.ndarray,
        transmission: hint.ndarray,
        radiance: hint.ndarray,
    ) -> None:
        """Initialize the atmospheric transmission and radiance container.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength axis of the grid data that defines the transmission
            and radiance data.
        zenith_angle : ndarray
            The zenith angle axis of the grid data that defines the
            transmission and radiance data.
        pwv : ndarray
            The precipitable water vapor axis of the grid data that defines
            the transmission and radiance data.
        transmission : ndarray
            The transmission data grid, axes defined by other attributes.
        radiance : ndarray
            The radiance data grid, axes defined by other attributes.

        """
        # Interpolation using airmass over zenith angle makes more sense as
        # airmass has a linear response.
        airmass = lezargus.library.atmosphere.airmass(zenith_angle=zenith_angle)

        # We check that the shape provided by the defining axes matches the
        # data shape. The provided axis order is reversed of Numpy's
        # conventions.
        domain = (wavelength, airmass, pwv)
        domain_shape = tuple(domaindex.size for domaindex in domain)
        if (
            reversed(domain_shape) != transmission.shape
            or reversed(domain_shape) != radiance.shape
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The shape of transmission {transmission.shape} or"
                    f" radiance {radiance.shape} does not match the expected"
                    f" shape of {domain_shape} from the input axes."
                ),
            )

        # We can properly build our class.
        self.wavelength = wavelength
        self.zenith_angle = zenith_angle
        self.airmass = airmass
        self.pwv = pwv
        self.transmission = transmission
        self.radiance = radiance

        # Building the interpolators.
        self._transmission_interpolator = (
            lezargus.library.interpolate.RegularNDInterpolate(
                domain=domain,
                v=self.transmission,
            )
        )
        self._radiance_interpolator = (
            lezargus.library.interpolate.RegularNDInterpolate(
                domain=domain,
                v=self.radiance,
            )
        )

    def generate_transmission_spectra(
        self: hint.Self,
        zenith_angle: float,
        pwv: float,
    ) -> hint.LezargusSpectrum:
        """TODO."""

    def generate_radiance_spectra(
        self: hint.Self,
        zenith_angle: float,
        pwv: float,
    ) -> hint.LezargusSpectrum:
        """TODO."""

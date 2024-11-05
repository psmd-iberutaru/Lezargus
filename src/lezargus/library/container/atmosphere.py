"""Container classes to hold both atmospheric transmission and radiance.

We define small wrappers to hold atmospheric transmission and radiance data
so that it can be used more easily. The data itself usually has been derived
from PSG. These container classes are just intuitive wrappers around
interpolation.
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


class AtmosphereSpectrumGenerator:
    """Atmospheric spectrum generator/interpolator.

    This class generates, via interpolation of a pre-computed grid. Typically
    this is used for atmospheric transmission and radiance spectrum, but,
    others may be used as well.

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
    data : ndarray
        The data grid, usually transmission or radiance. The axes are defined
        by the other attributes.
    wavelength_unit : Unit
        The wavelength unit of the wavelength array provided, if provided.
    data_unit : Unit
        The data unit of the data provided, if provided.
    spectral_scale : float
        The spectral resolution (scale) of the computed grid, if provided.
    _data_interpolator : RegularNDInterpolator
        The interpolator class for the data which we use as the backbone of
        this generator. This should not be called directly.

    """

    def __init__(
        self: AtmosphereSpectrumGenerator,
        wavelength: hint.NDArray,
        zenith_angle: hint.NDArray,
        pwv: hint.NDArray,
        data: hint.NDArray,
        wavelength_unit: hint.Unit | str = "m",
        data_unit: hint.Unit | str = "",
        spectral_scale: float | None = None,
    ) -> None:
        """Initialize the atmospheric transmission and radiance container.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength axis of the grid data that defines the transmission
            and radiance data.
        zenith_angle : ndarray
            The zenith angle axis of the grid data that defines the
            transmission and radiance data, in radians.
        pwv : ndarray
            The precipitable water vapor axis of the grid data that defines
            the transmission and radiance data, in millimeters.
        data : ndarray
            The atmospheric data for the generator to "generate" via
            interpolation. The shape of the data should match that created
            by the domain of the previous attributes.
        wavelength_unit : Unit or str, default = "m"
            The wavelength unit of the wavelength provided. By default and by
            convention, this should be in meters.
        data_unit : Unit or str
            The data unit of the data provided. By default, we assume a
            dimensionless spectrum.
        spectral_scale : float, default = 0
            The spectral (scale) resolution of the simulation.

        """
        # Interpolation using airmass over zenith angle makes more sense as
        # airmass has a linear response.
        airmass = lezargus.library.atmosphere.airmass(zenith_angle=zenith_angle)

        # We can properly build our class.
        self.wavelength = wavelength
        self.zenith_angle = zenith_angle
        self.airmass = airmass
        self.pwv = pwv
        self.data = data

        # Meta data information.
        self.wavelength_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=wavelength_unit,
        )
        self.data_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=data_unit,
        )
        self.spectral_scale = spectral_scale

        # Building the interpolators.
        domain = (wavelength, airmass, pwv)
        self._data_interpolator = (
            lezargus.library.interpolate.RegularNDInterpolate(
                domain=domain,
                v=self.data,
            )
        )

    def interpolate(
        self: hint.Self,
        wavelength: hint.NDArray | None = None,
        zenith_angle: float = 0,
        pwv: float = 0.005,
    ) -> hint.NDArray:
        """Generate atmospheric spectrum, through interpolation.

        Parameters
        ----------
        wavelength : ndarray, default = None
            The wavelengths to compute the spectrum at. If None,
            we default to the wavelength basis used to generate this class.
        zenith_angle : float, default = 0
            The zenith angle for the transmission spectrum generation,
            in radians.
        pwv : float, default = 0.005
            The precipitable water vapor for the transmission spectrum
            generation, in meters.

        Returns
        -------
        generated_data : ndarray
            The generated/interpolated data sampled at the provided
            wavelengths.

        """
        # The default wavelength if not provided.
        wavelength = self.wavelength if wavelength is None else wavelength

        # Airmass is the defining axis, not zenith angle, because of its
        # more linear nature.
        airmass = lezargus.library.atmosphere.airmass(zenith_angle=zenith_angle)

        # Finally, we interpolate the values. Note, we need parallel arrays
        # to define the points, so we just extend the airmass and PWV values.
        airmass_extension = np.full_like(wavelength, airmass)
        pwv_extension = np.full_like(wavelength, pwv)
        generated_data = self._data_interpolator.interpolate(
            wavelength,
            airmass_extension,
            pwv_extension,
        )

        # All done.
        return generated_data

    def interpolate_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray | None = None,
        zenith_angle: float = 0,
        pwv: float = 0.005,
    ) -> hint.LezargusSpectrum:
        """Generate a atmospheric LezargusSpectrum, through interpolation.

        This function really is a wrapper around the usual interpolator,
        repackaging the results as a LezargusSpectrum. Note, most of the
        meta data parameters are wrong and need to be fixed manually.

        Parameters
        ----------
        wavelength : ndarray, default = None
            The wavelengths to compute the spectrum at. If None,
            we default to the wavelength basis used to generate this class.
        zenith_angle : float, default = 0
            The zenith angle for the transmission spectrum generation,
            in radians.
        pwv : float, default = 0.005
            The precipitable water vapor for the transmission spectrum
            generation, in meters.

        Returns
        -------
        generated_spectrum : LezargusSpectrum
            The generated/interpolated data sampled at the provided
            wavelengths. This is packaged as a LezargusSpectrum.

        """
        # The default.
        wavelength = self.wavelength if wavelength is None else wavelength

        # We find the generated data.
        generated_data = self.interpolate(
            wavelength=wavelength,
            zenith_angle=zenith_angle,
            pwv=pwv,
        )

        # We repackage it.
        generated_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=generated_data,
            uncertainty=None,
            spectral_scale=self.spectral_scale,
            wavelength_unit=self.wavelength_unit,
            data_unit=self.data_unit,
            header=None,
        )
        return generated_spectrum

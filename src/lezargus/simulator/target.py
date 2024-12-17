"""Simulation code to simulate an astrophysical object or target.

This code simulates an astrophysical target, creating the theoretical cube
which represents it on sky.

We name this file "target.py" to prematurely avoid name conflicts with the
Python built-in "object".
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import astropy.constants
import numpy as np

import lezargus
from lezargus.library import logging


class TargetSimulator:
    """Simulate an astrophysical target.

    This class is a bit of a wrapper, acting as an effective means of
    generating the simulation of an astrophysical field. However, this class
    nevertheless acts as a nice point of abstraction for the instruments
    simulators.

    Other property attributes exist but are not documented here as they have
    their own docstrings. In order to reduce the amount of repeat computing,
    every important computed property likely has a cache variant which stores
    the most recently calculated result as a cache and uses it instead of
    recomputing things. This is generally an internal application.

    """

    target = None
    """LezargusCube : The target, represented as a LezargusCube. The exact
    properties of the cube are determined from how the simulator was created."""

    target_spectrum = None
    """LezargusSpectrum : The spectrum of the target. This is only made if the
    target cube was made by extending a point source, making the cube from a
    Spectrum; otherwise, this is None."""

    atmosphere = None
    """AtmosphereSimulator : The atmosphere simulator which describes and
    simulates atmospheric effects. If not provided by
    py:meth:`add_atmosphere`, this defaults to None."""

    use_cache = True
    """bool : If True, we cache calculated values so that they do not need to
    be calculated every time when not needed. If False, caches are never
    returned and instead everything is always recomputed."""

    # Cache objects.
    _cache_target_photon = None
    _cache_observed = None

    def __init__(
        self: TargetSimulator,
        *args: hint.Any,
        _cube: hint.LezargusCube,
        **kwargs: hint.Any,
    ) -> None:
        """Create the target simulator.

        This class should not be called directly, but the helper class
        functions should be used instead to create the simulator. Internal
        options are provided and documented but their use is discouraged.

        Parameters
        ----------
        *args : Any
            Arguments we catch. If there are any arguments, we give an error.
        _cube : hint.LezargusCube
            The main target cube. It should be the same as py:attr:`target`.
        **kwargs : Any
            Extra keyword arguments we catch. If there are any, we give an
            error.

        Returns
        -------
        None

        """
        if len(args) != 0 or len(kwargs) != 0:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "User arguments not accepted. TargetSimulator should be"
                    " created from helper class methods."
                ),
            )

        # The input cube is the same as the target cube.
        self.target = _cube

    @classmethod
    def from_blackbody(
        cls: type[hint.Self],
        wavelength: hint.NDArray,
        temperature: float,
        magnitude: float,
        photometric_filter: (
            hint.PhotometricABFilter | hint.PhotometricVegaFilter
        ),
        spatial_grid_shape: tuple,
        spatial_fov_shape: tuple,
        spectral_scale: float,
        **kwargs: hint.Any,
    ) -> hint.Self:
        """Create a target simulation object from a point source blackbody.

        This is a convenience wrapper around the :py:meth:`from_spectrum`
        function. Some required parameters for that function are needed and
        not otherwise described here. See the documentation for that
        function to properly use this wrapper.

        Parameters
        ----------
        wavelength : ndarray
            The wavelength sampling that we will sample the black body at.
        temperature : float
            The temperature of the black body spectrum.
        magnitude : float
            The magnitude of the object in the photometric filter system
            provided.
        photometric_filter : PhotometricVegaFilter | PhotometricABFilter
            The photometric filter which we are using to scale the blackbody
            to match the magnitude provided; which is assumed to be in the
            correct photometric system.
        spatial_grid_shape : tuple
            The spatial pixel grid shape. This defines the array shape of the
            simulation's spatial component. The pixel and slice scale is
            calculated from this and the field of view.
        spatial_fov_shape : tuple
            The defined field of view shape. This defines the on-sky field of
            view shape of the array, and is in radians. The pixel and slice
            scale is calculated from this and the field of view.
        spectral_scale : float
            The spectral scale of the simulated spectra, as a resolution,
            in wavelength separation (in meters) per pixel.
        **kwargs : Any
            Additional keyword arguments passed to the py:meth:`from_spectrum`
            function which does the heavy lifting.

        Returns
        -------
        target_instance : TargetSimulator
            The target simulator instance derived from the input parameters.

        """
        # We construct the blackbody function.
        blackbody_function = lezargus.library.wrapper.blackbody_function(
            temperature=temperature,
        )
        # Then we evaluate the blackbody function, of course the scale of which
        # will be wrong but it will be fixed.
        blackbody_flux = blackbody_function(wavelength)
        # We integrate over the solid angle.
        solid_angle = np.pi
        integrated_blackbody_flux = blackbody_flux * solid_angle
        # Packaging the spectra. The pixel scale and slice scales are handled
        # later.
        blackbody_spectra = lezargus.library.container.LezargusSpectrum(
            wavelength=wavelength,
            data=integrated_blackbody_flux,
            uncertainty=None,
            wavelength_unit="m",
            data_unit="W m^-2 m^-1",
            spectral_scale=spectral_scale,
            pixel_scale=None,
            slice_scale=None,
            mask=None,
            flags=None,
            header=None,
        )

        # We scale the flux, applying a photometric correction for the
        # provided filter profile, zero point, and filter magnitude.
        # We do not really care about the error term.
        correction_factor, __ = (
            photometric_filter.calculate_photometric_correction(
                spectrum=blackbody_spectra,
                magnitude=magnitude,
                magnitude_uncertainty=0,
            )
        )

        # Photometrically calibrating it.
        target_spectrum = blackbody_spectra * correction_factor

        # We pass it to the main function for us to create the actual target
        # from the spectrum.
        target_instance = cls.from_spectrum(
            spectrum=target_spectrum,
            spatial_grid_shape=spatial_grid_shape,
            spatial_fov_shape=spatial_fov_shape,
            **kwargs,
        )
        return target_instance

    @classmethod
    def from_spectrum(
        cls: type[hint.Self],
        spectrum: hint.LezargusSpectrum,
        spatial_grid_shape: tuple,
        spatial_fov_shape: tuple,
        location: tuple | str = "center",
    ) -> hint.Self:
        """Create a target simulation object from a point source spectrum.

        Parameters
        ----------
        spectrum : LezargusSpectrum
            The point source spectrum which we will use as the target to make
            the target cube of. The spectrum should be an energy-based
            spectrum.
        spatial_grid_shape : tuple
            The spatial pixel grid shape. This defines the array shape of the
            simulation's spatial component. The pixel and slice scale is
            calculated from this and the field of view.
        spatial_fov_shape : tuple
            The defined field of view shape. This defines the on-sky field of
            view shape of the array, and is in radians The pixel and slice
            scale is calculated from this and the field of view.
        location : tuple or str, default = "center"
            Where the spectra, as a point source, be placed spatially. If a
            string, we compute the location from the instruction:

                - `center` : It is placed in the center, or close to it,
                  rounded down, for even valued shapes.

        Returns
        -------
        target_instance : TargetSimulator
            The target simulator instance derived from the input parameters.

        """
        # We first check if we have a proper LezargusCube spectrum.
        if not isinstance(
            spectrum,
            lezargus.library.container.LezargusSpectrum,
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input cube is type {type(spectrum)}, not an expected"
                    " LezargusSpectrum."
                ),
            )

        # We calculate the pixel and slice scale from the provided grid.
        pixel_scale = spatial_fov_shape[0] / spatial_grid_shape[0]
        slice_scale = spatial_fov_shape[1] / spatial_grid_shape[1]
        spectrum.pixel_scale = pixel_scale
        spectrum.slice_scale = slice_scale

        # We assume that background space is dark, so a zero fill value.
        background_data = 0
        background_uncertainty = 0

        # From there, we can create a cube based on broadcasting the spectrum
        # into a cube.
        broadcast_cube = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=spectrum,
                shape=spatial_grid_shape,
                location=location,
                fill_value=background_data,
                fill_uncertainty=background_uncertainty,
            )
        )

        # We pass it to the main function for us to create the actual target
        # from the derived cube.
        target_instance = cls.from_cube(cube=broadcast_cube)

        # We save the target spectrum in the event it is needed.
        target_instance.target_spectrum = spectrum
        return target_instance

    @classmethod
    def from_cube(
        cls: type[hint.Self],
        cube: hint.LezargusCube,
    ) -> hint.Self:
        """Create a target simulation object from a provided cube.

        This function is just a formality. Usually if a cube has already been
        defined and provided, the cube itself is the data of the simulated
        target. Nevertheless, we allow for the specification of a target
        based on a cube to provide this common interface.

        Parameters
        ----------
        cube : LezargusCube
            The target cube which define the object we are simulating.

        Returns
        -------
        target_instance : TargetSimulator
            The target simulator instance derived from the input parameters.

        """
        # We first check if we have a proper LezargusCube.
        if not isinstance(cube, lezargus.library.container.LezargusCube):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input cube is type {type(cube)}, not an expected"
                    " LezargusCube."
                ),
            )
        # The cube provided is the same as the target cube.
        target_instance = cls(_cube=cube)
        return target_instance

    def clear_cache(self: hint.Self) -> None:
        """Clear the cache of computed result objects.

        This function clears the cache of computed results, allowing for
        updated values to properly be utilized in future calculations and
        simulations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # We get all of the names of the cache attributes to then clear.
        cache_prefix = "_cache"
        self_attributes = dir(self)
        cache_attributes = [
            keydex
            for keydex in self_attributes
            if keydex.startswith(cache_prefix)
        ]
        # Removing the cache values by removing their reference and then
        # setting them to None as the default.
        for keydex in cache_attributes:
            setattr(self, keydex, None)
        # All done.

    @staticmethod
    def _convert_to_photon(
        container: hint.LezargusContainerArithmetic,
    ) -> hint.LezargusContainerArithmetic:
        """Convert Lezargus spectral flux density to photon flux density.

        This function is a convenience function to convert the spectral flux
        density of any container to a photon flux density. Please note that
        the units may change in unexpected ways because of unit conversions
        related to the constants and unit decomposition.

        Parameters
        ----------
        container : LezargusContainerArithmetic
            The container we are converting, or more accurately, a subclass
            of the container.

        Returns
        -------
        photon_container : LezargusContainerArithmetic
            The converted container as a photon flux instead of an energy flux.
            However, please note that the units may change in unexpected ways.

        """
        # It is easiest to work in SI units.
        si_wavelength_unit = "m"
        si_data_unit = "W m^-2 m^-1"
        si_container = container.to_unit(
            data_unit=si_data_unit,
            wavelength_unit=si_wavelength_unit,
        )

        # We determine the energy of the photon at the provided wavelength.
        # The value is the only one needed as we are working in SI.
        photon_energy = (astropy.constants.h * astropy.constants.c) / (
            si_container.wavelength * si_container.wavelength_unit
        )
        photon_energy_value = photon_energy.value
        # We keep the "photon" unit implicit to avoid unit conversion
        # problems.
        photon_energy_unit = photon_energy.unit

        # Broadcasting it so we can apply it as a simple division.
        broadcast_conversion = np.broadcast_to(
            photon_energy_value,
            shape=si_container.data.shape,
        )

        # Converting to a photon flux only requires us to do calculations
        # on the data and uncertainty.
        si_container.data = si_container.data / broadcast_conversion
        si_container.uncertainty = (
            si_container.uncertainty / broadcast_conversion
        )
        si_container.data_unit = si_container.data_unit / photon_energy_unit

        # As the actual "photon" unit was implicit this entire time, we add it
        # to be explicit.
        photon_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string="photon",
        )
        si_container.data_unit = (
            si_container.data_unit * photon_unit
        ).decompose()

        # Aliasing.
        photon_container = si_container
        # All done.
        return photon_container

    @property
    def at_target_spectrum(self: hint.Self) -> hint.LezargusCube | None:
        """Alias for py:attr:`target_spectrum` to match naming convention.

        By self-imposed convention, the attributes are generally named as
        `at_[stage]` where the result is the simulated result right after
        simulating whichever stage is named.

        This may be a read-only alias, for for most cases, that is fine.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target_spectrum

    @property
    def at_target(self: hint.Self) -> hint.LezargusCube:
        """Alias for py:attr:`target` to match naming convention.

        By self-imposed convention, the attributes are generally named as
        `at_[stage]` where the result is the simulated result right after
        simulating whichever stage is named.

        This may be a read-only alias, for for most cases, that is fine.

        Parameters
        ----------
        None

        Returns
        -------
        target : LezargusCube
            The energy based flux simulation data cube of the target.

        """
        return self.target

    @property
    def at_target_photon(self: hint.Self) -> hint.LezargusCube:
        """Target photon flux, calculated from target spectral energy density.

        Please note that the units may change in unexpected ways because of
        unit conversions related to the constants and unit decomposition.

        Parameters
        ----------
        None

        Returns
        -------
        target_photon : LezargusCube
            Exactly the same as the target, except the data (flux) is a photon
            flux and not an energy based flux.

        """
        # We use a cached value if there exists one.
        if self._cache_target_photon is not None and self.use_cache:
            return self._cache_target_photon

        # No valid cache, computing it ourselves.
        # A simple conversion.
        target_photon = self._convert_to_photon(container=self.at_target)

        # Saving the result later in the cache.
        if self.use_cache:
            self._cache_target_photon = target_photon
        return target_photon

    def add_atmosphere(
        self: hint.Self,
        atmosphere: hint.AtmosphereSimulator,
    ) -> None:
        """Add an atmosphere simulator to simulate the atmospheric effects.

        Note, we only allow one atmosphere at a time.

        Parameters
        ----------
        atmosphere : AtmosphereSimulator
            The atmosphere simulator to add to this target simulator to
            allow it to simulate different effects of the atmosphere.

        Returns
        -------
        None

        """
        # We just make sure the atmosphere is of the proper type before
        # just adding it.
        if isinstance(atmosphere, lezargus.simulator.AtmosphereSimulator):
            self.atmosphere = atmosphere
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input atmosphere, type {type(atmosphere)}, is not an"
                    " AtmosphereSimulator."
                ),
            )
            self.atmosphere = None
        # All done.

    @property
    def at_transmission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after atmospheric transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            atmospheric transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_target_photon

        # We actually need an atmosphere specified to simulate the atmosphere.
        if not isinstance(
            self.atmosphere,
            lezargus.simulator.AtmosphereSimulator,
        ):
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "No atmosphere applied, cannot apply transmission effects."
                ),
            )
            return previous_state

        # Determining the atmospheric transmission function. We broadcast it
        # to a cube to apply it to the previous state.
        transmission_spectrum = self.atmosphere.generate_transmission(
            template=previous_state,
        )
        transmission_cube = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=transmission_spectrum,
                shape=previous_state.data.shape,
                location="full",
                fill_value=0,
                fill_uncertainty=0,
            )
        )

        # Applying the transmission via simple multiplication of the
        # efficiencies.
        current_state = previous_state * transmission_cube

        return current_state

    @property
    def at_radiance(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after atmospheric radiance.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            atmospheric radiance.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_transmission

        # We actually need an atmosphere specified to simulate the atmosphere.
        if not isinstance(
            self.atmosphere,
            lezargus.simulator.AtmosphereSimulator,
        ):
            logging.error(
                error_type=logging.WrongOrderError,
                message="No atmosphere applied, cannot apply radiance effects.",
            )
            return previous_state

        # Determining the atmospheric transmission function. We broadcast it
        # to a cube to apply it to the previous state.
        radiance_spectrum = self.atmosphere.generate_radiance(
            template=previous_state,
        )
        radiance_cube = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=radiance_spectrum,
                shape=previous_state.data.shape,
                location="full",
                fill_value=0,
                fill_uncertainty=0,
            )
        )

        # We integrate the radiance to provide a proper photon spectral
        # irradiance which can be added.
        solid_angle = np.pi
        solid_angle_unit = lezargus.library.conversion.parse_astropy_unit("sr")
        irradiance_cube = radiance_cube * solid_angle
        irradiance_cube.data_unit = radiance_cube.data_unit * solid_angle_unit

        # The radiance provided by the atmosphere is in energy units, while
        # we are working in photon units.
        irradiance_photon_cube = self._convert_to_photon(
            container=irradiance_cube,
        )

        # The data units ought to be the same, else adding them together
        # becomes problematic.
        if irradiance_photon_cube.data_unit != previous_state.data_unit:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    "Irradiance photon cube unit"
                    f" {irradiance_photon_cube.data_unit} not the same as the"
                    f" previous state {previous_state.data_unit}."
                ),
            )

        # Applying the transmission via simple multiplication of the
        # efficiencies.
        current_state = previous_state + irradiance_photon_cube

        return current_state

    @property
    def at_seeing(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after atmospheric seeing.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            atmospheric seeing.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_radiance

        # We actually need an atmosphere specified to simulate the atmosphere.
        if not isinstance(
            self.atmosphere,
            lezargus.simulator.AtmosphereSimulator,
        ):
            logging.error(
                error_type=logging.WrongOrderError,
                message="No atmosphere applied, cannot apply radiance effects.",
            )
            return previous_state

        # We use the atmosphere to derive our atmospheric seeing kernels.
        # There are multiple kernels because seeing changes with wavelength.
        seeing_kernels = self.atmosphere.generate_seeing_kernels(
            template=previous_state,
        )

        # Applying the seeing effects via convolution.
        current_state = previous_state.convolve_image(
            kernel_stack=seeing_kernels,
        )

        return current_state

    @property
    def at_refraction(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after atmospheric refraction.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            atmospheric refraction.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_seeing

        # We actually need an atmosphere specified to simulate the atmosphere.
        if not isinstance(
            self.atmosphere,
            lezargus.simulator.AtmosphereSimulator,
        ):
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "No atmosphere applied, cannot apply refraction effects."
                ),
            )
            return previous_state

        # We use the atmosphere to derive our refraction vectors. We
        # rearrange the data to break the vectors into their components.
        refraction_vectors = self.atmosphere.generate_refraction_vectors(
            template=previous_state,
        )
        refraction_x = refraction_vectors[:, 0]
        refraction_y = refraction_vectors[:, 1]

        # Applying the refraction, modeling it as a shear transformation
        # along the spectral axis (parallel to the spatial axes).
        # We assume that it is only just more "uniform" sky everywhere else.
        # (We split up the function just for line length.)
        _functionality = lezargus.library.container.functionality
        current_state = _functionality.transform_shear_cube_spectral(
            cube=previous_state,
            x_shifts=refraction_x,
            y_shifts=refraction_y,
            mode="nearest",
            constant=np.nan,
        )
        return current_state

    @property
    def at_observed(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after an "observation".

        This object is basically the preferred alias for referring to the
        simulation at the point right after atmospheric effects. This is where
        the a target simulation ends and further simulation is done by
        specific instrument simulators.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after an "observation".

        """
        # We use a cached value if there exists one.
        if self._cache_observed is not None and self.use_cache:
            return self._cache_observed

        # No cached value, we calculate it from the previous state.
        previous_state = self.at_refraction
        # This is just an alias so the current state is the same.
        current_state = previous_state

        # Saving the result later in the cache.
        if self.use_cache:
            self._cache_observed = current_state
        return current_state

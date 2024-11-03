"""Simulation code to simulate a SPECTRE observation.

This code simulates a SPECTRE observation, from the target on the sky, all
the way to the detector. We use other smaller simulators to simulate common
things (like the object itself or the atmosphere) and the specifics of their
implementations can be found there. Implementation specifics to the SPECTRE
instrument itself are found here.

For ease, we package the smaller simulators within this single simulator class.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import copy

import numpy as np

import lezargus
from lezargus.library import logging


class SpectreSimulator:  # pylint: disable=too-many-public-methods
    """Main SPECTRE simulator class."""

    target: hint.TargetSimulator
    """The target object simulation. This is the working copy."""

    input_target: hint.TargetSimulator
    """The inputted target object simulation. We store this
    original copy as the actual working copy is being modified in place."""

    telescope: hint.IrtfTelescopeSimulator
    """The telescope simulation. This is the working copy."""

    input_telescope: hint.IrtfTelescopeSimulator
    """The inputted telescope simulation. We store this original copy as
    the actual working copy is being modified in place."""

    atmosphere: hint.AtmosphereSimulator
    """The atmosphere simulation. This is the working copy."""

    input_atmosphere: hint.AtmosphereSimulator | None
    """The inputted atmosphere simulation. We store this original copy as
    the actual working copy is being modified in place."""

    channel: hint.Literal["visible", "nearir", "midir"]  # noqa: F821, UP037
    """The specific channel of the three channels of SPECTRE that we are
    simulating."""

    use_cache = True
    """If True, we cache calculated values so that they do not need to
    be calculated every time when not needed. If False, caches are never
    returned and instead everything is always recomputed."""

    # Cache objects.
    _cache_at_telescope = None

    def __init__(
        self: SpectreSimulator,
        target: hint.TargetSimulator,
        telescope: hint.IrtfTelescopeSimulator,
        channel: str,
        atmosphere: hint.AtmosphereSimulator | None = None,
    ) -> None:
        """Initialize the SPECTRE simulator.

        Parameters
        ----------
        target : TargetSimulator
            The target simulator representing the object for simulated
            observing.
        telescope : IrtfTelescopeSimulator
            The telescope that the instrument is on. As the SPECTRE instrument
            is on the IRTF, we expect it to be an IRTF object.
        channel : str
            The name of the channel (of the three) which we are simulating.
        atmosphere : AtmosphereSimulator, default = None
            The intervening atmosphere simulation object. If None, we skip
            applying an atmosphere; use this if the target object already has
            the correct atmosphere applied.

        Returns
        -------
        None

        """
        # We first store the original copies of the provided inputs.
        self.input_target = copy.deepcopy(target)
        self.input_telescope = copy.deepcopy(telescope)
        self.input_atmosphere = copy.deepcopy(atmosphere)
        # And the copies that we use.
        self.target = target
        self.telescope = telescope

        # We need to make sure the atmosphere is properly applied.
        if atmosphere is None:
            # There is no provided atmosphere, check the target simulation.
            if target.atmosphere is None:
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message="No atmosphere provided or found in the target.",
                )
            else:
                self.atmosphere = copy.deepcopy(target.atmosphere)
        else:
            # We apply the atmosphere to the target.
            self.atmosphere = atmosphere
            self.target.add_atmosphere(atmosphere=self.atmosphere)

        # Parsing the channel name.
        channel = channel.casefold()
        if channel == "visible":
            self.channel = "visible"
        elif channel == "nearir":
            self.channel = "nearir"
        elif channel == "midir":
            self.channel = "midir"
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Channel name input {channel} does not match: visible,"
                    " nearir, midir."
                ),
            )
        # All done.

    @classmethod
    def from_parameters(
        cls: type[hint.Self],
        channel: str,
        wavelength: hint.NDArray,
        blackbody_temperature: float,
        magnitude: float,
        photometric_filter: (
            hint.PhotometricABFilter | hint.PhotometricVegaFilter
        ),
        spatial_shape: tuple,
        field_of_view: tuple,
        spectral_scale: float,
        atmosphere_temperature: float,
        atmosphere_pressure: float,
        atmosphere_ppw: float,
        atmosphere_pwv: float,
        atmosphere_seeing: float,
        zenith_angle: float,
        parallactic_angle: float,
        reference_wavelength: float,
        telescope_temperature: float,
        transmission_generator: hint.AtmosphereSpectrumGenerator | None = None,
        radiance_generator: hint.AtmosphereSpectrumGenerator | None = None,
    ) -> hint.Self:
        """Initialize the SPECTRE simulator, only using parameter values.

        By default, the initialization of the SPECTRE simulator requires the
        creation of three different inner simulator classes. This convenience
        function does that for the user, as long as they provide the
        environmental parameters for all three.

        We assume a blackbody simulated target, an Earth-like atmosphere, and
        the IRTF telescope. The parameters modify just the specifics. For a
        more detailed approach, please construct the classes instead.

        Parameters
        ----------
        channel : str
            The name of the channel that will be simulated; one of three
            channels: visible, nearir, and midir.
        wavelength : ndarray
            The wavelength basis of the simulator; this defines the wavelength
            axis and are its values.
        blackbody_temperature : float
            The blackbody temperature of the object that we are simulating,
            in Kelvin.
        magnitude : float
            The simulated magnitude of the object. The photometric filter
            system this magnitude is in must match the inputted photometric
            filter.
        photometric_filter : PhotometricABFilter or PhotometricVegaFilter
            The photometric filter (system) that the inputted magnitude is in.
        spatial_shape : tuple
            The spatial shape of the simulation array, the units are in pixels.
            This parameter should not be confused with the field of view
            parameter.
        field_of_view : tuple
            A tuple describing the field of view of the spatial area of the
            simulation array, the units are in radians. We suggest oversizing
            this a little more than the traditional 7.2 by 7.2 arcseconds.
        spectral_scale : float
            The spectral scale of the simulated spectra, as a resolution,
            in wavelength separation (in meters) per pixel.
        atmosphere_temperature : float
            The temperature of the intervening atmosphere, in Kelvin.
        atmosphere_pressure : float
            The pressure of the intervening atmosphere, in Pascal.
        atmosphere_ppw : float
            The partial pressure of water in the atmosphere, in Pascal.
        atmosphere_pwv : float
            The precipitable water vapor in the atmosphere, in meters.
        atmosphere_seeing : float
            The seeing of the atmosphere, given as the FWHM of the seeing disk,
            often approximated as a Gaussian distribution, at zenith and at
            the reference wavelength. The units are in radians.
        zenith_angle : float
            The zenith angle of the simulated object, at the reference
            wavelength in radians; primarily used to determine airmass.
        parallactic_angle : float
            The parallactic angle of the simulated object, in radians; primarily
            used to atmospheric dispersion direction.
        reference_wavelength : float
            The reference wavelength which defines the seeing and zenith angle
            parameters. Assumed to be in the same units as the provided
            wavelength axis.
        telescope_temperature : float
            The local temperature of the telescope, usually the temperatures
            of the primary and other mirrors; in Kelvin.
        transmission_generator : AtmosphereSpectrumGenerator, default = None
            The transmission spectrum generator used to generate the
            specific transmission spectra. If None, we default to the built-in
            generators.
        radiance_generator : AtmosphereSpectrumGenerator, default = None
            The transmission spectrum generator used to generate the
            specific transmission spectra. If None, we default to the built-in
            generators.


        Returns
        -------
        spectre_simulator : SpectreSimulator
            The simulator, with the properties provided from the parameters.

        """
        # Creating the three simulator objects.
        # The target.
        using_target = lezargus.simulator.TargetSimulator.from_blackbody(
            wavelength=wavelength,
            temperature=blackbody_temperature,
            magnitude=magnitude,
            photometric_filter=photometric_filter,
            spatial_grid_shape=spatial_shape,
            spatial_fov_shape=field_of_view,
            spectral_scale=spectral_scale,
        )
        # The atmosphere.
        using_atmosphere = lezargus.simulator.AtmosphereSimulator(
            temperature=atmosphere_temperature,
            pressure=atmosphere_pressure,
            ppw=atmosphere_ppw,
            pwv=atmosphere_pwv,
            seeing=atmosphere_seeing,
            zenith_angle=zenith_angle,
            parallactic_angle=parallactic_angle,
            reference_wavelength=reference_wavelength,
            transmission_generator=transmission_generator,
            radiance_generator=radiance_generator,
        )
        # The telescope.
        using_telescope = lezargus.simulator.IrtfTelescopeSimulator(
            temperature=telescope_temperature,
        )

        # Creating the main simulator class, using the above three component
        # simulators.
        spectre_simulator = cls(
            target=using_target,
            telescope=using_telescope,
            channel=channel,
            atmosphere=using_atmosphere,
        )

        # All done.
        return spectre_simulator

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
        # We first clear the cache of the target itself as well. Though it is
        # not likely needed, it makes things work as expected from the surface.
        self.target.clear_cache()

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

    def _convert_to_photon(
        self: hint.Self,
        container: hint.LezargusContainerArithmetic,
    ) -> hint.LezargusContainerArithmetic:
        """Convert Lezargus spectral flux density to photon flux density.

        The core implementation can be found in
        py:mod:`lezargus.simulator.target._convert_to_photon`.

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
        photon_container = self.target._convert_to_photon(  # noqa: SLF001 # pylint: disable=W0212
            container=container,
        )
        return photon_container

    @property
    def at_observed(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_observed` instance.

        See py:meth:`lezargus.simulator.target.TargetSimulator.at_observed()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_observed

    @property
    def at_telescope(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after telescope area integration.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the area of the telescope.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_observed

        # Multiplying by the area of the telescope.
        telescope_area = self.telescope.telescope_area
        current_state = previous_state * telescope_area

        return current_state

    @property
    def at_primary_reflectivity(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after primary mirror reflectivity.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the primary mirror reflectivity.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_telescope

        # We need to obtain the reflectivity.
        primary_reflectivity = self.telescope.primary_reflectivity_spectrum(
            wavelength=previous_state.wavelength,
        )
        # And, broadcasting the reflectivity spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        primary_reflectivity_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=primary_reflectivity,
                shape=broadcast_shape,
                location="full",
            )
        )
        current_state = previous_state * primary_reflectivity_broadcast

        return current_state

    @property
    def at_primary_emission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after primary mirror emission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the primary mirror emission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_primary_reflectivity

        # We need to obtain the emission.
        solid_angle = 0
        primary_emission = self.telescope.primary_emission_spectrum(
            wavelength=previous_state.wavelength,
            solid_angle=solid_angle,
        )
        # We want this emission in photon counting form.
        primary_photon_emission = self._convert_to_photon(
            container=primary_emission,
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        primary_photon_emission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=primary_photon_emission,
                shape=broadcast_shape,
                location="full",
            )
        )
        # The integrated primary emission spectrum is calculated as the entire
        # area, and we assume that each pixel has an equal contribution.
        n_pixels = np.prod(broadcast_shape)
        primary_photon_emission_pixel = (
            primary_photon_emission_broadcast / n_pixels
        )

        current_state = previous_state + primary_photon_emission_pixel

        return current_state

    @property
    def at_secondary_reflectivity(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after secondary mirror reflectivity.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the secondary mirror reflectivity.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_primary_emission

        # We need to obtain the reflectivity.
        secondary_reflectivity = self.telescope.secondary_reflectivity_spectrum(
            wavelength=previous_state.wavelength,
        )
        # And, broadcasting the reflectivity spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        secondary_reflectivity_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=secondary_reflectivity,
                shape=broadcast_shape,
                location="full",
            )
        )
        current_state = previous_state * secondary_reflectivity_broadcast

        return current_state

    @property
    def at_secondary_emission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after secondary mirror emission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the secondary mirror emission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_secondary_reflectivity

        # We need to obtain the emission.
        solid_angle = 0
        secondary_emission = self.telescope.secondary_emission_spectrum(
            wavelength=previous_state.wavelength,
            solid_angle=solid_angle,
        )
        # We want this emission in photon counting form.
        secondary_photon_emission = self._convert_to_photon(
            container=secondary_emission,
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        secondary_photon_emission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=secondary_photon_emission,
                shape=broadcast_shape,
                location="full",
            )
        )
        # The integrated secondary emission spectrum is calculated as the
        # entire area, and we assume that each pixel has an equal contribution.
        n_pixels = np.prod(broadcast_shape)
        secondary_photon_emission_pixel = (
            secondary_photon_emission_broadcast / n_pixels
        )

        current_state = previous_state + secondary_photon_emission_pixel

        return current_state

    @property
    def at_window_transmission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after entrance window transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the secondary mirror emission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_secondary_emission

        # We get the window transmission spectrum.
        window_transmission = lezargus.data.EFFICIENCY_SPECTRE_WINDOW
        window_transmission_spectrum = window_transmission.interpolate_spectrum(
            wavelength=previous_state.wavelength,
        )
        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        window_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=window_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )
        current_state = previous_state * window_transmission_broadcast

        return current_state

    @property
    def at_window_emission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after entrance window emission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the entrance window emission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_window_transmission

        # We need to obtain the window emission. The basic parameters which
        # are needed.
        window_temperature = 273
        science_beam_diameter = 1.0
        solid_angle = 1.0
        # We assume a blackbody emission function.
        window_blackbody = lezargus.library.wrapper.blackbody_function(
            temperature=window_temperature,
        )
        window_blackbody_radiance = window_blackbody(previous_state.wavelength)

        # The blackbody is modulated by...
        # ...the window's own transmission,
        window_transmission = lezargus.data.EFFICIENCY_SPECTRE_WINDOW
        window_transmission_spectrum = window_transmission.interpolate_spectrum(
            wavelength=previous_state.wavelength,
        )
        emission_efficiency = 1 - window_transmission_spectrum
        # ...the area of the window, more specifically, the area of the science
        # beam,
        window_area = (np.pi / 4) * science_beam_diameter**2
        # ...and the integrating solid angle. Though, this is custom provided.
        solid_angle = float(solid_angle)

        # Performing the "integration" of the blackbody.
        window_emission = (emission_efficiency * window_blackbody_radiance) * (
            window_area * solid_angle
        )

        # We want this emission in photon counting form.
        window_photon_emission = self._convert_to_photon(
            container=window_emission,
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        window_photon_emission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=window_photon_emission,
                shape=broadcast_shape,
                location="full",
            )
        )
        # The integrated window emission spectrum is calculated as the
        # entire area, and we assume that each pixel has an equal contribution.
        n_pixels = np.prod(broadcast_shape)
        window_photon_emission_pixel = (
            window_photon_emission_broadcast / n_pixels
        )

        current_state = previous_state + window_photon_emission_pixel

        return current_state

    @property
    def at_window_ghost(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after the entrance window ghost.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the entrance window ghost.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_window_emission

        # Skipping for now.
        logging.error(
            error_type=logging.ToDoError,
            message="SPECTRE Window ghost to be done.",
        )
        current_state = previous_state

        return current_state

    @property
    def at_foreoptics(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after fore-optics.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the fore-optics.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_window_ghost

        # We get the transmission (or reflectivity) spectra for the collimator
        # and camera mirrors of the fore-optics.
        collimator_transmission = lezargus.data.EFFICIENCY_SPECTRE_COLLIMATOR
        collimator_transmission_spectrum = (
            collimator_transmission.interpolate_spectrum(
                wavelength=previous_state.wavelength,
            )
        )
        camera_transmission = lezargus.data.EFFICIENCY_SPECTRE_CAMERA
        camera_transmission_spectrum = camera_transmission.interpolate_spectrum(
            wavelength=previous_state.wavelength,
        )
        foreoptics_transmission_spectrum = (
            collimator_transmission_spectrum * camera_transmission_spectrum
        )
        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        foreoptics_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=foreoptics_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )
        current_state = previous_state * foreoptics_transmission_broadcast

        return current_state

    @property
    def at_ifu_transmission(self: hint.Self) -> hint.LezargusCube:
        """State of simulation after the transmission of the IFU.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the IFU transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_foreoptics

        # We get the transmission (or reflectivity) spectra for the image
        # slicer portion of the IFU.
        slicer_transmission = lezargus.data.EFFICIENCY_SPECTRE_IMAGE_SLICER
        slicer_transmission_spectrum = slicer_transmission.interpolate_spectrum(
            wavelength=previous_state.wavelength,
        )
        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        broadcast_shape = previous_state.data.shape
        slicer_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=slicer_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )
        current_state = previous_state * slicer_transmission_broadcast

        return current_state

    @property
    def at_ifu_image_slicer(self: hint.Self) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the IFU image slicer.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the IFU image slicer.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_ifu_transmission

        # For convenience, we just wrap array slicing so we only need to
        # supply a slice number.
        def image_slice_array(
            array: hint.NDArray,
            slice_index: int,
        ) -> hint.NDArray:
            """Slice an image array based on the slice index.

            Parameters
            ----------
            array : NDArray
                The array which we are going to slice, should be the 2D array
                of data, uncertainties, or something similar.
            slice_index : int
                The slice which we are slicing out. We follow the general
                slice numbering convention.

            Returns
            -------
            sliced_array : NDArray
                The sliced portion of the array, sliced for the given slice.

            """
            # Basic information of the current situation.
            fov_arcsec = 7.2
            slice_count = 36
            pixel_scale = previous_state.pixel_scale
            slice_scale = previous_state.slice_scale
            # We need to make sure there is actually a provided pixel scale
            # and slice scale, else we cannot assume the size of the current
            # array.
            if pixel_scale is None:
                logging.error(
                    error_type=logging.InputError,
                    message="Pixel scale is None, needs to be provided.",
                )
                pixel_scale = np.nan
            if slice_scale is None:
                logging.error(
                    error_type=logging.InputError,
                    message="Slice scale is None, needs to be provided.",
                )
                slice_scale = np.nan
            # We determine the pixel size of the cropped array, given that
            # we are determining the crop based on array slicing.
            # Pixels...
            pixel_modulo, __ = lezargus.library.math.modulo(
                numerator=fov_arcsec,
                denominator=pixel_scale,
            )
            if not np.isclose(pixel_modulo, 0):
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Non-integer pixel edge length"
                        f" {fov_arcsec / pixel_scale}, based on pixel scale at"
                        " image slicer stop; overcropping."
                    ),
                )
            max_pixel_dim = int(np.floor(fov_arcsec / pixel_scale))
            # Slices...
            slice_modulo, __ = lezargus.library.math.modulo(
                numerator=fov_arcsec,
                denominator=slice_scale,
            )
            if not np.isclose(slice_modulo, 0):
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Non-integer slice edge length"
                        f" {fov_arcsec / slice_scale}, based on slice scale at"
                        " image slicer stop; overcropping."
                    ),
                )
            max_slice_dim = int(np.floor(fov_arcsec / slice_scale))

            # We assume the center of the array is the center of the crop.
            crop_array = lezargus.library.transform.crop_2d(
                array=array,
                new_shape=(max_pixel_dim, max_slice_dim),
                location="center",
            )

            # Now we just slice the array, based on the number of array pixel
            # elements per slice and indexing it.
            crop_modulo, __ = lezargus.library.math.modulo(
                numerator=crop_array.shape[1],
                denominator=slice_count,
            )
            if not np.isclose(crop_modulo, 0):
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Non-integer number of array pixels"
                        f" {crop_array.shape[1] / slice_count} within a single"
                        " slice, underslicing."
                    ),
                )
            pixel_per_slice = int(crop_array.shape[1] / slice_count)

            # Slicing the array based on the slice index.
            sliced_array = crop_array[
                :,
                pixel_per_slice
                * slice_index : pixel_per_slice
                * (slice_index + 1),
            ]
            return sliced_array

        # We create sub-cubes for each of the slices.
        slice_count = 36
        slice_cube_list = []
        for slicedex in range(slice_count):
            # Slicing the important parts of the cube.
            data_slice = image_slice_array(
                array=previous_state.data,
                slice_index=slicedex,
            )
            uncertainty_slice = image_slice_array(
                array=previous_state.uncertainty,
                slice_index=slicedex,
            )
            mask_slice = image_slice_array(
                array=previous_state.mask,
                slice_index=slicedex,
            )
            flag_slice = image_slice_array(
                array=previous_state.flags,
                slice_index=slicedex,
            )
            # We copy over all of the other parts of the data that are not
            # sliced.
            wavelength = previous_state.wavelength
            wavelength_unit = previous_state.wavelength_unit
            data_unit = previous_state.data_unit
            spectral_scale = previous_state.spectral_scale
            header = previous_state.header
            # The cropping of the image by the image slicer (acting as the
            # stop) does not change the spatial resolution. However, if
            # the input arrays do not have evenly divisible shapes, this can
            # lead to problematic array shapes and data loss. The image slice
            # array warns about it.
            pixel_scale = previous_state.pixel_scale
            slice_scale = previous_state.slice_scale
            # We construct the new data cube that represents this slice.
            slice_cube = lezargus.library.container.LezargusCube(
                wavelength=wavelength,
                data=data_slice,
                uncertainty=uncertainty_slice,
                wavelength_unit=wavelength_unit,
                data_unit=data_unit,
                spectral_scale=spectral_scale,
                pixel_scale=pixel_scale,
                slice_scale=slice_scale,
                mask=mask_slice,
                flags=flag_slice,
                header=header,
            )
            slice_cube_list.append(slice_cube)

        # All done.
        current_state = tuple(slice_cube_list)
        return current_state

    @property
    def at_ifu_pupil_mirror_transmission(
        self: hint.Self,
    ) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the IFU pupil mirror transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the IFU pupil mirror transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_ifu_image_slicer

        # We get the transmission (or reflectivity) spectra pupil mirrors.
        # For now, we assume a single transmission spectra.
        pupil_mirror_transmission = (
            lezargus.data.EFFICIENCY_SPECTRE_PUPIL_MIRROR
        )
        pupil_mirror_transmission_spectrum = (
            pupil_mirror_transmission.interpolate_spectrum(
                wavelength=previous_state[0].wavelength,
            )
        )
        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        pupil_mirror_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=pupil_mirror_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = slicedex *  pupil_mirror_transmission_broadcast

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_ifu_diffraction(self: hint.Self) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the IFU image slicer diffraction.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the IFU image slicer diffraction.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_ifu_pupil_mirror_transmission

        # Skipping for now.
        logging.error(
            error_type=logging.ToDoError,
            message="SPECTRE IFU image slicer diffraction needs to be done.",
        )
        current_state = previous_state

        return current_state

    @property
    def at_dichroic(self: hint.Self) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the channel dichroic transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the channel dichroic transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_ifu_diffraction

        # We get the transmission (or reflectivity) spectra of the channel
        # dichroic, which of course depends on the channel.
        if self.channel == "visible":
            dichroic_transmission = (
                lezargus.data.EFFICIENCY_SPECTRE_DICHROIC_VISIBLE
            )
        elif self.channel == "nearir":
            dichroic_transmission = (
                lezargus.data.EFFICIENCY_SPECTRE_DICHROIC_NEARIR
            )
        elif self.channel == "midir":
            dichroic_transmission = (
                lezargus.data.EFFICIENCY_SPECTRE_DICHROIC_MIDIR
            )
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel name is {self.channel}, which is not a supported"
                    " channel."
                ),
            )
        # Interpolating it to the wavelength grid.
        dichroic_transmission_spectrum = (
            dichroic_transmission.interpolate_spectrum(
                wavelength=previous_state[0].wavelength,
            )
        )
        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        dichroic_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=dichroic_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = (
                slicedex * dichroic_transmission_broadcast
            )

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_relay_mirrors(self: hint.Self) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the channel relay mirrors transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the channel relay mirrors transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_dichroic

        # We get the transmission (or reflectivity) spectra of the channel
        # dichroic, which of course depends on the channel.
        if self.channel == "visible":
            relay_transmission = lezargus.data.EFFICIENCY_SPECTRE_RELAY_VISIBLE
        elif self.channel == "nearir":
            relay_transmission = lezargus.data.EFFICIENCY_SPECTRE_RELAY_NEARIR
        elif self.channel == "midir":
            relay_transmission = lezargus.data.EFFICIENCY_SPECTRE_RELAY_MIDIR
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel name is {self.channel}, which is not a supported"
                    " channel."
                ),
            )
        # There are three relay mirrors. The transmission curve for all three
        # are the same.
        trirelay_transmission = relay_transmission**3
        # Interpolating it to the wavelength grid.
        trirelay_transmission_spectrum = (
            trirelay_transmission.interpolate_spectrum(
                wavelength=previous_state[0].wavelength,
            )
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        trirelay_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=trirelay_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = (
                slicedex * trirelay_transmission_broadcast
            )

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_prisms_transmission(
        self: hint.Self,
    ) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the channel prisms transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the prisms transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_relay_mirrors

        # We get the transmission (or reflectivity) spectra of the channel
        # prisms. They are sometimes different materials.
        if self.channel == "visible":
            bk7_transmission = lezargus.data.EFFICIENCY_SPECTRE_PRISM_BK7
            prism_transmission = bk7_transmission * bk7_transmission
        elif self.channel == "nearir":
            silica_transmission = lezargus.data.EFFICIENCY_SPECTRE_PRISM_SILICA
            znse_transmission = lezargus.data.EFFICIENCY_SPECTRE_PRISM_ZNSE
            prism_transmission = silica_transmission * znse_transmission
        elif self.channel == "midir":
            sapphire_transmission = (
                lezargus.data.EFFICIENCY_SPECTRE_PRISM_SAPPHIRE
            )
            prism_transmission = sapphire_transmission * sapphire_transmission
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel name is {self.channel}, which is not a supported"
                    " channel."
                ),
            )
        # The prisms are in double pass, so the transmission function is
        # applied twice.
        double_prism_transmission = prism_transmission**2

        # Interpolating it to the wavelength grid.
        double_prism_transmission_spectrum = (
            double_prism_transmission.interpolate_spectrum(
                wavelength=previous_state[0].wavelength,
            )
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        double_prism_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=double_prism_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = (
                slicedex * double_prism_transmission_broadcast
            )

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_fold_mirror(self: hint.Self) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the channel fold mirror transmission.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the channel fold mirror transmission.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_prisms_transmission

        # We get the transmission (or reflectivity) spectra of the channel
        # fold mirror, which of course depends on the channel.
        if self.channel == "visible":
            fold_transmission = lezargus.data.EFFICIENCY_SPECTRE_FOLD_VISIBLE
        elif self.channel == "nearir":
            fold_transmission = lezargus.data.EFFICIENCY_SPECTRE_FOLD_NEARIR
        elif self.channel == "midir":
            fold_transmission = lezargus.data.EFFICIENCY_SPECTRE_FOLD_MIDIR
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel name is {self.channel}, which is not a supported"
                    " channel."
                ),
            )

        # Interpolating it to the wavelength grid.
        fold_transmission_spectrum = fold_transmission.interpolate_spectrum(
            wavelength=previous_state[0].wavelength,
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        fold_transmission_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=fold_transmission_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = (
                slicedex * fold_transmission_broadcast
            )

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_detector_quantum_efficiency(
        self: hint.Self,
    ) -> tuple[hint.LezargusCube, ...]:
        """State of simulation after the detector quantum efficiency.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusCube
            The state of the simulation after applying the effects of
            the detector quantum efficiency.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_fold_mirror

        # We get the quantum efficiency profile of the channel's detector.
        if self.channel == "visible":
            detector_efficiency = lezargus.data.EFFICIENCY_SPECTRE_CCD_VISIBLE
        elif self.channel == "nearir":
            detector_efficiency = lezargus.data.EFFICIENCY_SPECTRE_H2RG_NEARIR
        elif self.channel == "midir":
            detector_efficiency = lezargus.data.EFFICIENCY_SPECTRE_H2RG_MIDIR
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel name is {self.channel}, which is not a supported"
                    " channel."
                ),
            )

        # Interpolating it to the wavelength grid.
        detector_efficiency_spectrum = detector_efficiency.interpolate_spectrum(
            wavelength=previous_state[0].wavelength,
        )

        # And, broadcasting the emission spectra to allow us to apply it
        # to the previous state.
        # The previous state is a tuple of the slice cubes, we just need one
        # for the sake of defining the shape of the broadcast.
        broadcast_shape = previous_state[0].data.shape
        detector_efficiency_broadcast = (
            lezargus.library.container.functionality.broadcast_spectrum_to_cube(
                input_spectrum=detector_efficiency_spectrum,
                shape=broadcast_shape,
                location="full",
            )
        )

        # We then apply the transmission function to each of the slices.
        new_state = list(previous_state)
        for index, slicedex in enumerate(previous_state):
            new_state[index] = (
                slicedex * detector_efficiency_broadcast
            )

        # All done.
        current_state = tuple(new_state)
        return current_state

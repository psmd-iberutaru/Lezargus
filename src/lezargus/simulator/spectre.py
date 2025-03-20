"""Simulation code to simulate a SPECTRE observation.

This code simulates a SPECTRE observation, from the target on the sky, all
the way to the detector. We use other smaller simulators to simulate common
things (like the object itself or the atmosphere) and the specifics of their
implementations can be found there. Implementation specifics to the SPECTRE
instrument itself are found here.

By self-imposed convention, the attributes are generally named as `at_[stage]`
where the result is the simulated result right after simulating whichever
stage is named.

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

import astropy.units
import numpy as np
import scipy.stats

import lezargus
from lezargus.library import logging


class SpectreSimulator:  # pylint: disable=too-many-public-methods
    """Main SPECTRE simulator class.

    By self-imposed convention, the attributes are generally named as
    `at_[stage]` where the result is the simulated result right after
    simulating whichever stage is named.
    """

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

    exposure_time: float
    """The exposure time of the integration, used near the end for Poisson
    statistics on the observation."""

    coadds: int
    """Co-adds are a way to combine multiple exposures into a single file,
    giving better signal to noise without saturating."""

    use_cache = True
    """If True, we cache calculated values so that they do not need to
    be calculated every time when not needed. If False, caches are never
    returned and instead everything is always recomputed."""

    def __init__(
        self: SpectreSimulator,
        target: hint.TargetSimulator,
        telescope: hint.IrtfTelescopeSimulator,
        channel: str,
        exposure_time: float,
        coadds: int,
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
        exposure_time : float
            The exposure time of the observation integration, in seconds.
        coadds : int
            The number of co-adds of the provided exposure time for the
            observation.
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

        # Other miscellaneous values.
        self.coadds = coadds
        self.exposure_time = exposure_time

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
    def from_advanced_parameters(
        cls: type[hint.Self],
        channel: str,
        wavelength: hint.NDArray,
        blackbody_temperature: float,
        magnitude: float,
        photometric_filter: (
            hint.PhotometricABFilter | hint.PhotometricVegaFilter
        ),
        exposure_time: float,
        coadds: int,
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
        exposure_time : float
            The exposure time of the observation integration, in seconds.
        coadds : int
            The number of co-adds of the provided exposure time for the
            observation.
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
            exposure_time=exposure_time,
            coadds=coadds,
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
    def at_target_spectrum(self: hint.Self) -> hint.LezargusCube | None:
        """Exposing the target's `at_target_spectrum` instance.

        See
        py:meth:`lezargus.simulator.target.TargetSimulator.at_target_spectrum()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_target_spectrum

    @property
    def at_target(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_target` instance.

        See py:meth:`lezargus.simulator.target.TargetSimulator.at_target()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_target

    @property
    def at_target_photon(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_target_photon` instance.

        See
        py:meth:`lezargus.simulator.target.TargetSimulator.at_target_photon()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_target_photon

    @property
    def at_transmission(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_transmission` instance.

        See
        py:meth:`lezargus.simulator.target.TargetSimulator.at_transmission()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_transmission

    @property
    def at_radiance(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_radiance` instance.

        See py:meth:`lezargus.simulator.target.TargetSimulator.at_radiance()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_radiance

    @property
    def at_seeing(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_seeing` instance.

        See py:meth:`lezargus.simulator.target.TargetSimulator.at_seeing()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_seeing

    @property
    def at_refraction(self: hint.Self) -> hint.LezargusCube:
        """Exposing the target's `at_refraction` instance.

        See py:meth:`lezargus.simulator.target.TargetSimulator.at_refraction()`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.target.at_refraction

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
        common_wavelength = previous_state.wavelength
        # We assume a blackbody emission function.
        window_blackbody = lezargus.library.wrapper.blackbody_function(
            temperature=window_temperature,
        )
        window_blackbody_radiance = window_blackbody(common_wavelength)

        # The blackbody is modulated by...
        # ...the window's own transmission,
        window_transmission = lezargus.data.EFFICIENCY_SPECTRE_WINDOW
        window_transmission_data, __, __, __ = window_transmission.interpolate(
            wavelength=common_wavelength,
            extrapolate=False,
        )
        emission_efficiency = 1 - window_transmission_data
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
        window_emission_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=common_wavelength,
            data=window_emission,
            uncertainty=0,
            wavelength_unit=previous_state.wavelength_unit,
            data_unit="W m^-2 m^-1",
        )
        window_photon_emission = self._convert_to_photon(
            container=window_emission_spectrum,
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
                The array which we are going to slice, should be the 3D array
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

            # The field of view ought to be in radians per the SI convention.
            fov_radian = lezargus.library.conversion.convert_units(
                value=fov_arcsec,
                value_unit="arcsec",
                result_unit="radian",
            )

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
                numerator=fov_radian,
                denominator=pixel_scale,
            )
            if not np.isclose(pixel_modulo, 0):
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Non-integer pixel edge length"
                        f" {fov_radian / pixel_scale}, based on pixel scale at"
                        " image slicer stop; overcropping."
                    ),
                )
            max_pixel_dim = int(np.floor(fov_radian / pixel_scale))
            # Slices...
            slice_modulo, __ = lezargus.library.math.modulo(
                numerator=fov_radian,
                denominator=slice_scale,
            )
            if not np.isclose(slice_modulo, 0):
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Non-integer slice edge length"
                        f" {fov_radian / slice_scale}, based on slice scale at"
                        " image slicer stop; overcropping."
                    ),
                )
            max_slice_dim = int(np.floor(fov_radian / slice_scale))

            # We assume the center of the array is the center of the crop.
            crop_array = lezargus.library.transform.crop_3d(
                array=array,
                new_shape=(
                    max_pixel_dim,
                    max_slice_dim,
                    len(previous_state.wavelength),
                ),
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
                pixel_per_slice
                * slice_index : pixel_per_slice
                * (slice_index + 1),
                :,
                :,
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
            new_state[index] = slicedex * pupil_mirror_transmission_broadcast

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
            new_state[index] = slicedex * dichroic_transmission_broadcast

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
            new_state[index] = slicedex * trirelay_transmission_broadcast

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
            new_state[index] = slicedex * double_prism_transmission_broadcast

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
            new_state[index] = slicedex * fold_transmission_broadcast

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
            new_state[index] = slicedex * detector_efficiency_broadcast

        # All done.
        current_state = tuple(new_state)
        return current_state

    @property
    def at_spectral_dispersion(self: hint.Self) -> hint.LezargusImage:
        """State of simulation after modeling spectral dispersion on detector.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after applying the effects of
            spectral dispersion onto the detector.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_quantum_efficiency

        # We need to apply all of the slice images to the same detector image.
        # However, it must have the same dimensions of the channel's detector
        # and we assume they are square.
        if self.channel == "visible":
            detector_size = int(lezargus.data.CONST_VISIBLE_DETECTOR_SIZE)
        elif self.channel == "nearif":
            detector_size = int(lezargus.data.CONST_NEARIR_DETECTOR_SIZE)
        elif self.channel == "midir":
            detector_size = int(lezargus.data.CONST_MIDIR_DETECTOR_SIZE)
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # We need to determine the binning factor. We bin the data after
        # it has been dispersed, simulating pixels. We assume all pixel scales
        # of the slices are the same.
        expected_pixel_scale = 0.1 / 206265
        actual_pixel_scale = previous_state[0].pixel_scale
        if actual_pixel_scale is None:
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "Simulated images need a pixel scale, this one does not."
                ),
            )
        else:
            bin_factor_modulo, __ = lezargus.library.math.modulo(
                numerator=expected_pixel_scale,
                denominator=actual_pixel_scale,
            )
        if not np.isclose(bin_factor_modulo, 0):
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Binning factor is non-integer due to mismatch in"
                    " detector shapes."
                ),
            )
        bin_factor = int(expected_pixel_scale / actual_pixel_scale)

        # The detector image. We scale it up based on the binning factor
        # to match the current slices, we bin it down later.
        detector_shape = (
            detector_size * bin_factor,
            detector_size * bin_factor,
        )
        detector_data = np.zeros(detector_shape)

        # Going over every single slice, we need to put it on the detector.
        for sliceindex, slicedex in enumerate(previous_state):
            # Each wavelength slice of the cubes is integrated over their
            # wavelength spread. We compute it here and extend it for the
            # last wavelength point.
            wavelength_delta = (
                slicedex.wavelength[1:] - slicedex.wavelength[:-1]
            )
            wavelength_delta = np.append(wavelength_delta, wavelength_delta[-1])

            # We need to "integrate" the slice observation's monochromatic
            # image, so we loop over all of them.
            for index, (wavedex, wdeldex) in enumerate(
                zip(
                    slicedex.wavelength,
                    wavelength_delta,
                    strict=True,
                ),
            ):
                # The image data at this wavelength.
                datadex = slicedex.data[:, :, index]
                # We also integrate over the wavelength range.
                integrated_data = datadex * wdeldex

                # We pad the data to have it the same size, shape, and scale
                # of the detector data. We track the corners of the data
                # so we can transform the data per dispersion.
                padded_data = np.zeros(detector_shape)
                full_height, full_width = detector_shape
                # Manually placing it on the detector.
                slice_height, slice_width = integrated_data.shape
                base_bottom_edge = full_height // 2 - slice_height // 2
                base_top_edge = base_bottom_edge + slice_height
                base_left_edge = full_width // 2 - slice_width // 2
                base_right_edge = base_left_edge + slice_width
                # Placing the detector data on the array.
                padded_data[
                    base_bottom_edge:base_top_edge,
                    base_left_edge:base_right_edge,
                ] = integrated_data

                # The defining four base coordinates.
                base_bottom_left = (base_left_edge, base_bottom_edge)
                base_bottom_right = (base_right_edge, base_bottom_edge)
                base_top_left = (base_left_edge, base_top_edge)
                base_top_right = (base_right_edge, base_top_edge)

                # We determine where this data should be put on the array.
                new_bottom_left = (
                    lezargus.data.DISPERSION_SPECTRE.get_slice_dispersion_pixel(
                        channel=self.channel,
                        slice_=sliceindex + 1,
                        location="bottom_left",
                        wavelength=wavedex,
                    )
                )
                new_bottom_right = (
                    lezargus.data.DISPERSION_SPECTRE.get_slice_dispersion_pixel(
                        channel=self.channel,
                        slice_=sliceindex + 1,
                        location="bottom_right",
                        wavelength=wavedex,
                    )
                )
                new_top_left = (
                    lezargus.data.DISPERSION_SPECTRE.get_slice_dispersion_pixel(
                        channel=self.channel,
                        slice_=sliceindex + 1,
                        location="top_left",
                        wavelength=wavedex,
                    )
                )
                new_top_right = (
                    lezargus.data.DISPERSION_SPECTRE.get_slice_dispersion_pixel(
                        channel=self.channel,
                        slice_=sliceindex + 1,
                        location="top_right",
                        wavelength=wavedex,
                    )
                )

                # Given the coordinates, we compute the transformation needed
                # to move it to the right location.
                base_coord = np.array(
                    [
                        base_bottom_left,
                        base_bottom_right,
                        base_top_left,
                        base_top_right,
                    ],
                )
                # The needed coordinates are nested so we need to bring it out.
                new_coord = np.array(
                    [
                        *new_bottom_left,
                        *new_bottom_right,
                        *new_top_left,
                        *new_top_right,
                    ],
                )
                # We need to adapt the coordinates to the increased binning
                # scale of the current simualted data.
                new_scaled_coord = np.array(new_coord) * bin_factor

                # Calculating the transform matrix given our points.
                affine_transform_matrix = (
                    lezargus.library.transform.calculate_affine_matrix(
                        in_points=base_coord,
                        out_points=new_scaled_coord,
                    )
                )

                # And we actually do the transformation; we are using
                # homogenous coordinates as the typical output of the
                # calculation.
                dispersed_data = lezargus.library.transform.affine_transform(
                    array=padded_data,
                    matrix=affine_transform_matrix,
                    offset=None,
                    constant=0,
                )

                # We are ignoring masks and flags for now.

                # Putting it on the detector.
                detector_data = detector_data + dispersed_data

        # Finally, we bin the resulting data down to the expected size.
        # Binning the current slice to match the resolution of the
        # detector.
        binned_detector_data = lezargus.library.array.bin_image_array(
            image=detector_data,
            x_bin=bin_factor,
            y_bin=bin_factor,
            mode="add",
        )

        # We now just assemble the LezargusImage from the data we have.
        detector_image = lezargus.library.container.LezargusImage(
            data=binned_detector_data,
            uncertainty=0,
            wavelength=None,
            wavelength_unit=None,
            data_unit="photon",
            spectral_scale=None,
            pixel_scale=previous_state[0].pixel_scale * bin_factor,
            slice_scale=previous_state[0].slice_scale * bin_factor,
            mask=None,
            flags=None,
            header=previous_state[0].header,
        )
        current_state = detector_image

        return current_state

    @property
    def at_scattered_light(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after modeling scattered light.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after adding scattered light
            interfearence.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_spectral_dispersion

        # We are just skipping it for now.
        logging.error(
            error_type=logging.ToDoError,
            message="Scattered light to be implemented.",
        )
        current_state = previous_state

        # All done.
        return current_state

    @property
    def at_photon_poisson_noise(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after accounting for photon poisson noise.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after adding photon noise.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_scattered_light

        # Integrating the observation via time.
        integrated_data, integrated_uncertainty = (
            lezargus.library.math.multiply(
                multiplier=previous_state.data,
                multiplicand=self.exposure_time,
                multiplier_uncertainty=previous_state.uncertainty,
                multiplicand_uncertainty=0,
            )
        )
        # Units.
        integrated_unit = previous_state.data_unit * astropy.units.Unit("s")

        # We apply photon poisson statstics on the expected value: integrrated
        # flux values.
        poisson_data = scipy.stats.poisson.rvs(integrated_data)
        poisson_uncertainty = integrated_uncertainty
        # No change needed.
        poisson_unit = integrated_unit

        # Rebuilding the image from the new data.
        current_state = copy.deepcopy(previous_state)
        current_state.data = np.asarray(poisson_data)
        current_state.uncertainty = np.asarray(poisson_uncertainty)
        current_state.data_unit = poisson_unit

        # All done.
        return current_state

    @property
    def at_cosmic_rays(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after accounting for cosmic ray hits.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after adding cosmic ray hits.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_photon_poisson_noise

        # If a cosmic ray hits, it gets typically a hot value.
        cosmic_ray_value = lezargus.data.CONST_COSMIC_RAY_VALUE

        # We need to determine which pixels have a cosmic ray.
        # Starting with the expected number of cosmic rays per pixel,
        # and the area of each pixel. The expected count is likely much
        # less than 1, but non-zero.
        if self.channel == "visible":
            pixel_size = lezargus.data.CONST_VISIBLE_PIXEL_SIZE
        elif self.channel == "nearir":
            pixel_size = lezargus.data.CONST_NEARIR_PIXEL_SIZE
        elif self.channel == "midir":
            pixel_size = lezargus.data.CONST_MIDIR_PIXEL_SIZE
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )
        pixel_area = pixel_size**2
        expected_rate = lezargus.data.CONST_COSMIC_RAY_RATE
        expected_count = expected_rate * pixel_area * self.exposure_time

        # We use Poisson statistics to generate the prediction, and have it
        # as a sort of mask.
        # Generally speaking, cosmic rays also impart a huge uncertainty but
        # cosmic rays themselves don't have any intrinisic uncertainty.
        cosmic_ray_mask = np.array(
            scipy.stats.poisson.rvs(
                expected_count,
                size=previous_state.data.shape,
            ),
            dtype=int,
        )
        cosmic_ray_data = cosmic_ray_mask * cosmic_ray_value
        cosmic_ray_uncertainty = 0

        # Just adding the cosmic ray information to the current state.
        current_state = copy.deepcopy(previous_state)
        current_state.data = previous_state.data + cosmic_ray_data
        current_state.uncertainty = (
            previous_state.uncertainty + cosmic_ray_uncertainty
        )

        # All done.
        return current_state

    @property
    def at_detector_gain(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after accounting the detector gain.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after acconting for the detetor gain
            factor.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_cosmic_rays

        # We need to determine the detector gain.
        if self.channel == "visible":
            gain_factor = lezargus.data.CONST_VISIBLE_DETECTOR_GAIN
        elif self.channel == "nearir":
            gain_factor = lezargus.data.CONST_NEARIR_DETECTOR_GAIN
        elif self.channel == "midir":
            gain_factor = lezargus.data.CONST_MIDIR_DETECTOR_GAIN
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # Applying the gain.
        current_state = previous_state * gain_factor

        # All done.
        return current_state

    @property
    def at_detector_flat_field(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after accounting the detector flat field.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after acconting for detector flat
            field variations.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_gain

        # We derive the flat field variations from the theorical "perfect"
        # flat. This may also just be a historical flat. Each channel is likely
        # a little different
        if self.channel == "visible":
            perfect_flat = lezargus.data.CONST_VISIBLE_FLAT_FIELD
            flat_stddev = lezargus.data.CONST_VISIBLE_FLAT_FIELD_STDDEV
        elif self.channel == "nearir":
            perfect_flat = lezargus.data.CONST_NEARIR_FLAT_FIELD
            flat_stddev = lezargus.data.CONST_NEARIR_FLAT_FIELD_STDDEV
        elif self.channel == "midir":
            perfect_flat = lezargus.data.CONST_MIDIR_FLAT_FIELD
            flat_stddev = lezargus.data.CONST_MIDIR_FLAT_FIELD_STDDEV
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # We generate a flat field from the perfect flat field, simulating
        # an observation and some variance with regards to it. Applying the
        # variations via a Gaussian/normal distribution should be good enough.
        flat_field = scipy.stats.norm.rvs(loc=perfect_flat, scale=flat_stddev)

        # Applying the flat field.
        current_state = previous_state * flat_field

        # All done.
        return current_state

    @property
    def at_detector_bias(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after adding in the detector bias level.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after adding in the detector bias
            level.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_flat_field

        # The detector bias level (either as a file or for the entire field)
        # is different per detector. This may be a historical bias level as
        # well.
        if self.channel == "visible":
            perfect_bias = lezargus.data.CONST_VISIBLE_BIAS_LEVEL
            bias_stddev = lezargus.data.CONST_VISIBLE_BIAS_LEVEL_STDDEV
        elif self.channel == "nearir":
            perfect_bias = lezargus.data.CONST_NEARIR_BIAS_LEVEL
            bias_stddev = lezargus.data.CONST_NEARIR_BIAS_LEVEL_STDDEV
        elif self.channel == "midir":
            perfect_bias = lezargus.data.CONST_MIDIR_BIAS_LEVEL
            bias_stddev = lezargus.data.CONST_MIDIR_BIAS_LEVEL_STDDEV
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # We generate a the detector bias image, simulating an observation
        # and some variance with regards to it. Applying the
        # variations via a Gaussian/normal distribution should be good enough.
        total_bias_level = np.zeros_like(previous_state.data)
        for __ in range(self.coadds):
            # Every readout (co-add) adds a new bias level to the overall
            # total for the FITS file.
            single_bias_level = scipy.stats.norm.rvs(
                loc=perfect_bias,
                scale=bias_stddev,
            )
            total_bias_level = total_bias_level + single_bias_level

        # Applying the bias.
        current_state = previous_state + total_bias_level

        # All done.
        return current_state

    @property
    def at_detector_dark_current(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after adding in the detector dark current.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after adding in the detector dark
            current

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_bias

        # The detector detector dark current, either as a value or a map.
        # The dark current itself may vary from its actual value. We model
        # such variations using a Gaussian. These values may also be
        # calculated from actual darks.
        if self.channel == "visible":
            perfect_dark_current = lezargus.data.CONST_VISIBLE_DARK_CURRENT
            dark_current_stddev = (
                lezargus.data.CONST_VISIBLE_DARK_CURRENT_STDDEV
            )
        elif self.channel == "nearir":
            perfect_dark_current = lezargus.data.CONST_NEARIR_DARK_CURRENT
            dark_current_stddev = lezargus.data.CONST_NEARIR_DARK_CURRENT_STDDEV
        elif self.channel == "midir":
            perfect_dark_current = lezargus.data.CONST_MIDIR_DARK_CURRENT
            dark_current_stddev = lezargus.data.CONST_MIDIR_DARK_CURRENT_STDDEV
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # We generate a dark current of the image, simulating an observation
        # and some variance with regards to it. Applying the
        # variations via a Gaussian/normal distribution should be good enough.
        dark_current = scipy.stats.norm.rvs(
            loc=perfect_dark_current,
            scale=dark_current_stddev,
        )

        # Applying the dark current.
        current_state = previous_state + dark_current * self.exposure_time

        # All done.
        return current_state

    @property
    def at_detector_linearity(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after the detector linearity functions.

        This function is mostly applying an approximation of the effect,
        see the documentation for more information.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after acounting for the detector
            linearity functions.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_dark_current

        # Detector linearity is approximated by applying the effects of
        # each linearity on each co-add. Though we already combined them
        # together given the dark current and espeically the bias, this is
        # just an approximation.
        if self.channel == "visible":
            linearity = lezargus.data.FUNCTION_SPECTRE_VISIBLE_LINEARITY
        elif self.channel == "nearir":
            linearity = lezargus.data.FUNCTION_SPECTRE_NEARIR_LINEARITY
        elif self.channel == "midir":
            linearity = lezargus.data.FUNCTION_SPECTRE_MIDIR_LINEARITY
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # Calculating the linearity function.
        current_data = self.coadds * linearity(
            data=previous_state.data / self.coadds,
        )
        current_uncertainty = previous_state.uncertainty

        # Applying the new linearity data.
        current_state = copy.deepcopy(previous_state)
        current_state.data = current_data
        current_state.uncertainty = current_uncertainty

        # All done.
        return current_state

    @property
    def at_detector_read_noise(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after the detector read noise.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after acounting for the detector
            read noise.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_linearity

        # Computing read noise as a positive only gaussian distribution.
        def positive_gaussian_generator(
            center: float | hint.NDArray,
            stddev: float | hint.NDArray,
            shape: tuple[int],
        ) -> hint.NDArray:
            """Generate only possitive values with a Gaussian distribution.

            Parameters
            ----------
            center : float | NDArray
                The center of the distribution.
            stddev : float | NDarray
                The stddev of the Gaussian distribution.
            shape : tuple
                The shape of the output of the generated numbers, determines
                how many numbers are being made.

            Returns
            -------
            positive_gaussian_values : NDArray
                The generated numbers.

            """
            # Usually Gaussian distribution.
            gaussian_values = scipy.stats.norm.rvs(center, stddev, size=shape)
            # Positive only.
            positive_gaussian_values = np.where(
                gaussian_values <= 0,
                0,
                gaussian_values,
            )
            return positive_gaussian_values

        # Non-destructive reads reduce the detector read noise, and each of
        # the three detectors have different values.
        if self.channel == "visible":
            perfect_read_noise = lezargus.data.CONST_VISIBLE_READ_NOISE
            read_noise_stddev = lezargus.data.CONST_VISIBLE_READ_NOISE_STDDEV
            ndr = lezargus.data.CONST_VISIBLE_NONDESTRUCTIVE_READS
        elif self.channel == "nearir":
            perfect_read_noise = lezargus.data.CONST_NEARIR_READ_NOISE
            read_noise_stddev = lezargus.data.CONST_NEARIR_READ_NOISE_STDDEV
            ndr = lezargus.data.CONST_NEARIR_NONDESTRUCTIVE_READS
        elif self.channel == "midir":
            perfect_read_noise = lezargus.data.CONST_MIDIR_READ_NOISE
            read_noise_stddev = lezargus.data.CONST_MIDIR_READ_NOISE_STDDEV
            ndr = lezargus.data.CONST_MIDIR_NONDESTRUCTIVE_READS
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # Calculating read noise given the co-adds.
        full_read_noise = np.zeros_like(previous_state.data)
        for __ in range(self.coadds):
            temp_read_noise = positive_gaussian_generator(
                center=perfect_read_noise,
                stddev=read_noise_stddev,
                shape=full_read_noise.shape,
            )
            full_read_noise = full_read_noise + temp_read_noise

        # Applying the NDR factor.
        read_noise = full_read_noise / np.sqrt(ndr)

        # Applying the noise.
        current_state = copy.deepcopy(previous_state)
        current_state.data = previous_state.data + read_noise
        current_state.uncertainty = previous_state.uncertainty

        # All done.
        return current_state

    @property
    def at_detector_hot_dead_pixels(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation after the detector hot and dead pixels.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation after applying hot and dead pixels.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_read_noise

        # We just pull cached hot and dead pixels maps.
        if self.channel == "visible":
            hot_pixel_map = lezargus.data.CONST_VISIBLE_HOT_PIXEL_MAP
            dead_pixel_map = lezargus.data.CONST_VISIBLE_DEAD_PIXEL_MAP
        elif self.channel == "nearir":
            hot_pixel_map = lezargus.data.CONST_NEARIR_HOT_PIXEL_MAP
            dead_pixel_map = lezargus.data.CONST_NEARIR_DEAD_PIXEL_MAP
        elif self.channel == "midir":
            hot_pixel_map = lezargus.data.CONST_MIDIR_HOT_PIXEL_MAP
            dead_pixel_map = lezargus.data.CONST_MIDIR_DEAD_PIXEL_MAP
        else:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Channel {self.channel}, is not one of the available"
                    " three."
                ),
            )

        # And the values.
        hot_pixel_value = lezargus.data.CONST_HOT_PIXEL_VALUE
        dead_pixel_value = lezargus.data.CONST_DEAD_PIXEL_VALUE

        # Applying them.
        current_state = copy.deepcopy(previous_state)
        current_state.data[np.where(hot_pixel_map)] = hot_pixel_value
        current_state.data[np.where(dead_pixel_map)] = dead_pixel_value

        # All done.
        return current_state

    @property
    def at_fits_file(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """State of simulation near the read out of the FITS file.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the simulation ready to read out to a FITS file.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_detector_hot_dead_pixels

        logging.error(
            error_type=logging.ToDoError,
            message=(
                "Fix up the detector so FITS handling is better, if needed."
            ),
        )

        # All done.
        current_state = previous_state
        return current_state

    @property
    def at_readout(
        self: hint.Self,
    ) -> hint.LezargusImage:
        """The state of the simulation at readout, fully simulated.

        This is mostly a convience function, but it is also done as a final
        caching level if needed.

        Parameters
        ----------
        None

        Returns
        -------
        current_state : LezargusImage
            The state of the finished simulation.

        """
        # No cached value, we calculate it from the previous state.
        previous_state = self.at_fits_file
        # All done.
        current_state = previous_state
        return current_state

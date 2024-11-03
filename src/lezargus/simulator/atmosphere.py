"""Simulator class for emulating atmospheric effects and properties.

Any and all simulating functions which the atmosphere deals with is handled
here. Namely, the main four functions are the atmospheric transmission,
radiance, seeing, and diffraction.
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


class AtmosphereSimulator:
    """The atmospheric simulation class.

    Attributes
    ----------
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    ppw : float
        The partial pressure of water in the atmosphere, in Pascals.
    pwv : float
        The precipitable water vapor, in meters.
    seeing : float
        The atmospheric seeing parameter, in radians. Measured at the
        reference zenith angle (0) and the reference wavelength.
    zenith_angle : float
        The zenith angle of the observation, in radians. Namely, the direct
        observable for airmass.
    parallactic_angle : float
        The parallactic angle of the observation, in radians. Used to determine
        the rotations of the properties in the atmosphere.
    reference_wavelength : float
        The reference wavelength which form the basis for the atmospheric
        refraction and the seeing, typically in meters.
    reference_zenith_angle : float
        The reference zenith angle which form the basis for the atmospheric
        refraction and the seeing, in radians. This should always be 0.
    transmission_generator : AtmosphereSpectrumGenerator
        The transmission spectrum generator used to generate the
        specific transmission spectra.
    radiance_generator : AtmosphereSpectrumGenerator
        The transmission spectrum generator used to generate the
        specific transmission spectra.

    """

    def __init__(
        self: AtmosphereSimulator,
        temperature: float,
        pressure: float,
        ppw: float,
        pwv: float,
        seeing: float,
        zenith_angle: float,
        parallactic_angle: float,
        reference_wavelength: float,
        *args: hint.Any,
        transmission_generator: hint.AtmosphereSpectrumGenerator | None = None,
        radiance_generator: hint.AtmosphereSpectrumGenerator | None = None,
    ) -> None:
        """Create the atmospheric simulator, provided atmospheric properties.

        Parameters
        ----------
        temperature : float
            The temperature of the atmosphere, in Kelvin.
        pressure : float
            The pressure of the atmosphere, in Pascals.
        ppw : float
            The partial pressure of water in the atmosphere, in Pascals.
        pwv : float
            The precipitable water vapor, in meters.
        seeing : float
            The atmospheric seeing parameter, in radians.
        zenith_angle : float
            The zenith angle of the observation, in radians.
        parallactic_angle : float
            The parallactic angle of the observation, in radians.
        reference_wavelength : float
            The reference wavelength at which seeing is measured at and where
            relative refraction is 0.
        *args : Any
            A catch to ensure that non-basic atmospheric parameters are to be
            keywords only.
        transmission_generator : AtmosphereSpectrumGenerator, default = None
            The transmission spectrum generator used to generate the
            specific transmission spectra. If invalid, we use the built-in.
        radiance_generator : AtmosphereSpectrumGenerator
            The transmission spectrum generator used to generate the
            specific transmission spectra. If invalid, we use the built-in.

        """
        # We just assign the main primary values, the rest will be calculated.
        self.temperature = temperature
        self.pressure = pressure
        self.ppw = ppw
        self.pwv = pwv
        self.seeing = seeing
        self.zenith_angle = zenith_angle
        self.parallactic_angle = parallactic_angle
        self.reference_wavelength = reference_wavelength
        self.reference_zenith_angle = 0

        # We catch all other atmospheric parameters which are not key-based
        # parameters.
        if len(args) != 0:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Extra (+{len(args)}),atmospheric parameters provided."
                    " Non-basic parameters need to be keyword parameters."
                ),
            )

        # Handling the defaults, if we use the basic generators.
        if not isinstance(
            transmission_generator,
            lezargus.library.container.AtmosphereSpectrumGenerator,
        ):
            transmission_generator = lezargus.data.ATM_TRANS_GEN
        self.transmission_generator = transmission_generator
        if not isinstance(
            radiance_generator,
            lezargus.library.container.AtmosphereSpectrumGenerator,
        ):
            radiance_generator = lezargus.data.ATM_RADIANCE_GEN
        self.radiance_generator = radiance_generator

        # All done.

    @staticmethod
    def _convolve_atmospheric_spectrum(
        spectrum: hint.LezargusSpectrum,
        output_resolution: float | None = None,
        output_resolving: float | None = None,
        reference_wavelength: float | None = None,
        input_resolution: float | None = None,
        input_resolving: float | None = None,
    ) -> hint.LezargusSpectrum:
        """Convolve the input spectrum match its resolution with the output.

        Nominally, we do this to convolve down the transmission and radiance
        spectra to better match the resolution of the spectrum we will be
        applying it to. If they did not match otherwise, it would give
        erroneous results.

        We leverage :py:func:`kernel_1d_gaussian_resolution` to make the kernel.

        Parameters
        ----------
        spectrum : LezargusSpectrum
            The input spectrum which we will be preparing. This spectrum should
            also have its current resolution. A different value is used if
            overridden if an explicit resolving power is input.
        output_resolution : float, default = None
            The spectral resolution of the simulation spectra. Must be in
            the same units as the simulation spectra.
        output_resolving : float, default = None
            The spectral resolving power of the simulation spectra, relative
            to the wavelength `reference_wavelength`.
        reference_wavelength : float, default = None
            The reference wavelength for any needed conversion.
        input_resolution : float, default = None
            The spectral resolution of the input spectra. Must be in
            the same units as the spectra. Overrides any inherent values from
            the input spectrum.
        input_resolving : float, default = None
            The spectral resolving power of the input spectra, relative
            to the wavelength `reference_wavelength`. Overrides any inherent
            values from the input spectrum.
        **kwargs : dict
            Keyword argument catcher.

        Returns
        -------
        convolved_spectra : LezargusSpectrum
            The spectra, after convolution based on the input parameters.

        """
        # We need to determine the input resolution. Namely, if any input
        # values were provided to override the spectrum's.
        if input_resolution is not None or input_resolving is not None:
            using_resolution = input_resolution
            using_resolving = input_resolving
        else:
            using_resolution = spectrum.spectral_scale
            using_resolving = None
        # Double checking that we have a valid input resolution or resolving
        # power.
        if using_resolution is None and using_resolving is None:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "No input resolution/resolving power is found in the"
                    " spectrum or manually provided."
                ),
            )

        # We assume the kernel size based on the wavelength of the input
        # spectra. Namely, the kernel must be smaller than the number of points.
        # We assume that we have Nyquist sampling and 1 extra degree of
        # freedom.
        reduction_factor = 1
        kernel_size = int(np.ceil(len(spectrum.wavelength) / reduction_factor))
        kernel_shape = (kernel_size,)

        # We have the input, we rely on the kernel determination to figure out
        # the mode.
        gaussian_kernel = (
            lezargus.library.convolution.kernel_1d_gaussian_resolution(
                shape=kernel_shape,
                template_wavelength=spectrum.wavelength,
                base_resolution=using_resolution,
                target_resolution=output_resolution,
                base_resolving_power=using_resolving,
                target_resolving_power=output_resolving,
                reference_wavelength=reference_wavelength,
            )
        )

        # We then convolve the input spectra.
        convolved_spectra = spectrum.convolve(kernel=gaussian_kernel)

        # All done.
        return convolved_spectra

    def generate_transmission(
        self: hint.Self,
        template: hint.LezargusContainerArithmetic,
    ) -> hint.LezargusSpectrum:
        """Generate a transmission spectrum applicable to the template.

        This function generates a transmission spectrum with the internal
        atmospheric parameters. The provided template spectrum is the
        spectrum class the transmission function is generated for, so the
        wavelength and resolutions properly match.

        Parameters
        ----------
        template : LezargusContainerArithmetic
            The template container which is used to define the proper format
            of the output spectrum. The template container is not affected.

        Returns
        -------
        transmission_spectrum : LezargusSpectrum
            The transmission spectrum.

        """
        # To ensure we do not touch the template spectrum, we work on a copy.
        template = copy.deepcopy(template)

        # We determine the transmission spectrum from our current atmospheric
        # parameters, using the template wavelength as the basis.
        raw_transmission_spectrum = (
            self.transmission_generator.interpolate_spectrum(
                wavelength=template.wavelength,
                zenith_angle=self.zenith_angle,
                pwv=self.pwv,
            )
        )

        # The transmission spectrum needs to be convolved to properly match
        # the template spectrum's resolution.
        if template.spectral_scale is None:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Template container has no spectral resolution scale, no"
                    " convolution."
                ),
            )
            transmission_spectrum = raw_transmission_spectrum
        else:
            # Convolving, the raw transmission spectrum should have its input
            # spectral resolution.
            template_resolution = template.spectral_scale
            transmission_spectrum = self._convolve_atmospheric_spectrum(
                spectrum=raw_transmission_spectrum,
                output_resolution=template_resolution,
                output_resolving=None,
                reference_wavelength=self.reference_wavelength,
                input_resolution=None,
                input_resolving=None,
            )

        # Sometimes the convolution creates negative transmission values, it
        # should just be zero instead.
        is_zero = transmission_spectrum.data <= 0
        transmission_spectrum.data[is_zero] = 0
        # It is extremely unlikely, but transmission above 1 is not physical.
        is_one = transmission_spectrum.data >= 1
        transmission_spectrum.data[is_one] = 1

        # All done.
        return transmission_spectrum

    def generate_radiance(
        self: hint.Self,
        template: hint.LezargusContainerArithmetic,
    ) -> hint.LezargusSpectrum:
        """Generate a radiance spectrum applicable to the template.

        This function generates a radiance spectrum with the internal
        atmospheric parameters. The provided template container is the
        container class the radiance function is generated for, so the
        wavelength and container properly match.

        Parameters
        ----------
        template : LezargusContainerArithmetic
            The template container which is used to define the proper format
            of the output spectrum. The template container is not affected.

        Returns
        -------
        radiance_spectrum : LezargusSpectrum
            The radiance spectrum.

        """
        # To ensure we do not touch the template container, we work on a copy.
        template = copy.deepcopy(template)

        # We determine the radiance spectrum from our current atmospheric
        # parameters, using the template wavelength as the basis.
        raw_radiance_spectrum = self.radiance_generator.interpolate_spectrum(
            wavelength=template.wavelength,
            zenith_angle=self.zenith_angle,
            pwv=self.pwv,
        )

        # The radiance container needs to be convolved to properly match
        # the template container's resolution.
        if template.spectral_scale is None:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Template container has no spectral resolution scale, no"
                    " convolution."
                ),
            )
            radiance_spectrum = raw_radiance_spectrum
        else:
            # Convolving, the raw radiance container should have its input
            # spectral resolution.
            template_resolution = template.spectral_scale
            radiance_spectrum = self._convolve_atmospheric_spectrum(
                spectrum=raw_radiance_spectrum,
                output_resolution=template_resolution,
                output_resolving=None,
                reference_wavelength=self.reference_wavelength,
                input_resolution=None,
                input_resolving=None,
            )
        # All done.
        return radiance_spectrum

    def seeing_function(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """Seeing function over wavelength for the current atmosphere.

        This function primarily takes the proportionality relationships of
        seeing with regards to zenith angle and wavelength, using the input
        seeing parameter as the base.

        Parameters
        ----------
        wavelength : ndaraay
            The wavelength to evaluate the seeing at. Should be in the same
            unit as the reference wavelength.

        Returns
        -------
        seeing : ndarray
            The computed seeing, based on the ratios.

        """
        # The library function handles the work for us, we just give it our
        # current conditions.
        seeing = lezargus.library.atmosphere.seeing(
            wavelength=wavelength,
            zenith_angle=self.zenith_angle,
            reference_seeing=self.seeing,
            reference_wavelength=self.reference_wavelength,
            reference_zenith_angle=self.reference_zenith_angle,
        )
        return seeing

    def generate_seeing_kernels(
        self: hint.Self,
        template: hint.LezargusContainerArithmetic,
    ) -> hint.NDArray:
        """Generate a seeing kernel, accounting for wavelength variations.

        We create a seeing kernel for convolution. We take into account the
        seeing variations as a function of wavelength when creating the
        seeing kernels. By default, we create a stacked 2D set of kernels
        per the template container's wavelength.

        Parameters
        ----------
        template : LezargusContainerArithmetic
            The template container which is used to define the proper format
            of the output kernels. The template container is not affected.

        Returns
        -------
        seeing_kernels : ndarray
            The seeing kernel(s).

        """
        # To ensure we do not touch the template container, we work on a copy.
        template = copy.deepcopy(template)

        # We determine the seeing.
        seeing = self.seeing_function(wavelength=template.wavelength)
        # The seeing values are in angles, we convert it to pixels via the
        # pixel scales.
        if template.pixel_scale is None or template.slice_scale is None:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "No pixel or slice scale exists on the template, cannot"
                    " determine seeing angle to pixel conversion."
                ),
            )
            pixel_scale = 1
            slice_scale = 1
        else:
            pixel_scale = template.pixel_scale
            slice_scale = template.slice_scale
        pixel_seeing = seeing / pixel_scale
        slice_seeing = seeing / slice_scale

        # We need to determine the size of the kernels. If the template can
        # provide a guide for the spatial shape of the kernels, we use that,
        # else we guess based on the seeing values.
        image_shape_len = 2
        cube_shape_len = 3
        if (
            len(template.data.shape) == image_shape_len
            or len(template.data.shape) == cube_shape_len
        ):
            kernel_shape_guide = template.data.shape[0], template.data.shape[1]
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Template data is neither an image nor cube, cannot"
                    " determine kernel shape."
                ),
            )
            # Approximating it based on the 68-95-99.7 rule; everything ought
            # to be within 5-sigma.
            sigma_multiple = 5
            longest_edge = (
                np.nanmax([pixel_seeing, slice_seeing]) * sigma_multiple
            )
            kernel_shape_guide = longest_edge, longest_edge
        # We also want to make sure the kernel has odd edges, just in case
        # discrete convolution is needed.
        kernel_shape = tuple(
            (valdex + 1 if valdex % 2 == 0 else valdex)
            for valdex in kernel_shape_guide
        )

        # We build the kernel layer by layer.
        seeing_kernels_list = []
        for pix_see_dex, sli_see_dex in zip(
            pixel_seeing,
            slice_seeing,
            strict=True,
        ):
            kernel_layer = lezargus.library.convolution.kernel_2d_gaussian(
                shape=kernel_shape,
                x_stddev=pix_see_dex,
                y_stddev=sli_see_dex,
                rotation=self.parallactic_angle,
            )
            seeing_kernels_list.append(kernel_layer)
        seeing_kernels = np.stack(seeing_kernels_list, axis=-1)

        # All done.
        return seeing_kernels

    def generate_refraction_vectors(
        self: hint.Self,
        template: hint.LezargusContainerArithmetic,
    ) -> hint.NDArray:
        """Generate a set of translation vectors mimicking refraction.

        We create a set of translation vectors which simulate fraction for
        cube-like data. Namely different wavelengths of light are refracted
        differently (different magnitudes), so they apply a sheer-like
        translation transformation across the wavelength axis. The vectors
        generated describe the translation per wavelength.

        Parameters
        ----------
        template : LezargusContainerArithmetic
            The template container which is used to define the proper format
            of the output vectors. The template container is not affected.

        Returns
        -------
        refraction_vectors : ndarray
            The refraction vectors as an ND array of (x, y) pairs per
            wavelength describing the translation.

        """
        # To ensure we do not touch the template container, we work on a copy.
        template = copy.deepcopy(template)

        # We calculate the relative atmospheric refraction.
        relative_refraction = (
            lezargus.library.atmosphere.relative_atmospheric_refraction(
                wavelength=template.wavelength,
                reference_wavelength=self.reference_wavelength,
                zenith_angle=self.zenith_angle,
                temperature=self.temperature,
                pressure=self.pressure,
                water_pressure=self.ppw,
            )
        )

        # The relative refraction is the magnitude of the vector. The
        # direction of refraction is provided by the parallactic angle. We
        # use it to determine our vectors.
        x_refraction = relative_refraction * np.cos(self.parallactic_angle)
        y_refraction = relative_refraction * np.sin(self.parallactic_angle)

        # The refraction values are in angles, we convert it to pixels via the
        # pixel scales.
        if template.pixel_scale is None or template.slice_scale is None:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "No pixel or slice scale exists on the template, cannot"
                    " determine refraction angle to pixel conversion."
                ),
            )
            pixel_scale = 1
            slice_scale = 1
        else:
            pixel_scale = template.pixel_scale
            slice_scale = template.slice_scale
        x_pixel_refraction = x_refraction / pixel_scale
        y_pixel_refraction = y_refraction / slice_scale

        # We pair up the components of the refraction vector.
        refraction_vectors = np.array(
            [x_pixel_refraction, y_pixel_refraction],
        ).T
        return refraction_vectors

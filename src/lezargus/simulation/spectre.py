"""Simulation code for simulating SPECTRE observations.

For more information on the simulation of SPECTRE observations, see the
documentation file: [[TODO]].
"""

import astropy.constants
import numpy as np

import lezargus
from lezargus.library import hint
from lezargus.library import logging


class SimulatorSpectre:
    """Simulate a SPECTRE observation.

    We group all of the functions needed to simulate a SPECTRE observation
    into this class. It it easier to group all of data and needed functions
    this way.

    By default, all attributes are public to allow for maximum transparency.
    By convention, please treat the attributes as read-only. Consult the
    documentation for changing them. By default, they are None, this allows
    us to track the process of the simulation.

    Attributes
    ----------
    astrophysical_object_spectra : LezargusSpectra
        The "perfect" spectra of the astrophysical object who's observation is
        being modeled.
    astrophysical_object_cube : LezargusCube
        The cube form of the perfect astrophysical object who's observation is
        being modeled.
    astrophysical_object_cube_atm_trn : LezargusCube
        The astrophysical object after applying the atmospheric transmission.
    astrophysical_object_cube_atm_rad : LezargusCube
        The astrophysical object after applying the atmospheric radiance or
        emission after transmission.
    astrophysical_object_cube_atm_see : LezargusCube
        The astrophysical object after applying the atmospheric seeing
        convolution; after transmission and radiance.
    astrophysical_object_cube_atm_ref : LezargusCube
        The astrophysical object after applying the atmospheric refraction,
        after transmission, radiance, and seeing. This is actually just an
        alias for :py:attr:`astronomical_object_cube`.
    astronomical_object_cube : LezargusCube
        The astronomical object, obtained from applying atmospheric conditions
        to the astrophysical object. Noted as "astronomical" as it is
        considered "as-observed" from the Earth ground.
    """

    astrophysical_object_spectra = None
    astrophysical_object_cube = None
    astrophysical_object_cube_atm_trn = None
    astrophysical_object_cube_atm_rad = None
    astrophysical_object_cube_atm_see = None
    astrophysical_object_cube_atm_ref = None
    astronomical_object_cube = None

    def __init__(self: "SimulatorSpectre") -> None:
        """Instantiate the SPECTRE simulation class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

    def create_astrophysical_object_spectra(
        self: hint.Self,
        temperature: float,
        magnitude: float,
        filter_spectra: hint.LezargusSpectra,
        filter_zero_point: float,
    ) -> hint.LezargusSpectra:
        """Create the astrophysical object from first principles.

        This function creates and stores the astrophysical object spectra
        modeled as a blackbody of a specific temperature. If a custom spectra
        is to be provided, please see
        :py:meth:`custom_astrophysical_object_spectra`.
        The data is stored in this class internally as
        :py:attr:`astrophysical_object_spectra`.

        Parameters
        ----------
        temperature : float
            The temperature of the black body spectra.
        magnitude : float
            The magnitude of the object in the photometric filter system
            provided.
        filter_spectra : LezargusSpectra
            The filter transmission profile, packaged as a LezargusSpectra. It
            does not need to have any header data. We assume a Vega-based
            photometric system.
        filter_zero_point : float
            The zero point value of the filter.

        Returns
        -------
        spectra : LezargusSpectra
            The astrophysical object spectra; it is returned as a courtesy as
            the result is stored in this class.
        """
        # We need to construct our own wavelength base line, we rely on the
        # limits of SPECTRE itself.
        wavelength = np.linspace(
            lezargus.library.config.SPECTRE_SIMULATION_WAVELENGTH_MINIMUM,
            lezargus.library.config.SPECTRE_SIMULATION_WAVELENGTH_MAXIMUM,
            lezargus.library.config.SPECTRE_SIMULATION_WAVELENGTH_COUNT,
        )

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
        # Packaging the spectra.
        blackbody_spectra = lezargus.container.LezargusSpectra(
            wavelength=wavelength,
            data=integrated_blackbody_flux,
            uncertainty=None,
            wavelength_unit="m",
            data_unit="W m^-2 m^-1",
            mask=None,
            flags=None,
            header=None,
        )

        # We scale the flux to properly photometric scaling based on the
        # input filter and zero point values. For the purposes of the
        # simulation, we do not really care all too much about error
        # propagation as there is no way to communicate it for the output.
        calibration_factor, __ = (
            lezargus.library.photometry.calculate_photometric_correction_factor_vega(
                star_spectra=blackbody_spectra,
                filter_spectra=filter_spectra,
                star_magnitude=magnitude,
                filter_zero_point=filter_zero_point,
                star_magnitude_uncertainty=None,
                filter_zero_point_uncertainty=None,
            )
        )

        # Calibrating the flux.
        calibrated_flux = blackbody_spectra.data * calibration_factor

        # We convert the flux to a photon flux, dividing out the photon
        # energy.
        photon_energy = (
            astropy.constants.h * astropy.constants.c
        ).value / blackbody_spectra.wavelength
        photon_flux = calibrated_flux / photon_energy

        # Although we do not need a fully fledged header, we add some small
        # information where we know.
        header = {"LZI_INST": "SPECTRE", "LZO_NAME": "Simulation"}

        # Compiling the spectra class and storing it.
        self.astrophysical_object_spectra = lezargus.container.LezargusSpectra(
            wavelength=wavelength,
            data=photon_flux,
            uncertainty=None,
            wavelength_unit="m",
            data_unit="ph s^-1 m^-2 m^-1",
            mask=None,
            flags=None,
            header=header,
        )
        # All done.
        return self.astrophysical_object_spectra

    def custom_astrophysical_object_spectra(
        self: hint.Self,
        custom_spectra: hint.LezargusSpectra,
    ) -> hint.LezargusSpectra:
        """Use a provided spectra for a custom astrophysical object.

        This function is used to provide a custom spectra class to use to
        define the astrophysical object. If it should be derived instead from
        much simpler first principles, then please use
        :py:meth:`create_astrophysical_object_spectra` instead. The data is
        stored in this class internally as
        :py:attr:`astrophysical_object_spectra`.

        The object spectra should be a point source object. If you have a
        custom cube that you want to use, see
        :py:meth:`custom_astrophysical_object_cube` instead.

        Note that the wavelength axis of the custom spectra is used to define
        the wavelength scaling of the astrophysical object. We do not add
        any unknown information.


        Parameters
        ----------
        custom_spectra : LezargusSpectra
            The custom provided spectral object to use for the custom
            astrophysical object.

        Returns
        -------
        spectra : LezargusSpectra
            The astrophysical object spectra; it is returned as a courtesy as
            the result is stored in this class. This is the same as the input
            spectra and the return is for consistency.
        """
        # We really just use it as is, aside from a simple check to make sure
        # the input is not going to screw things up down the line.
        if not isinstance(custom_spectra, lezargus.container.LezargusSpectra):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The custom input spectra is not a LezargusSpectra"
                    f" instance but is instead has type {type(custom_spectra)}."
                ),
            )
        self.astrophysical_object_spectra = custom_spectra
        return self.astrophysical_object_spectra

    def generate_astrophysical_object_cube(
        self: hint.Self,
    ) -> hint.LezargusCube:
        """Use the stored astrophysical spectra to generate a field cube.

        This function takes the stored astrophysical object spectra and
        develops a mock field-of-view field of the object represented with a
        cube. A custom cube may be provided instead with
        :py:meth:`custom_astrophysical_object_cube`. The data is stored in
        this class internally as
        :py:attr:`astrophysical_object_cube`.

        The astrophysical object spectra is required to use this function,
        see :py:meth:`create_astrophysical_object_spectra` or
        :py:meth:`custom_astrophysical_object_spectra` to create it.

        Parameters
        ----------
        None

        Returns
        -------
        cube : LezargusCube
            The astrophysical object cube; it is returned as a courtesy as
            the result is stored in this class.
        """
        # We first need to make sure there is a spectra for us to use.
        if self.astrophysical_object_spectra is None:
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "There is no astrophysical object spectra to generate the"
                    " cube from, please create or provide one."
                ),
            )

        # From here, we determine the cube based on the configured parameters
        # defining the cube. We need to define a dummy cube before creating
        # the actual cube by broadcast.
        dummy_data_shape = (
            lezargus.library.config.SPECTRE_SIMULATION_FOV_E_W_COUNT,
            lezargus.library.config.SPECTRE_SIMULATION_FOV_N_S_COUNT,
            self.astrophysical_object_spectra.wavelength.size,
        )
        dummy_data_cube = np.empty(shape=dummy_data_shape)
        template_cube = lezargus.container.LezargusCube(
            wavelength=self.astrophysical_object_spectra.wavelength,
            data=dummy_data_cube,
            uncertainty=dummy_data_cube,
            wavelength_unit=self.astrophysical_object_spectra.wavelength_unit,
            data_unit=self.astrophysical_object_spectra.data_unit,
            mask=None,
            flags=None,
            header=None,
        )

        # We use this template cube to broadcast it to a center-pixel to
        # simulate a point source target.
        self.astrophysical_object_cube = (
            lezargus.container.broadcast.broadcast_spectra_to_cube_center(
                input_spectra=self.astrophysical_object_spectra,
                template_cube=template_cube,
                wavelength_mode="error",
                allow_even_center=True,
            )
        )
        # Just returning the cube as well.
        return self.astrophysical_object_cube

    def custom_astrophysical_object_cube(
        self: hint.Self,
        custom_cube: hint.LezargusCube,
    ) -> hint.LezargusCube:
        """Use a provided cube for a custom astrophysical cube.

        This function is used to provide a custom cube class to use to
        define the astrophysical object field. If it should be derived instead
        from a point-source spectra, then please use
        :py:meth:`generate_astrophysical_object_cube` instead. The data is
        stored in this class internally as
        :py:attr:`astrophysical_object_cube`.

        Note that the wavelength axis of the custom cube is used to define
        the wavelength scaling of the astrophysical object. We do not add
        any unknown information.


        Parameters
        ----------
        custom_cube : LezargusSpectra
            The custom provided spectral cube object to use for the custom
            astrophysical object field.

        Returns
        -------
        cube : LezargusCube
            The astrophysical object cube; it is returned as a courtesy as
            the result is stored in this class. This is the same as the input
            spectra and the return is for consistency.
        """
        # We really just use it as is, aside from a simple check to make sure
        # the input is not going to screw things up down the line.
        if not isinstance(custom_cube, lezargus.container.LezargusCube):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The custom input cube is not a LezargusCube instance but"
                    f" is instead has type {type(custom_cube)}."
                ),
            )
        self.astrophysical_object_cube = custom_cube
        return self.astrophysical_object_cube

    @classmethod
    def prepare_spectra(
        cls: hint.Type[hint.Self],
        spectra: hint.LezargusSpectra,
        *args: object,
        skip_convolve: bool = False,
        **kwargs: object,
    ) -> hint.LezargusSpectra:
        """Prepare the provided spectra for future steps.

        Any provided spectra (transmission curves, emission curves, etc) must
        be properly prepared before its application to the simulation data.
        We do the following steps in order (if not otherwise skipped):

            - Convolve: We match the spectral resolution (or resolving power)
            to the simulation's. We leverage
            :py:meth:`_prepare_convolve_atmospheric_transmission`.

        Please see the linked functions in each of the steps for the parameters
        required for each step of the preparation, if it is not to be skipped.
        Without the required inputs, the preparation will likely fail; failure
        will likely be noisy (logged or raised).

        Parameters
        ----------
        spectra : LezargusSpectra
            The input spectra which we will be preparing.
        skip_convolve : bool, default = False
            If True, we skip the resolution convolution step. The backend
            function will not be called.
        *args : Any
            The positional arguments. We forbid any positional arguments for
            informing the backend functions because of its ambiguity.
        **kwargs : Any
            The keyword arguments which will be fed into the backend functions.

        Returns
        -------
        finished_spectra : LezargusSpectra
            The finished prepared spectra after all of the steps have been
            done.
        """
        # Type check on the input spectra.
        if not isinstance(spectra, lezargus.container.LezargusSpectra):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input spectra is not a LezargusSpectra, is instead:"
                    f" {type(spectra)}"
                ),
            )

        # There should be no positional arguments.
        if len(args) != 0:
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    "Spectra preparation cannot have positional arguments, use"
                    " keyword  arguments."
                ),
            )

        # Now, we just go down the list making sure that we do all of the
        # procedures in order, unless the user wants it skipped.
        # ...convolution...
        if skip_convolve:
            convolved_spectra = spectra
        else:
            convolved_spectra = cls._prepare_convolve_spectra(
                spectra=spectra,
                **kwargs,
            )

        # All done.
        finished_spectra = convolved_spectra
        return finished_spectra

    @classmethod
    def _prepare_convolve_spectra(
        cls: hint.Type[hint.Self],
        spectra: hint.LezargusSpectra,
        input_resolution: float | None = None,
        input_resolving: float | None = None,
        simulation_resolution: float | None = None,
        simulation_resolving: float | None = None,
        reference_wavelength: float | None = None,
        **kwargs: object,
    ) -> hint.LezargusSpectra:
        """Convolve the input spectra to make its resolution match.

        Spectra comes in many resolutions. If the resolution of an input
        spectra is too high for the simulation, its application can give
        erroneous results. Here, we use a Gaussian kernel to convolve the
        spectral data to better match the resolution of the input and the
        simulation.

        We leverage :py:func:`kernel_1d_gaussian_resolution` to make the kernel.

        Parameters
        ----------
        spectra : LezargusSpectra
            The transmission spectra which we will be preparing.
        input_resolution : float, default = None
            The spectral resolution of the input spectra. Must be in
            the same units as the spectra.
        input_resolving : float, default = None
            The spectral resolving power of the input spectra, relative
            to the wavelength `reference_wavelength`.
        simulation_resolution : float, default = None
            The spectral resolution of the simulation spectra. Must be in
            the same units as the simulation spectra.
        simulation_resolving : float, default = None
            The spectral resolving power of the simulation spectra, relative
            to the wavelength `reference_wavelength`.
        reference_wavelength : float, default = None
            The reference wavelength for any needed conversion.
        **kwargs : dict
            Keyword argument catcher.

        Returns
        -------
        convolved_spectra : LezargusSpectra
            The spectra, after convolution based on the input parameters.
        """
        # This is just to catch and use the keyword arguments.
        __ = kwargs

        # We assume the kernel size based on the wavelength of the input
        # spectra. Namely, the kernel must be smaller than the number of points.
        # We assume that we have Nyquist sampling and 1 extra degree of
        # freedom.
        reduction_factor = 2 * 2
        kernel_size = int(np.ceil(len(spectra.wavelength) / reduction_factor))
        kernel_shape = (kernel_size,)

        # We have the input, we rely on the kernel determination to figure out
        # the mode.
        gaussian_kernel = (
            lezargus.library.convolution.kernel_1d_gaussian_resolution(
                shape=kernel_shape,
                template_wavelength=spectra.wavelength,
                base_resolution=input_resolution,
                target_resolution=simulation_resolution,
                base_resolving_power=input_resolving,
                target_resolving_power=simulation_resolving,
                reference_wavelength=reference_wavelength,
            )
        )

        # We then convolve the input spectra.
        convolved_spectra = spectra.convolve(kernel=gaussian_kernel)

        # All done.
        return convolved_spectra

    def apply_atmospheric_transmission(
        self: hint.Self,
        transmission_spectra: hint.LezargusSpectra,
    ) -> hint.LezargusCube:
        """Apply the atmospheric transmission to the object.

        The astrophysical object cube is required to use this function,
        see :py:meth:`create_astrophysical_object_cube` or
        :py:meth:`custom_astrophysical_object_cube` to create it.

        Moreover, consider using :py:meth:`prepare_atmospheric_transmission`
        to properly match the resolving power or resolution of the simulation
        spectra and the transmission spectra.

        Parameters
        ----------
        transmission_spectra : LezargusSpectra
            The atmospheric transmission spectra. The wavelength unit of
            this spectra should be meters.

        Returns
        -------
        cube : LezargusCube
            The cube of the object after atmospheric transmission has been
            applied.
        """
        # We first need to make sure there is the object cube for us to use.
        if self.astrophysical_object_cube is None:
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "There is no astrophysical object spectra to generate the"
                    " cube from, please create or provide one."
                ),
            )

        # We also need to make sure the transmission spectra is a
        # LezargusSpectra.
        if not isinstance(
            transmission_spectra,
            lezargus.container.LezargusSpectra,
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The atmospheric transmission spectra has type"
                    f" {type(transmission_spectra)}, not the expected"
                    " LezargusSpectra."
                ),
            )

        # We need to align the transmission spectra to the Simulators
        # wavelength base.
        trans_wave = self.astrophysical_object_cube.wavelength
        trans_data, trans_uncert, trans_mask, trans_flags = (
            transmission_spectra.interpolate(
                wavelength=trans_wave,
                skip_flags=True,
                skip_mask=True,
            )
        )
        # It is convenient to reconstruct a spectra for it.
        aligned_transmission_spectra = lezargus.container.LezargusSpectra(
            wavelength=trans_wave,
            data=trans_data,
            uncertainty=trans_uncert,
            wavelength_unit=transmission_spectra.wavelength_unit,
            data_unit=transmission_spectra.data_unit,
            mask=trans_mask,
            flags=trans_flags,
            header=transmission_spectra.header,
        )

        # We then pad this spectra out to a cube for us to apply across the
        # board.
        aligned_transmission_cube = (
            lezargus.container.broadcast.broadcast_spectra_to_cube_uniform(
                input_spectra=aligned_transmission_spectra,
                template_cube=self.astrophysical_object_cube,
            )
        )

        # Applying the transmission is simple multiplication.
        self.astrophysical_object_cube_atm_trn = (
            self.astrophysical_object_cube * aligned_transmission_cube
        )
        # All done.
        return self.astrophysical_object_cube_atm_trn

    def apply_atmospheric_radiance(
        self: hint.Self,
        transmission_spectra: hint.LezargusSpectra,
    ) -> hint.LezargusCube:
        """Apply the atmospheric transmission to the object.

        The astrophysical object cube is required to use this function,
        see :py:meth:`create_astrophysical_object_cube` or
        :py:meth:`custom_astrophysical_object_cube` to create it.

        Moreover, consider using :py:meth:`prepare_atmospheric_transmission`
        to properly match the resolving power or resolution of the simulation
        spectra and the transmission spectra.

        Parameters
        ----------
        transmission_spectra : LezargusSpectra
            The atmospheric transmission spectra. The wavelength unit of
            this spectra should be meters.

        Returns
        -------
        cube : LezargusCube
            The cube of the object after atmospheric transmission has been
            applied.
        """
        # We first need to make sure there is the object cube for us to use.
        if self.astrophysical_object_cube is None:
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "There is no astrophysical object spectra to generate the"
                    " cube from, please create or provide one."
                ),
            )

        # We also need to make sure the transmission spectra is a
        # LezargusSpectra.
        if not isinstance(
            transmission_spectra,
            lezargus.container.LezargusSpectra,
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The atmospheric transmission spectra has type"
                    f" {type(transmission_spectra)}, not the expected"
                    " LezargusSpectra."
                ),
            )

        # We need to align the transmission spectra to the Simulators
        # wavelength base.
        trans_wave = self.astrophysical_object_cube.wavelength
        trans_data, trans_uncert, trans_mask, trans_flags = (
            transmission_spectra.interpolate(
                wavelength=trans_wave,
                skip_flags=True,
                skip_mask=True,
            )
        )
        # It is convenient to reconstruct a spectra for it.
        aligned_transmission_spectra = lezargus.container.LezargusSpectra(
            wavelength=trans_wave,
            data=trans_data,
            uncertainty=trans_uncert,
            wavelength_unit=transmission_spectra.wavelength_unit,
            data_unit=transmission_spectra.data_unit,
            mask=trans_mask,
            flags=trans_flags,
            header=transmission_spectra.header,
        )

        # We then pad this spectra out to a cube for us to apply across the
        # board.
        aligned_transmission_cube = (
            lezargus.container.broadcast.broadcast_spectra_to_cube_uniform(
                input_spectra=aligned_transmission_spectra,
                template_cube=self.astrophysical_object_cube,
            )
        )

        # Applying the transmission is simple multiplication.
        self.astrophysical_object_cube_atm_trn = (
            self.astrophysical_object_cube * aligned_transmission_cube
        )
        # All done.
        return self.astrophysical_object_cube_atm_trn

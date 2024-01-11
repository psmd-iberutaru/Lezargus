"""Simulation code for simulating SPECTRE observations.

For more information on the simulation of SPECTRE observations, see the
documentation file: [[TODO:LINK]].
"""

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
    """

    astrophysical_object_spectra = None
    astrophysical_object_cube = None

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
        self: "SimulatorSpectre",
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
        blackbody_spectra = lezargus.container.LezargusSpectra(
            wavelength=wavelength,
            data=blackbody_flux,
            uncertainty=None,
            wavelength_unit="um",
            data_unit="flam",
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
        calibrated_flux = blackbody_flux * calibration_factor

        # Although we do not need a fully fledged header, we add some small
        # information where we know.
        header = {"LZI_INST": "SPECTRE", "LZO_NAME": "Simulation"}

        # Compiling the spectra class and storing it.
        self.astrophysical_object_spectra = lezargus.container.LezargusSpectra(
            wavelength=wavelength,
            data=calibrated_flux,
            uncertainty=None,
            wavelength_unit="um",
            data_unit="flam",
            mask=None,
            flags=None,
            header=header,
        )
        # All done.
        return self.astrophysical_object_spectra

    def custom_astrophysical_object_spectra(
        self: "SimulatorSpectre",
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

    def generate_astrophysical_object_cube(self: "SimulatorSpectre") -> None:
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
            lezargus.library.config.SPECTRE_SIMULATION_FOV_ZONAL_COUNT,
            lezargus.library.config.SPECTRE_SIMULATION_FOV_MERIDIONAL_COUNT,
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
        self: "SimulatorSpectre",
        custom_cube: hint.LezargusCube,
    ) -> None:
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

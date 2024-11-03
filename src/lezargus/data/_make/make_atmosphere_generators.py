"""Make functions to create the AtmosphereSpectrumGenerator for the atmosphere.

This module is just the part of the data making procedure to make the
atmospheric generators. Only a specific subset of atmospheric conditions are
supported. Open a new issue to add more conditions or scenarios if desired.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.table
import numpy as np

import lezargus
from lezargus.data._make import functionality


def make_atmosphere_transmission_generator(
    basename: str,
) -> hint.AtmosphereSpectrumGenerator:
    """Load a atmospheric transmission file to make the generator object.

    Note, the format of the atmospheric transmission generator file is
    very specific. User usage of the this function is discouraged.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the atmospheric transmission
        generator. The paths are handled automatically.

    Returns
    -------
    transmission_generator : AtmosphereSpectrumGenerator
        The atmospheric transmission generator.

    """
    # The PSG atmospheric files are generated outside of this package and so
    # the defined zenith angles and precipitable water vapor values are known
    # before hand. The values here are defined based on the filenames and
    # should be valid for both transmission and radiance.
    zenith_angles_degree = np.array([0, 30, 45, 60])
    pwv = np.array([0.5, 1.0, 2.0, 3.0])

    # PSG atmospheric files have a common estimated spectral resolution.
    psg_spectral_scale = 1e-9

    # Loading the transmission file.
    filename = functionality.parse_basename(basename=basename)
    transmission_table = astropy.table.Table.read(filename, format="ascii.mrt")

    # The domain is the zenith angles, PWV, and wavelength. The filenames use
    # angular degrees while the generator uses radians.
    wavelength = np.asarray(transmission_table["wavelength"])
    wavelength_unit = "m"
    zenith_angle_radians = np.deg2rad(zenith_angles_degree)

    # We package the transmission data so that it matches what the generator
    # expects.
    transmission_shape = (
        wavelength.size,
        len(zenith_angles_degree),
        len(pwv),
    )
    transmission_data = np.empty(transmission_shape, dtype=float)
    transmission_data_unit = ""
    for zindex, zenithdex in enumerate(zenith_angles_degree):
        for pindex, pwvdex in enumerate(pwv):
            column_name = f"za{zenithdex}_pwv{pwvdex}"
            transmission_data[:, zindex, pindex] = transmission_table[
                column_name
            ]

    # The PWV values are provided as millimeters in the files and filenames.
    # However, we use SI in this module so we need to convert.
    pwv_si = pwv / 1000

    # Creating the atmospheric transmission generator. We then add it to the
    # data module.
    transmission_generator = (
        lezargus.library.container.AtmosphereSpectrumGenerator(
            wavelength=wavelength,
            zenith_angle=zenith_angle_radians,
            pwv=pwv_si,
            data=transmission_data,
            wavelength_unit=wavelength_unit,
            data_unit=transmission_data_unit,
            spectral_scale=psg_spectral_scale,
        )
    )
    return transmission_generator


def make_atmosphere_radiance_generator(
    basename: str,
) -> hint.AtmosphereSpectrumGenerator:
    """Load a atmospheric radiance file to make the generator object.

    Note, the format of the atmospheric radiance generator file is
    very specific. User usage of the this function is discouraged.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the atmospheric radiance
        generator. The paths are handled automatically.

    Returns
    -------
    radiance_generator : AtmosphereSpectrumGenerator
        The atmospheric radiance generator.

    """
    # The PSG atmospheric files are generated outside of this package and so
    # the defined zenith angles and precipitable water vapor values are known
    # before hand. The values here are defined based on the filenames and
    # should be valid for both transmission and radiance.
    zenith_angles_degree = np.array([0, 30, 45, 60])
    pwv = np.array([0.5, 1.0, 2.0, 3.0])

    # PSG atmospheric files have a common estimated spectral resolution.
    psg_spectral_scale = 1e-9

    # Loading the transmission file.
    filename = functionality.parse_basename(basename=basename)
    radiance_table = astropy.table.Table.read(filename, format="ascii.mrt")

    # The domain is the zenith angles, PWV, and wavelength. The filenames use
    # angular degrees while the generator uses radians.
    wavelength = np.asarray(radiance_table["wavelength"])
    wavelength_unit = "m"
    zenith_angle_radians = np.deg2rad(zenith_angles_degree)

    # We package the radiance data so that it matches what the generator
    # expects.
    radiance_shape = (
        wavelength.size,
        len(zenith_angles_degree),
        len(pwv),
    )
    radiance_data = np.empty(radiance_shape, dtype=float)
    radiance_data_unit = "W m^-2 sr^-1 m^-1"
    for zindex, zenithdex in enumerate(zenith_angles_degree):
        for pindex, pwvdex in enumerate(pwv):
            column_name = f"za{zenithdex}_pwv{pwvdex}"
            radiance_data[:, zindex, pindex] = radiance_table[column_name]

    # The PWV values are provided as millimeters in the files and filenames.
    # However, we use SI in this module so we need to convert.
    pwv_si = pwv / 1000

    # Creating the atmospheric radiance generator. We then add it to the
    # data module.
    radiance_generator = lezargus.library.container.AtmosphereSpectrumGenerator(
        wavelength=wavelength,
        zenith_angle=zenith_angle_radians,
        pwv=pwv_si,
        data=radiance_data,
        wavelength_unit=wavelength_unit,
        data_unit=radiance_data_unit,
        spectral_scale=psg_spectral_scale,
    )
    return radiance_generator

"""Simulation code to simulate the telescope properties.

We simulate telescope effects, primarily the emission and reflectivity aspects
of it. We break this module up so that we can potentially simulate different
telescopes other than the IRTF. This is unlikely, but who knows.
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


class IRTFSimulator:
    """The NASA IRTF telescope simulation class.

    Here we implement the effects of the primary and secondary mirror of the
    NASA IRTF telescope. Most focus is to the emissive and reflectivity effects
    of the mirrors, but other effects may also be simulated here.

    Attributes
    ----------
    _primary_reflectivity_interpolator : Spline1DInterpolate
        The interpolation class for the primary mirror reflectivity.
    _secondary_reflectivity_interpolator : Spline1DInterpolate
        The interpolation class for the secondary mirror reflectivity.

    """

    def __init__(self: IRTFSimulator) -> None:
        """Create an instance of the IRTF telescope.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

    @property
    def telescope_area(self: hint.Self) -> float:
        """Light collecting area of the telescope.

        Parameters
        ----------
        None

        Returns
        -------
        area : float
            The effective light collecting area of the telescope.

        """
        # We determine it from the configured radius.
        irtf_radius = lezargus.config.OBSERVATORY_IRTF_TELESCOPE_RADIUS
        area = np.pi * irtf_radius**2
        return area

    def primary_reflectivity(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """Compute the reflectivity of the IRTF primary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            reflectivity.

        Returns
        -------
        reflectivity : NDArray
            The reflectivity of the primary mirror at the wavelengths provided.

        """

    def primary_emission(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """Compute the spectral flux emission of the IRTF primary mirror.

        Parameters
        ----------
        wavelength : NDArray
            The wavelengths which we will compute the primary mirror
            spectral flux emission.

        Returns
        -------
        emission : NDArray
            The spectral flux emission of the primary mirror at the
        wavelengths provided.

        """

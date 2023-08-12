"""Containers for data.

This module contains the containers for spectral data. We have 4 main classes,
broken into different files for ease. There is a parent class which we use
to define connivent arithmetic.
"""

# The parent class used to properly handle the arithmetic of spectra
# and data cubes.
from lezargus.container.cube import LezargusCube
from lezargus.container.image import LezargusImage
from lezargus.container.mosaic import LezargusMosaic
from lezargus.container.parent import LezargusContainerArithmetic
from lezargus.container.spectra import LezargusSpectra

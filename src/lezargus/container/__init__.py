"""Containers for data.

This module contains the containers to hold different data. We have 4 major
classes for <=3-dimensional data, broken into different files for ease.
There is a parent class which we use to define connivent arithmetic.

Aside from the 4 major classes, there are other minor containers with more
specific purposes.
"""

# The parent class used to properly handle the arithmetic of spectrum
# and data cubes.
from lezargus.container.parent import LezargusContainerArithmetic

# isort: split

# The major classes for dimensional data.
from lezargus.container.cube import LezargusCube
from lezargus.container.image import LezargusImage
from lezargus.container.mosaic import LezargusMosaic
from lezargus.container.spectrum import LezargusSpectrum

# isort: split

# The minor classes for dimensional data, with a very specific implementation
# details.
from lezargus.container.atmosphere import AtmosphereSpectrumGenerator

# isort: split

# Extra and utility functions which operate on containers. Though user facing,
# it is more likely the user would not directly use these functions.
from lezargus.container import function

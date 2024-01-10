"""Containers for data.

This module contains the containers for spectral data. We have 4 main classes,
broken into different files for ease. There is a parent class which we use
to define connivent arithmetic.
"""

# The parent class used to properly handle the arithmetic of spectra
# and data cubes.
from lezargus.container.parent import LezargusContainerArithmetic

# isort: split

# And all of the following data structures for any given N-dimensional data.
from lezargus.container.cube import LezargusCube
from lezargus.container.image import LezargusImage
from lezargus.container.mosaic import LezargusMosaic
from lezargus.container.spectra import LezargusSpectra

# isort: split

# Other inner packages. Although easily accessible by the user, the depth of
# these packages and functions, compared to the main containers, implies that
# we should still abstract it away from the end-user.
from lezargus.container import broadcast

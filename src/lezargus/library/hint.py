"""A collection of types for linting.

These are redefinitions and wrapping variables for type hints. Its purpose
is for just uniform hinting types.

This should only be used for types which are otherwise not native and would
require an import, including the typing module. The whole point of this is to
be a central collection of types for the purpose of type hinting.

This module should never be used for anything other than hinting. Use proper
imports to access these classes. Otherwise, you will likely get circular
imports and other nasty things.
"""

# pylint: disable=W0401,W0611,W0614

from argparse import ArgumentParser
from argparse import Namespace
from collections import *
from collections.abc import *
from logging import *
from subprocess import CompletedProcess
from typing import *

from astropy.io.fits import FITS_rec
from astropy.io.fits import Header
from astropy.table import Row
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.backend_bases import MouseEvent
from numpy import generic as numpy_generic

# Arrays. This is done because ArrayLike casts a rather larger union
# documentation.
from numpy import ndarray
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike

# The windows.
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# Derived types and aliases.
Array = ndarray
Widget = QtWidgets.QWidget
Window = QtWidgets.QMainWindow

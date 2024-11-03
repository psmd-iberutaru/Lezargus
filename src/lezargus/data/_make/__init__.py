"""Make functions to create data files and insatiate the data classes.

We need to convert the plain text data to the more helpful containers found
in the library. In this module we just store a lot of the code to do that.
"""

# Atmospheric generator make functions.
from lezargus.data._make.make_atmosphere_generators import (
    make_atmosphere_radiance_generator,
)
from lezargus.data._make.make_atmosphere_generators import (
    make_atmosphere_transmission_generator,
)

# Efficiency function spectrum make functions.
from lezargus.data._make.make_optic_efficiencies import (
    make_irtf_primary_efficiency,
)
from lezargus.data._make.make_optic_efficiencies import (
    make_irtf_secondary_efficiency,
)
from lezargus.data._make.make_optic_efficiencies import make_optic_efficiency

# Photometric filter make functions.
from lezargus.data._make.make_photometric_filters import (
    make_ab_photometric_filter,
)
from lezargus.data._make.make_photometric_filters import (
    make_vega_photometric_filter,
)

# Standard star spectrum make functions.
from lezargus.data._make.make_standard_spectra import make_standard_spectrum

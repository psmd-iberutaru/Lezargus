"""Storage place for all of the configuration parameters in Python.

The configuration file yaml is generated from this module, on the fly, by
converting it to a valid yaml file.

Notice! Users: Do not edit this file as a Python file.
"""

import os

# We begin the actual configuration file after this line. Just helpful to
# notate it for the automatic yaml generation.
# fmt: off
#############################################################################
#############################################################################
# <BEGIN ALL>

# These are configurable parameters for running Lezargus.
# You should pass your own configuration.yaml file when running this software.


# <BEGIN META>
# # # Lezargus Meta Parameter Configuration
# # # # # # # # # # # # # # # # # # # #

META_CONTAINER_FLOAT_DATA_TYPE = "float"

# # # # # # # # # # # # # # # # # # # #
# </END META>




# <BEGIN TEMPORARY>
# # # Lezargus Temporary Directory Configuration
# # # # # # # # # # # # # # # # # # # #

# A temporary directory is needed to store large data during processing.
# The path here is the directory which will be created and destroyed as needed
# as the Lezargus temporary directory. An initial temporary file serves as a
# marker flag that a directory is temporary; the extension is automatic.
LEZARGUS_TEMPORARY_DIRECTORY = "./lz_temp/"
LEZARGUS_TEMPORARY_DIRECTORY_FLAG_FILENAME = "lezargus_temporary_directory"

# If set to True, we overwrite the temporary directory regardless of what
# is currently inside of it.
LEZARGUS_TEMPORARY_OVERWRITE_DIRECTORY = False

# We usually clean up temporary directory by deleting it afterwards. Set to
# True to skip the deletion. We also make specific checks to ensure we are not
# deleting anything bad by mistake, but the deletion can be forced to skip the
# checks.
LEZARGUS_TEMPORARY_SKIP_DELETION = False
LEZARGUS_TEMPORARY_FORCE_DELETION = False

# If set to True, we do not delete the temporary directory upon program
# termination. We make no guarantees of the state of the temporary directory.
LEZARGUS_TEMPORARY_SKIP_DELETION = False

# # # # # # # # # # # # # # # # # # # #
# </END TEMPORARY>



# <BEGIN LOGGING>
# # # Lezargus Logging Configuration
# # # # # # # # # # # # # # # # # # # #

# To prevent multiple console handlers being created when Lezargus is
# initialized multiple times, we check for console based handlers. In order to
# identify the console handler, we assign it a rather unique name defined here.
LOGGING_SPECIFIC_CONSOLE_HANDLER_FLAG_NAME = "Lezargus818LoggerConsoleHandler"

# By default, warnings and errors do not raise exceptions, only critical does.
# Change these flags to elevate them to critical problems.
LOGGING_ELEVATE_WARNING_TO_CRITICAL = False
LOGGING_ELEVATE_ERROR_TO_CRITICAL = False

# This is the string specifier for how the log records should be formatted.
# The formatting specification is detailed in...
# https://docs.python.org/3/library/logging.html#logging.Formatter
LOGGING_RECORD_FORMAT_STRING = "[Lezargus] %(asctime)s %(levelname)8s -- %(message)s"
LOGGING_DATETIME_FORMAT_STRING = "%Y-%m-%dT%H:%M:%S"

# By default, we add the exception type (of warning, error, or critical) as a
# prefix to the logging error message. Turn to False to disable this.
LOGGING_INCLUDE_EXCEPTION_TYPE_IN_MESSAGE = True

# And, the HEX colors for the stream logs. The HEX values must have the
# preceding hash/"#" symbol. The HEX values are likely case-insensitive, but
# upper case is usually nicer.
LOGGING_STREAM_USE_COLOR = True
LOGGING_STREAM_DEBUG_COLOR_HEX = "#FFFFFF"
LOGGING_STREAM_INFO_COLOR_HEX = "#004488"
LOGGING_STREAM_WARNING_COLOR_HEX = "#FFE100"
LOGGING_STREAM_ERROR_COLOR_HEX = "#FF7900"
LOGGING_STREAM_CRITICAL_COLOR_HEX = "#BD2024"

# # # # # # # # # # # # # # # # # # # #
# </END LOGGING>


# <BEGIN OBSERVATORY>
# # # Observatory and Environmental Configuration
# # # # # # # # # # # # # # # # # # # #

# Here local atmospheric climate conditions of the observatory:
# The temperature (Kelvin), pressure (Pascal), and partial pressure of
# water (Pascal). These values are usually for simulation, however, it is
# less a property of the instrument and more a global facility property so
# we have it here.
OBSERVATORY_ATMOSPHERE_TEMPERATURE = 274
OBSERVATORY_ATMOSPHERE_PRESSURE = 0
OBSERVATORY_ATMOSPHERE_PARTIAL_PRESSURE_WATER = 0

# Here, the local conditions of the IRTF telescope, if used. If the IRTF
# telescope is not used, these parameters should not affect the operation
# of the software.
# The telescope radius is the radius of the IRTF primary mirror (meters).
# The temperatures are of the primary and secondary mirror (Kelvin).
OBSERVATORY_IRTF_PRIMARY_MIRROR_RADIUS = 1.6
OBSERVATORY_IRTF_SECONDARY_MIRROR_RADIUS = 0.8
OBSERVATORY_IRTF_PRIMARY_TEMPERATURE = 274
OBSERVATORY_IRTF_SECONDARY_TEMPERATURE = 274



# # # # # # # # # # # # # # # # # # # #
# </END OBSERVATORY>



# <BEGIN SIMULATION>
# # # SPECTRE Simulation Instrument Configuration
# # # # # # # # # # # # # # # # # # # #


# The perfect wavelength bounds of the astrophysical object. This is used for
# the simulation when building the wavelength array. Per the unit convention,
# the wavelength units here are in meters. We use a uniform distribution with
# the number of "count" data points.
SPECTRE_SIMULATION_WAVELENGTH_MINIMUM = 0.35e-6
SPECTRE_SIMULATION_WAVELENGTH_MAXIMUM = 4.20e-6
SPECTRE_SIMULATION_WAVELENGTH_COUNT = 2000

# The perfect field of view of the astrophysical target. This is used for the
# simulation when building the perfect data cube representing the object
# itself. Namely, the field of view length across the N-S and E-W axes are
# set here. The length units here are in arcseconds on the sky. The square
# resolution of the perfect field of view is based on the "count", data points
# across for each axis.
# Note that because we usually assume point sources, the meridional and zonal
# point count should be odd. Although possible with an even count, a warning
# is raised due to accuracy concerns.
SPECTRE_SIMULATION_FOV_N_S_LENGTH = 10
SPECTRE_SIMULATION_FOV_E_W_LENGTH = 10
SPECTRE_SIMULATION_FOV_N_S_COUNT = 201
SPECTRE_SIMULATION_FOV_E_W_COUNT = 201

# # # # # # # # # # # # # # # # # # # #
# </END SIMULATION>


# </END ALL>
#############################################################################
#############################################################################
# fmt: on
# We end the actual configuration file before this line. Just helpful to
# notate it for the automatic yaml generation.


# <BEGIN INTERNAL>
# # # Lezargus internal configuration.
# # # # # # # # # # # # # # # # # # # #

# This is the internal configuration parameters of Lezargus, do not modify.

# The default path which this module is installed in. It is one higher than
# this file which is within the library module of the Lezargus install.
INTERNAL_MODULE_INSTALLATION_PATH = os.path.dirname(
    os.path.realpath(os.path.join(os.path.realpath(__file__))),
)

# We need to get the actual directory of the data.
INTERNAL_MODULE_DATA_FILE_DIRECTORY = os.path.join(
    INTERNAL_MODULE_INSTALLATION_PATH,
    "data",
    "_files",
)

# <BEGIN DEBUG>
# # # SPECTRE Simulation Instrument Configuration
# # # # # # # # # # # # # # # # # # # #

# Due to the nature of loading data files, sometimes, loading data files needs
# to be disabled when creating the data files themselves. This parameter means
# nothing if changed during runtime. This should be False in most cases.
INTERNAL_DEBUG_SKIP_LOADING_DATA_FILES = False

# # # # # # # # # # # # # # # # # # # #
# </END DEBUG>

# # # # # # # # # # # # # # # # # # # #
# </END INTERNAL>
